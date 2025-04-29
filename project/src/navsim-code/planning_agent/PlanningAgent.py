import torch
import os
import glob
import re
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from PIL import Image
import numpy as np
import numpy.typing as npt
from pathlib import Path
# Keep transformers imports, but loading is conditional
from transformers import AutoProcessor, AutoModel
from typing import Dict, List, Union, Tuple, Any, Optional

import pytorch_lightning as pl
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

# Correct import for TrajectorySampling
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import (
    AgentInput,
    Trajectory,
    EgoStatus,
    SensorConfig,
    Scene,
)

# Assuming convert_absolute_to_relative_se2_array is available, e.g., from a utils module
from nuplan.common.actor_state.state_representation import StateSE2
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)

from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

def rel_to_abs(pred_rel, gt_abs):
    """
    Converts relative trajectory predictions to absolute using the first GT pose as origin.
    Assumes inputs are (B, T, 3) tensors.
    """
    origin = gt_abs[:, 0:1, :]             # (B,1,3) - extracts the first pose for each batch item
    return pred_rel + origin

import pytorch_lightning as pl

class FirstBatchDebugger(pl.Callback):
    """
    A callback to print debug information about the first batch's inputs and outputs.
    """
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Only print on the very first optimisation step
        if trainer.global_step == 0 and batch_idx == 0:
            print("\n" + "="*30 + " FIRST BATCH DEBUG START " + "="*30)
            features, targets = batch

            # We need the agent instance to call forward
            agent = pl_module # In the PTL loop, pl_module *is* the agent

            # ----- ground-truth (absolute) -----
            # Ensure target is moved to the correct device for rel_to_abs if needed
            # Although it's a callback, let's just fetch CPU numpy for printing
            # In compute_loss, tensors are moved to device.
            try:
                gt_abs = targets["trajectory_gt"][0].cpu().numpy()        # (T,3)
                print("\n[DEBUG] FIRST-BATCH  GT  (first 3 poses):", gt_abs[:3].round(3))
            except Exception as e:
                 print(f"[DEBUG] Could not access GT target: {e}")


            # ----- model output (relative) -----
            try:
                with torch.no_grad():
                    # Pass features on the correct device, assuming batch is on device
                    device = next(agent.parameters()).device # Get agent's device
                    # Make a copy and move to device defensively, in case batch wasn't fully moved
                    features_on_device = {k: v.to(device) for k, v in features.items()}
                    pred_rel = agent.forward(features_on_device)["trajectory"][0]  # (T,3)
                    pred_rel = pred_rel.cpu().numpy()
                print("[DEBUG] FIRST-BATCH  PRED (relative, first 3):", pred_rel[:3].round(3))

                # ----- convert to absolute for easy comparison -----
                if 'gt_abs' in locals(): # Only convert if GT was successfully loaded
                    origin = gt_abs[0:1, :]                    # (1,3)  current pose
                    pred_abs = rel_to_abs(torch.from_numpy(pred_rel), torch.from_numpy(gt_abs).unsqueeze(0))[0].cpu().numpy() # Convert back and use rel_to_abs logic
                    print("[DEBUG] FIRST-BATCH  PRED (ABS, first 3):", pred_abs[:3].round(3))
                else:
                     print("[DEBUG] Skipping absolute conversion - GT not available.")

            except Exception as e:
                print(f"[DEBUG] Error during agent forward pass in debugger: {e}")

            print("="*30 + " FIRST BATCH DEBUG END " + "="*30 + "\n")

class CameraImageFeatureBuilder(AbstractFeatureBuilder):
    """
    Feature builder for extracting the front camera image.
    Returns the raw image data as a tensor (CHW, float).
    """
    def get_unique_name(self) -> str:
        return "front_camera_image"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        # Check if cameras list is non-empty and has cam_f0 data
        # The agent's forward method expects this key to determine batch size/device,
        # even if no visual encoder is used.
        if not agent_input.cameras or len(agent_input.cameras) == 0 or \
           not hasattr(agent_input.cameras[-1], "cam_f0") or \
           agent_input.cameras[-1].cam_f0.image is None:
            print(f"Warning: {self.get_unique_name()}: Front camera image missing. Returning zero placeholder (3, 224, 224).")
            # Return a standard shape [C, H, W] float [0,1] placeholder
            dummy_image = np.zeros((3, 224, 224), dtype=np.float32)
            return {self.get_unique_name(): torch.from_numpy(dummy_image)}

        # Assuming the relevant camera is the last one in the history list
        # Convert HWC uint8 [0,255] to CHW float [0,1]
        image_np = agent_input.cameras[-1].cam_f0.image
        if image_np.ndim == 3 and image_np.shape[2] == 3:
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        else:
             print(f"Warning: {self.get_unique_name()}: Unexpected image shape {image_np.shape}. Returning zero placeholder.")
             dummy_image = np.zeros((3, 224, 224), dtype=np.float32)
             image_tensor = torch.from_numpy(dummy_image)


        return {self.get_unique_name(): image_tensor}

class EgoFeatureBuilder(AbstractFeatureBuilder):
    """
    Feature builder for extracting ego status features (velocity, acceleration, driving command).
    Formats the driving command into a one-hot vector.
    """
    NUM_DRIVING_COMMANDS = 4 # Example based on typical planning commands
    # Corresponds to EGO_DIM = 2 (vel) + 2 (accel) + 4 (command one-hot) = 8

    def get_unique_name(self) -> str:
        return "ego_features"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        # Use the last ego_status in the history list (current frame)
        if not agent_input.ego_statuses or len(agent_input.ego_statuses) == 0 or agent_input.ego_statuses[-1] is None:
            print(f"Warning: {self.get_unique_name()}: Ego status missing. Returning zero placeholder.")
            # Return a tensor with correct expected dimension (2+2+4=8)
            return {self.get_unique_name(): torch.zeros(2 + 2 + self.NUM_DRIVING_COMMANDS, dtype=torch.float32)}

        ego_status: EgoStatus = agent_input.ego_statuses[-1]

        # Ensure ego_velocity and ego_acceleration are 2D vectors
        velocity = torch.tensor(ego_status.ego_velocity, dtype=torch.float32)
        acceleration = torch.tensor(ego_status.ego_acceleration, dtype=torch.float32)

        if velocity.shape[-1] != 2 or acceleration.shape[-1] != 2:
             print(f"Warning: Builder: Unexpected vel/accel shape ({velocity.shape}, {acceleration.shape}). Returning zero placeholder.")
             return {self.get_unique_name(): torch.zeros(2 + 2 + self.NUM_DRIVING_COMMANDS, dtype=torch.float32)}


        command_raw = ego_status.driving_command
        command_one_hot = torch.zeros(self.NUM_DRIVING_COMMANDS, dtype=torch.float32)

        try:
            # Robustly handle different command formats
            if isinstance(command_raw, np.ndarray):
                if command_raw.size == 1:
                    command_index = int(command_raw.item())
                    if 0 <= command_index < self.NUM_DRIVING_COMMANDS: command_one_hot[command_index] = 1.0
                    else: print(f"Warning: Builder: Invalid command index {command_index} in array.")
                elif command_raw.shape == (self.NUM_DRIVING_COMMANDS,): # Check if it's already one-hot
                    command_one_hot = torch.from_numpy(command_raw).float()
                else: print(f"Warning: Builder: Unexpected numpy array size {command_raw.shape} for command.")
            elif isinstance(command_raw, (int, float)):
                command_index = int(command_raw)
                if 0 <= command_index < self.NUM_DRIVING_COMMANDS: command_one_hot[command_index] = 1.0
                else: print(f"Warning: Builder: Invalid command index {command_index}.")
            elif isinstance(command_raw, (list, tuple)) and len(command_raw) == self.NUM_DRIVING_COMMANDS:
                 command_one_hot = torch.tensor(command_raw, dtype=torch.float32)
            else: print(f"Warning: Builder: Unexpected command format {type(command_raw)}.")
        except Exception as e:
            print(f"Error processing driving command '{command_raw}': {e}. Using zero vector.")
            command_one_hot = torch.zeros(self.NUM_DRIVING_COMMANDS, dtype=torch.float32)

        ego_features = torch.cat([velocity, acceleration, command_one_hot], dim=-1)

        # Verify the final size matches the expected EGO_DIM
        expected_dim = 2 + 2 + self.NUM_DRIVING_COMMANDS
        if ego_features.shape[-1] != expected_dim:
             print(f"Warning: Builder: Final ego features dim mismatch {ego_features.shape[-1]} vs {expected_dim}. Returning zero placeholder.")
             return {self.get_unique_name(): torch.zeros(expected_dim, dtype=torch.float32)}

        return {self.get_unique_name(): ego_features}

# --- Target Builder ---

class TrajectoryTargetBuilderGT(AbstractTargetBuilder):
    """
    Target builder for extracting the ground truth future trajectory poses in absolute coordinates.
    """
    def __init__(self, trajectory_sampling: TrajectorySampling, num_history_frames: int):
        """
        Initializes the target builder.
        :param trajectory_sampling: Trajectory sampling specification (for num_poses).
        :param num_history_frames: Number of history frames (used internally for potential conversion if needed,
                                   though this builder returns absolute).
        """
        super().__init__()
        self._trajectory_sampling = trajectory_sampling
        self._num_history_frames = num_history_frames # Stored, but not used for conversion in this version

    def get_unique_name(self) -> str:
        return "trajectory_gt"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """
        Return the future trajectory in absolute map coordinates.
        """

        # 1) get the future trajectory from the scene
        future_traj: Trajectory = scene.get_future_trajectory(
            num_trajectory_frames=self._trajectory_sampling.num_poses
        )

        # 2) fallback to zeros if trajectory missing / too short
        if (future_traj is None or future_traj.poses is None or
            future_traj.poses.shape[0] < self._trajectory_sampling.num_poses):
            print(f"{self.get_unique_name()}: insufficient future poses "
                f"({0 if future_traj is None else future_traj.poses.shape[0]} "
                f"vs {self._trajectory_sampling.num_poses}). Returning zeros.")
            return {self.get_unique_name():
                    torch.zeros(self._trajectory_sampling.num_poses, 3,
                                dtype=torch.float32)}

        # 3) use absolute poses directly
        gt_abs = torch.as_tensor(future_traj.poses, dtype=torch.float32)

        # 4) final safety-check on shape
        if gt_abs.shape != (self._trajectory_sampling.num_poses, 3):
            print(f"{self.get_unique_name()}: shape mismatch {gt_abs.shape}, "
                "returning zeros.")
            gt_abs = torch.zeros(self._trajectory_sampling.num_poses, 3,
                                dtype=torch.float32)

        return {self.get_unique_name(): gt_abs}

# --- Simple CNN Visual Encoder Module ---
class SimpleCNNVisualEncoder(nn.Module):
    """
    A basic CNN to process camera images into a fixed-size feature vector.
    Designed for 224x224 input images.
    """
    def __init__(self, input_channels: int = 3, output_features: int = 1280):
        super().__init__()
        self.output_features = output_features

        # Example CNN layers reducing spatial size and increasing channels
        # Input: (B, 3, 224, 224)
        self.conv_layers = nn.Sequential(
            # Conv1: 224x224 -> 112x112
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # Conv2: 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # Conv3: 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            # Conv4: 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            # Conv5: 14x14 -> 7x7
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )

        # Global pooling to get a fixed-size vector regardless of final spatial size
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        # Linear layer to map features to the desired output dimension
        self.fc = nn.Linear(512, output_features)

        # Optional: Final activation/normalization before output
        self.final_layer = nn.Sequential(
             nn.ReLU(), # Apply ReLU before LayerNorm commonly
             nn.LayerNorm(output_features) # Normalizes the feature vector
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, C, H, W) e.g., (B, 3, 224, 224)
        # Check input shape
        if x.ndim != 4 or x.shape[1] != self.conv_layers[0].in_channels:
             raise ValueError(f"Expected input shape (B, {self.conv_layers[0].in_channels}, H, W), but got {x.shape}")

        x = self.conv_layers(x) # Output shape e.g., (B, 512, 7, 7)
        x = self.pooling(x)     # Output shape (B, 512, 1, 1)
        x = torch.flatten(x, 1) # Output shape (B, 512)
        x = self.fc(x)          # Output shape (B, output_features)
        x = self.final_layer(x) # Output shape (B, output_features)
        return x

class IJEPAPlanningAgent(AbstractAgent):
    """
    NAVSIM Agent with flexible visual encoding (I-JEPA, simple CNN, or none)
    and an MLP planning head.
    """
    # --- Constants ---
    # VISUAL_FEATURE_DIM is the size of the feature vector produced by any visual encoder
    VISUAL_FEATURE_DIM = 1280 # Matches I-JEPA's default ViT-H/14 output dimension
    EGO_DIM = 8 # Based on EgoFeatureBuilder: vx, vy, ax, ay, cmd (4) = 2+2+4=8
    HIDDEN_DIM = 256

    # --- Encoder Choices ---
    ENCODER_IJEPA = "ijepa"
    ENCODER_CNN = "cnn"
    ENCODER_NONE = "none" # Uses zero placeholder

    def __init__(
        self,
        mlp_weights_path: Optional[str] = None,
        visual_encoder_type: str = ENCODER_IJEPA, # New: Choose visual encoder
        ijepa_model_id: str = "facebook/ijepa_vith14_1k", # Only used if visual_encoder_type is 'ijepa'
        trajectory_sampling: TrajectorySampling = TrajectorySampling(
            time_horizon=4, interval_length=0.5
        ),
        num_history_frames: int = 4, # Passed to target builder
        use_cls_token_if_available: bool = True, # Only for I-JEPA
        requires_scene: bool = False, # Not used by this agent
        learning_rate: float = 1e-4,
        loss_criterion: str = "l1",
        max_epochs: int = 50, # For LR scheduler T_max
    ):
        """
        Initializes the IJEPAPlanningAgent with flexible visual encoding.
        :param mlp_weights_path: Path to pre-trained MLP weights (.pth or .ckpt).
        :param visual_encoder_type: Which visual encoder to use ('ijepa', 'cnn', 'none').
                                    'ijepa': Uses frozen I-JEPA.
                                    'cnn': Trains a SimpleCNNVisualEncoder.
                                    'none': Uses a zero placeholder for visual features.
        :param ijepa_model_id: Hugging Face model ID for I-JEPA encoder (only used if visual_encoder_type is 'ijepa').
        :param trajectory_sampling: Trajectory sampling specification.
        :param num_history_frames: Number of history frames (passed to target builder).
        :param use_cls_token_if_available: Whether to prefer CLS token over mean pooling (only for I-JEPA).
        :param requires_scene: Whether the agent's forward requires the full scene (not used by this agent).
        :param learning_rate: Learning rate for the optimizer.
        :param loss_criterion: Type of loss function ('l1' or 'mse').
        :param max_epochs: Maximum training epochs for LR scheduler.
        """
        super().__init__(
            trajectory_sampling=trajectory_sampling, requires_scene=requires_scene
        )

        self._num_history_frames = num_history_frames

        self._mlp_weights_path_config = mlp_weights_path
        self._ijepa_model_id = ijepa_model_id
        self._use_cls_token = use_cls_token_if_available
        self._learning_rate = learning_rate
        self._loss_criterion_type = loss_criterion
        self._max_epochs = max_epochs

        # --- Visual Encoder Configuration ---
        self._visual_encoder_type = visual_encoder_type.lower()
        if self._visual_encoder_type not in [self.ENCODER_IJEPA, self.ENCODER_CNN, self.ENCODER_NONE]:
             valid_types = [self.ENCODER_IJEPA, self.ENCODER_CNN, self.ENCODER_NONE]
             raise ValueError(f"Invalid visual_encoder_type: '{visual_encoder_type}'. Choose from {valid_types}.")

        # Modules that are conditionally initialized
        self._processor: Optional[AutoProcessor] = None # Only used for I-JEPA
        self._ijepa_encoder: Optional[AutoModel] = None # Only used for I-JEPA
        self._cnn_visual_encoder: Optional[SimpleCNNVisualEncoder] = None # Only used for CNN

        self._feature_extraction_method = "N/A (Initialization Pending)" # Initial state


        # Define MLP - Input dimension is fixed to VISUAL_FEATURE_DIM + EGO_DIM
        # The chosen visual encoder (or zero placeholder) outputs VISUAL_FEATURE_DIM features.
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.VISUAL_FEATURE_DIM + self.EGO_DIM, self.HIDDEN_DIM),
            torch.nn.ReLU(),
            nn.LayerNorm(self.HIDDEN_DIM),
            torch.nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM),
            torch.nn.ReLU(),
            nn.LayerNorm(self.HIDDEN_DIM),
            torch.nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM),
            torch.nn.ReLU(),
            nn.LayerNorm(self.HIDDEN_DIM),
            torch.nn.Linear(self.HIDDEN_DIM, self._trajectory_sampling.num_poses * 3),
        )

        # Define Loss
        if self._loss_criterion_type.lower() == "l1":
            self.criterion = torch.nn.L1Loss()
        elif self._loss_criterion_type.lower() == "mse":
            self.criterion = torch.nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss: {self._loss_criterion_type}. Choose 'l1' or 'mse'.")

    def name(self) -> str:
        """Provides a unique name for the agent based on its configuration."""
        base_name = self.__class__.__name__
        suffix = ""
        if self._visual_encoder_type == self.ENCODER_IJEPA:
            suffix = "_IJEP"
        elif self._visual_encoder_type == self.ENCODER_CNN:
            suffix = "_CNN"
        elif self._visual_encoder_type == self.ENCODER_NONE:
            suffix = "_NoVIS"
        return f"{base_name}{suffix}"

    def initialize(self) -> None:
        """
        Initializes agent components like the visual encoder and loads weights.
        """
        print(f"Initializing {self.name()}...")
        # Set performance flags (usually done once per process)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Move agent itself to device first so submodules inherit it
        self.to(device)

        # --- Load/Initialize Visual Encoder Based on Type ---
        if self._visual_encoder_type == self.ENCODER_IJEPA:
            try:
                print(f"Loading I-JEPA encoder: {self._ijepa_model_id}")
                self._processor = AutoProcessor.from_pretrained(self._ijepa_model_id, use_fast=True)
                # Map model to device during loading
                self._ijepa_encoder = AutoModel.from_pretrained(self._ijepa_model_id).to(device)
                # Freeze I-JEPA parameters
                for param in self._ijepa_encoder.parameters():
                    param.requires_grad = False
                self._ijepa_encoder.eval() # Ensure it's in evaluation mode

                print(f"Loaded and froze I-JEPA encoder: {self._ijepa_model_id}")

                # Determine Feature Extraction Method (I-JEPA specific)
                try:
                     with torch.no_grad():
                        # Dummy forward to check output structure/dims
                        dummy_input = torch.zeros(1, 3, 224, 224).to(device) # Assume standard input size
                        outputs = self._ijepa_encoder(pixel_values=dummy_input)
                        hidden = getattr(outputs, "last_hidden_state", None)
                        pooler = getattr(outputs, "pooler_output", None)

                        if self._use_cls_token:
                            # 1) Prefer real pooler_output if available and correct dim
                            if pooler is not None and pooler.shape[-1] == self.VISUAL_FEATURE_DIM:
                                self._feature_extraction_method = "Pooler Output"
                            # 2) Fallback to CLS token (first token of last_hidden_state)
                            elif hidden is not None and hidden.ndim == 3 and hidden.shape[1] > 0 and hidden[:, 0, :].shape[-1] == self.VISUAL_FEATURE_DIM:
                                self._feature_extraction_method = "CLS Token"
                            else:
                                # If neither pooler nor valid CLS token, default to mean pooling
                                print("Warning: I-JEPA Pooler Output or CLS token unavailable/dim mismatch; defaulting to Mean Pooling.")
                                self._feature_extraction_method = "Mean Pooling"
                        else:
                            # Forced mean pooling if use_cls_token is False
                            self._feature_extraction_method = "Mean Pooling"

                        # Final check on dimension for the chosen method
                        dummy_features = None
                        if self._feature_extraction_method == "Pooler Output":
                            dummy_features = ijepa_outputs.pooler_output
                        elif self._feature_extraction_method == "CLS Token":
                            dummy_features = ijepa_outputs.last_hidden_state[:, 0, :]
                        elif self._feature_extraction_method == "Mean Pooling":
                             dummy_features = ijepa_outputs.last_hidden_state.mean(dim=1)

                        if dummy_features is None or dummy_features.shape[-1] != self.VISUAL_FEATURE_DIM:
                             raise RuntimeError(f"I-JEPA extraction method '{self._feature_extraction_method}' did not yield features of size {self.VISUAL_FEATURE_DIM}. Got {dummy_features.shape if dummy_features is not None else 'None'}.")

                except Exception as e:
                     raise RuntimeError(f"Failed to verify I-JEPA output or determine feature extraction method: {e}") from e

                print(f"Using I-JEPA feature extraction: {self._feature_extraction_method}")

            except Exception as e:
                # If loading fails when I-JEPA is required, this is a fatal error
                raise RuntimeError(f"Failed to load I-JEPA model/processor as required: {e}") from e

        elif self._visual_encoder_type == self.ENCODER_CNN:
            try:
                print(f"Initializing Simple CNN Visual Encoder outputting {self.VISUAL_FEATURE_DIM} features.")
                # Instantiate the simple CNN encoder and move to device
                # This module's parameters require_grad=True by default, so they will be trained
                self._cnn_visual_encoder = SimpleCNNVisualEncoder(output_features=self.VISUAL_FEATURE_DIM).to(device)
                self._feature_extraction_method = "Simple CNN"
            except Exception as e:
                 raise RuntimeError(f"Failed to initialize Simple CNN Visual Encoder: {e}") from e

        elif self._visual_encoder_type == self.ENCODER_NONE:
             print("Visual encoder usage is disabled. Using zero placeholder for visual features.")
             self._feature_extraction_method = "N/A (Visual Disabled)"

        # --- Optional MLP-weight loading ─────────────────────────────────────────────
        # This part loads weights ONLY for the MLP head
        if self._mlp_weights_path_config:
            weights_path = Path(self._mlp_weights_path_config)

            if weights_path.is_file():
                print(f"Attempting to load MLP weights from: {weights_path}")
                try:
                    raw_checkpoint = torch.load(weights_path, map_location=device)

                    # Adapt to different checkpoint formats (e.g., Lightning .ckpt, plain .pth)
                    # Try loading from 'state_dict' key first, then assume raw dict
                    state = raw_checkpoint.get("state_dict", raw_checkpoint)

                    # Filter keys relevant to the MLP within the 'agent' module (common Lightning prefix)
                    mlp_state_dict = {
                        k.replace("agent.mlp.", ""): v
                        for k, v in state.items()
                        if k.startswith("agent.mlp.")
                    }

                    # If no keys matched that pattern, try 'mlp.' prefix (sometimes used)
                    if not mlp_state_dict:
                         mlp_state_dict = {
                            k.replace("mlp.", ""): v
                            for k, v in state.items()
                            if k.startswith("mlp.")
                         }

                    # If still no keys, try assuming it's just the state dict for the Sequential model
                    if not mlp_state_dict:
                        # Check if the keys look like Sequential layer indices (e.g., '0.weight', '1.bias')
                        if all(re.match(r'^\d+\.(weight|bias|running_mean|running_var|num_batches_tracked)$', k) for k in state.keys()):
                             mlp_state_dict = state
                             print("Assuming checkpoint keys match raw Sequential layer names.")
                        else:
                            print(f"Warning: Could not find MLP-specific keys in checkpoint {weights_path} using common patterns.")


                    if not mlp_state_dict:
                        print("Warning: No MLP-specific keys found or identified in checkpoint. Skipping MLP weight loading.")
                    else:
                        # Load the filtered state dict into the MLP module
                        # Use strict=False to allow partial loading if needed, though strict is preferred
                        try:
                            self.mlp.load_state_dict(mlp_state_dict, strict=True)
                            print(f"MLP weights loaded successfully ({len(mlp_state_dict)} tensors) with strict=True.")
                        except RuntimeError as e:
                             print(f"Warning: Strict load failed for MLP weights: {e}. Attempting non-strict load.")
                             try:
                                  self.mlp.load_state_dict(mlp_state_dict, strict=False)
                                  print(f"MLP weights loaded non-strictly ({len(mlp_state_dict)} tensors). Some keys might be missing or mismatched.")
                             except Exception as non_strict_e:
                                  print(f"Fatal Error: Non-strict MLP weight load also failed: {non_strict_e}. MLP weights will not be loaded.")


                except Exception as e:
                    print(f"Warning: An error occurred during MLP weight loading from {weights_path} ({e}). Starting fresh.")
            else:
                print(f"Warning: MLP weights file not found at {weights_path}. Starting fresh.")
        else:
            print("No MLP weights path provided. Starting training from scratch.")

        # Ensure MLP is on device (should be from self.to(device) earlier, but double check)
        self.mlp.to(device)
        print(f"{self.name()} initialization complete.")

    def get_sensor_config(self) -> SensorConfig:
        """
        Specifies which sensors are required by the agent's feature builders.
        Front camera is always required for the builder, even if the agent
        uses the 'none' visual encoder (needed for batch size inference).
        """
        return SensorConfig(cam_f0=True, cam_l0=False, cam_l1=False, cam_l2=False, cam_r0=False, cam_r1=False, cam_r2=False, cam_b0=False, lidar_pc=False)

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Returns the list of feature builders used by the agent."""
        # Keep both builders regardless of visual encoder choice
        return [CameraImageFeatureBuilder(), EgoFeatureBuilder()]

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Returns the list of target builders used for training."""
        # Target builder remains the same, providing absolute GT trajectory
        return [TrajectoryTargetBuilderGT(trajectory_sampling=self._trajectory_sampling,
                                         num_history_frames=self._num_history_frames)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the agent. Extracts features using the configured visual encoder
        and EgoFeatureBuilder, concatenates them, and passes through the MLP.
        Returns predicted relative trajectories.
        """
        # Check if agent components are ready
        if self.mlp is None:
            raise RuntimeError("Agent not properly initialized.")
        if self._visual_encoder_type == self.ENCODER_IJEPA and (self._ijepa_encoder is None or self._processor is None):
             raise RuntimeError("I-JEPA encoder/processor not initialized but required.")
        if self._visual_encoder_type == self.ENCODER_CNN and self._cnn_visual_encoder is None:
             raise RuntimeError("CNN visual encoder not initialized but required.")

        device = next(self.parameters()).device # Get model's device
        # Ensure modules are on the correct device (should be after initialize)
        # self.to(device) # This might not be needed every forward if done in initialize/training_step

        # Get features from builders' output
        try:
            # image_tensor_raw is needed by the visual encoder (or for batch size)
            # It's (B, 3, H, W) float [0, 1] thanks to the builder update
            # Ego features are (B, EGO_DIM) float
            image_tensor_raw = features.get("front_camera_image", None)
            ego_features_tensor = features.get("ego_features", None)

            # Basic check that required features are present for any mode
            if image_tensor_raw is None:
                 print(f"Error: 'front_camera_image' missing in features. Returning zero predictions.")
                 # Need batch size info - try ego features
                 batch_size = ego_features_tensor.shape[0] if ego_features_tensor is not None and ego_features_tensor.ndim > 1 else 1
                 zero_preds = torch.zeros(batch_size, self._trajectory_sampling.num_poses, 3, dtype=torch.float32, device=device)
                 return {"trajectory": zero_preds}

            if ego_features_tensor is None:
                 print(f"Error: 'ego_features' missing in features. Returning zero predictions.")
                 # Need batch size info - try image features
                 batch_size = image_tensor_raw.shape[0] if image_tensor_raw is not None and image_tensor_raw.ndim > 1 else 1
                 zero_preds = torch.zeros(batch_size, self._trajectory_sampling.num_poses, 3, dtype=torch.float32, device=device)
                 return {"trajectory": zero_preds}

            # Move fetched tensors to the correct device
            image_tensor_raw = image_tensor_raw.to(device)
            ego_features_tensor = ego_features_tensor.to(device)

        except Exception as e:
             print(f"Error during feature extraction from dict or moving to device: {e}. Returning zero predictions.")
             # Default batch size if we can't even get shape from inputs
             batch_size = 1
             zero_preds = torch.zeros(batch_size, self._trajectory_sampling.num_poses, 3, dtype=torch.float32, device=device)
             return {"trajectory": zero_preds}


        batch_size = ego_features_tensor.shape[0] # Get batch size from ego features


        # --- Visual Feature Extraction Based on Type ---
        visual_features = None

        if self._visual_encoder_type == self.ENCODER_IJEPA:
            # Original I-JEPA path (frozen)
            try:
                if self._processor is None or self._ijepa_encoder is None:
                    raise RuntimeError("I-JEPA encoder/processor is None in forward despite type being 'ijepa'")
                # Preprocess image using Hugging Face processor
                # Input: (B, C, H, W) float [0,1]. Processor expects PIL Images.
                # Convert tensor [B, C, H, W] float [0,1] to list of PIL images [B, H, W, C] uint8 [0,255]
                image_batch_np = (image_tensor_raw.mul(255)).permute(0, 2, 3, 1).byte().cpu().numpy()
                image_list_pil = [Image.fromarray(img) for img in image_batch_np]
                # Processor handles normalization, resizing etc. Returns 'pixel_values' (B, C, H, W) float
                processor_output = self._processor(images=image_list_pil, return_tensors="pt")
                processed_pixel_values = processor_output.get('pixel_values', None) # Use .get for safety
                if processed_pixel_values is None: raise ValueError("Processor returned None pixel_values")
                processed_pixel_values = processed_pixel_values.to(device)

                # Extract features with Frozen I-JEPA
                with torch.no_grad(): # Crucial: I-JEPA is frozen
                    ijepa_outputs = self._ijepa_encoder(pixel_values=processed_pixel_values)
                    # Extract features based on the method determined in initialize
                    if self._feature_extraction_method == "Pooler Output" and getattr(ijepa_outputs, "pooler_output", None) is not None:
                        visual_features = ijepa_outputs.pooler_output
                    elif self._feature_extraction_method == "CLS Token" and getattr(ijepa_outputs, "last_hidden_state", None) is not None and ijepa_outputs.last_hidden_state.shape[1] > 0:
                         visual_features = ijepa_outputs.last_hidden_state[:, 0, :] # Take the first token
                    elif getattr(ijepa_outputs, "last_hidden_state", None) is not None: # Fallback/Mean Pooling
                        visual_features = ijepa_outputs.last_hidden_state.mean(dim=1)
                    else:
                         # Should not happen if initialize was successful, but as a safeguard
                         raise ValueError(f"I-JEPA outputs unexpected format for method '{self._feature_extraction_method}'.")

                # Check I-JEPA output dimension
                if visual_features is None or visual_features.shape != (batch_size, self.VISUAL_FEATURE_DIM):
                    print(f"Warning: I-JEPA feature extraction failed or shape mismatch {visual_features.shape if visual_features is not None else 'None'}. Expected ({batch_size}, {self.VISUAL_FEATURE_DIM}). Returning zero visual features.")
                    visual_features = torch.zeros(batch_size, self.VISUAL_FEATURE_DIM, dtype=torch.float32, device=device)

            except Exception as e:
                print(f"Error during I-JEPA processing: {e}. Returning zero visual features.")
                visual_features = torch.zeros(batch_size, self.VISUAL_FEATURE_DIM, dtype=torch.float32, device=device)

        elif self._visual_encoder_type == self.ENCODER_CNN:
            # Use the custom CNN encoder (trainable)
            try:
                if self._cnn_visual_encoder is None:
                     raise RuntimeError("CNN encoder is None in forward despite type being 'cnn'")
                # image_tensor_raw is already (B, 3, H, W) float [0, 1]
                visual_features = self._cnn_visual_encoder(image_tensor_raw)
                if visual_features.shape != (batch_size, self.VISUAL_FEATURE_DIM):
                     print(f"Warning: CNN feature extraction output shape mismatch {visual_features.shape}. Expected ({batch_size}, {self.VISUAL_FEATURE_DIM}). Returning zero visual features.")
                     visual_features = torch.zeros(batch_size, self.VISUAL_FEATURE_DIM, dtype=torch.float32, device=device)
            except Exception as e:
                 print(f"Error during CNN visual encoding: {e}. Returning zero visual features.")
                 visual_features = torch.zeros(batch_size, self.VISUAL_FEATURE_DIM, dtype=torch.float32, device=device)

        elif self._visual_encoder_type == self.ENCODER_NONE:
            # No visual encoder, use zero placeholder
            visual_features = torch.zeros(batch_size, self.VISUAL_FEATURE_DIM, dtype=torch.float32, device=device)
            # print(f"DEBUG: Visual disabled, using zero visual features shape: {visual_features.shape}") # Optional debug print

        else:
            # Should be caught in __init__, but as a safeguard
            raise ValueError(f"Unknown visual_encoder_type: '{self._visual_encoder_type}'")

        # --- End Visual Feature Extraction ---

        # Final check before concatenation
        if visual_features is None or visual_features.shape != (batch_size, self.VISUAL_FEATURE_DIM):
             print(f"Fatal Error: Visual features preparation failed. Shape: {visual_features.shape if visual_features is not None else 'None'}. Returning zero predictions.")
             zero_preds = torch.zeros(batch_size, self._trajectory_sampling.num_poses, 3, dtype=torch.float32, device=device)
             return {"trajectory": zero_preds}


        # Predict with MLP
        try:
            # Ensure ego features have correct dim
            if ego_features_tensor.shape != (batch_size, self.EGO_DIM):
                 print(f"Warning: Ego feature dim mismatch: {ego_features_tensor.shape} vs ({batch_size}, {self.EGO_DIM}). Returning zero predictions.")
                 zero_preds = torch.zeros(batch_size, self._trajectory_sampling.num_poses, 3, dtype=torch.float32, device=device)
                 return {"trajectory": zero_preds}

            # Concatenate visual and ego features
            combined_features = torch.cat([visual_features, ego_features_tensor], dim=1)

            if combined_features.shape != (batch_size, self.VISUAL_FEATURE_DIM + self.EGO_DIM):
                 print(f"Fatal Error: Combined features dim mismatch: {combined_features.shape} vs ({batch_size}, {self.VISUAL_FEATURE_DIM + self.EGO_DIM}). Returning zero predictions.")
                 zero_preds = torch.zeros(batch_size, self._trajectory_sampling.num_poses, 3, dtype=torch.float32, device=device)
                 return {"trajectory": zero_preds}


            # Pass through MLP
            flat_predictions = self.mlp(combined_features)

            # Reshape prediction: (Batch, Flat_Dim) -> (Batch, Num_Poses, 3)
            expected_flat_dim = self._trajectory_sampling.num_poses * 3
            if flat_predictions.shape != (batch_size, expected_flat_dim):
                print(f"Warning: MLP output dim mismatch: {flat_predictions.shape} vs ({batch_size}, {expected_flat_dim}). Returning zero predictions.")
                zero_preds = torch.zeros(batch_size, self._trajectory_sampling.num_poses, 3, dtype=torch.float32, device=device)
                return {"trajectory": zero_preds}

            predicted_relative_poses = flat_predictions.view(batch_size, self._trajectory_sampling.num_poses, 3)

        except Exception as e:
            print(f"Error during MLP forward: {e}. Returning zero predictions.")
            # Attempt to infer batch_size if possible, otherwise default to 1
            batch_size = 1
            if visual_features is not None and visual_features.ndim > 1: batch_size = visual_features.shape[0]
            elif ego_features_tensor is not None and ego_features_tensor.ndim > 1: batch_size = ego_features_tensor.shape[0]

            zero_preds = torch.zeros(batch_size, self._trajectory_sampling.num_poses, 3, dtype=torch.float32, device=device)
            return {"trajectory": zero_preds}

        # Predictions are relative poses (delta_x, delta_y, delta_heading)
        return {"trajectory": predicted_relative_poses}


    def compute_loss(
        self,
        features: Dict[str, torch.Tensor], # Features are available but not needed for loss calculation itself
        targets:  Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes the loss between predicted relative poses and ground truth absolute poses.
        Converts predicted relative poses to absolute using the first GT pose as origin.
        """
        if self.criterion is None:
            raise RuntimeError("Loss criterion not initialized.")

        device = next(self.parameters()).device # Get model's device (where loss should be computed)

        # ---------------------------------------------------------------
        # 1) fetch tensors and move to device
        # ---------------------------------------------------------------
        try:
            # GT is in absolute coordinates from TrajectoryTargetBuilderGT
            gt_abs   = targets["trajectory_gt"].to(device)       # (B,T,3) absolute
            # Prediction is in relative coordinates from MLP output
            pred_rel = predictions["trajectory"].to(device)      # (B,T,3) relative
        except KeyError as e:
            print(f"Error: Missing key for loss computation: {e}. Returning zero loss.")
            # Return zero loss but ensure it requires gradients for backprop flow in PTL
            return torch.tensor(0.0, device=device, requires_grad=True)
        except Exception as e:
             print(f"Error moving tensors to device or other issue: {e}. Returning zero loss.")
             return torch.tensor(0.0, device=device, requires_grad=True)


        # Basic shape check
        if pred_rel.shape != gt_abs.shape:
            print(f"Warning: Shape mismatch during loss computation: pred {pred_rel.shape}, gt {gt_abs.shape}. Returning zero loss.")
            return torch.tensor(0.0, device=device, requires_grad=True)

        # ---------------------------------------------------------------
        # 2) Convert predicted relative poses to absolute using the first GT pose as the origin
        #    The first GT pose is the current ego pose (x, y, heading)
        # ---------------------------------------------------------------
        # Extract the origin (first pose) from the ground truth trajectory
        # Shape (B, 1, 3) - extracts the first pose for each batch item, keeping the time dimension
        origin = gt_abs[:, 0:1, :]

        # Add the origin to the relative predictions to get absolute predictions
        # Broadcasting handles the (B, 1, 3) + (B, T, 3) -> (B, T, 3)
        pred_abs = rel_to_abs(pred_rel, gt_abs)
        # pred_abs = pred_rel + origin # Equivalent to rel_to_abs in this case

        # ---------------------------------------------------------------
        # 3) Calculate loss between absolute predictions and absolute GT
        # ---------------------------------------------------------------
        # Loss is computed directly between the two (B, T, 3) tensors
        loss = self.criterion(pred_abs, gt_abs)

        return loss


    def get_optimizers(self) -> Dict[str, Union[Optimizer, LRScheduler]]:
        """
        Defines the optimizer and learning rate scheduler.
        Optimizes all parameters in the model that require gradients.
        """
        # self.parameters() automatically includes parameters from self.mlp
        # and potentially self._cnn_visual_encoder if it was initialized
        # and its parameters require_grad=True (default for nn.Module).
        # I-JEPA parameters are excluded because they are frozen (requires_grad=False).
        optimizer = torch.optim.AdamW(self.parameters(), lr=self._learning_rate)

        # Cosine Annealing scheduler
        # T_max should be the number of training epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self._max_epochs, # Use the value passed via config
            eta_min=self._learning_rate * 0.01 # Decay to 1% of initial LR (common)
            # eta_min=1e-6 # Alternative fixed minimum LR
        )

        # Return as a dictionary for PyTorch Lightning's setup
        opt_dict = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch', # Call scheduler.step() at the end of every training epoch
                'frequency': 1,
                # 'monitor': 'val_loss', # Optional: monitor validation loss for some schedulers
                # 'strict': True, # Optional: Whether to crash if monitor key is not found
            }
        }

        print(f"INFO: Optimizer setup:\n{opt_dict}")
        # Print which parameters are being optimized
        # print("Parameters being optimized:")
        # for name, param in self.named_parameters():
        #      if param.requires_grad:
        #          print(f"- {name} (Shape: {list(param.shape)})")


        return opt_dict

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Returns list of PyTorch Lightning callbacks for training."""
        # Include the FirstBatchDebugger callback
        return [ FirstBatchDebugger() ]


# Example Usage (conceptual, depends on your training script structure):

# To train with I-JEPA (default behavior if visual_encoder_type is not specified):
# agent_ijepa = IJEPAPlanningAgent(
#     visual_encoder_type=IJEPAPlanningAgent.ENCODER_IJEPA,
#     max_epochs=50,
#     learning_rate=1e-4,
#     trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5),
#     num_history_frames=4,
# )
# trainer = pl.Trainer(...) # Setup Trainer
# trainer.fit(agent_ijepa, train_dataloader, val_dataloader)


# To train with the Simple CNN encoder instead of I-JEPA:
# agent_cnn = IJEPAPlanningAgent(
#     visual_encoder_type=IJEPAPlanningAgent.ENCODER_CNN, # Choose CNN
#     max_epochs=50,
#     learning_rate=1e-4,
#     trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5),
#     num_history_frames=4,
#     # Note: ijepa_model_id and use_cls_token_if_available are ignored when visual_encoder_type='cnn'
# )
# trainer = pl.Trainer(...) # Setup Trainer
# trainer.fit(agent_cnn, train_dataloader, val_dataloader)


# To train ONLY the MLP with no visual input (zero placeholder):
# agent_mlp_only = IJEPAPlanningAgent(
#     visual_encoder_type=IJEPAPlanningAgent.ENCODER_NONE, # Choose None
#     max_epochs=50,
#     learning_rate=1e-4,
#     trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5),
#     num_history_frames=4,
#     # Note: ijepa_model_id and use_cls_token_if_available are ignored when visual_encoder_type='none'
# )
# trainer = pl.Trainer(...) # Setup Trainer
# trainer.fit(agent_mlp_only, train_dataloader, val_dataloader)


# Remember to handle data loading (Dataset, DataLoader) and PyTorch Lightning Trainer setup separately.