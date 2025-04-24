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
from pathlib import Path
from transformers import AutoProcessor, AutoModel
from typing import Dict, List, Union, Tuple, Any, Optional

import pytorch_lightning as pl
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import (
    AgentInput,
    Trajectory,
    EgoStatus,
    TrajectorySampling,
    SensorConfig,
    Scene,
    DrivingCommand,
)  # Import DrivingCommand

# Assuming convert_absolute_to_relative_se2_array is available, e.g., from a utils module
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)

from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

class CameraImageFeatureBuilder(AbstractFeatureBuilder):
    """
    Feature builder for extracting the front camera image.
    Returns the raw image data (numpy array or PIL Image) to be processed by the agent's forward.
    """

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "front_camera_image"  # Use a clear key

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        # Check for front camera data
        if (
            not agent_input.cameras
            or not hasattr(agent_input.cameras, "cam_f0")
            or agent_input.cameras.cam_f0.image is None
        ):
            # Handle missing data: return a placeholder (e.g., zero tensor) or raise error.
            # Returning a zero tensor is generally better for batching, but need consistent shape.
            # Let's return a zero tensor of a plausible image shape (e.g., 224x224 RGB numpy array)
            # The agent's forward will need to handle this zero input.
            print(
                f"Warning: {self.get_unique_name()}: \
                Front camera image missing. Returning zero placeholder."
            )
            # Return a zero numpy array matching expected image format (H, W, C)
            # Assuming standard image dimensions like 224x224 for I-JEPA ViT
            # The agent's forward must handle processing this zero array.
            dummy_image = np.zeros(
                (224, 224, 3), dtype=np.uint8
            )  # Use uint8 for image data type
            # Convert to tensor, consistent with how PIL image is handled later in agent.forward
            # HWC to CHW and float
            return {
                self.get_unique_name(): torch.from_numpy(dummy_image)
                .permute(2, 0, 1)
                .float()
            }

        # Extract image numpy array
        image_np = agent_input.cameras.cam_f0.image  # This is a numpy array (H, W, C)

        # Return the raw image data (as a tensor)
        # Convert HWC numpy to CHW tensor and to float (as expected by AutoProcessor later)
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()

        return {self.get_unique_name(): image_tensor}


class EgoFeatureBuilder(AbstractFeatureBuilder):
    """
    Feature builder for extracting ego status features (velocity, acceleration, driving command).
    Formats the driving command into a one-hot vector.
    """

    # Ensure this matches the agent's EGO_DIM
    NUM_DRIVING_COMMANDS = (
        4  # Hardcoded based on agent's expectation and likely data format
    )

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "ego_features"  # Use the key agent expects

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        if not agent_input.ego_statuses:
            print(
                f"Warning: {self.get_unique_name()}: \
                Ego status missing. Returning zero placeholder."
            )
            # Return a zero tensor matching expected ego feature dimension (EGO_DIM)
            return {
                self.get_unique_name(): torch.zeros(
                    self.NUM_DRIVING_COMMANDS + 4, dtype=torch.float32
                )
            }  # 2 vel + 2 acc + 4 cmd = 8

        ego_status: EgoStatus = agent_input.ego_statuses[-1]

        # Extract velocity and acceleration
        velocity = torch.tensor(
            ego_status.ego_velocity, dtype=torch.float32
        )  # Shape [2]
        acceleration = torch.tensor(
            ego_status.ego_acceleration, dtype=torch.float32
        )  # Shape [2]

        # Handle driving command - robustly convert to one-hot vector
        command_raw = ego_status.driving_command
        command_one_hot = torch.zeros(self.NUM_DRIVING_COMMANDS, dtype=torch.float32)

        # Check if it's a scalar index (int, float, or single-element container)
        if isinstance(command_raw, (int, float)):
            command_index = int(command_raw)
            if 0 <= command_index < self.NUM_DRIVING_COMMANDS:
                command_one_hot[command_index] = 1.0
            else:
                print(
                    f"Warning: Builder: Invalid command index {command_index}. Using zero vector."
                )
        elif (
            hasattr(command_raw, "__len__")
            and len(command_raw) == self.NUM_DRIVING_COMMANDS
        ):
            # Assume it's already a vector, potentially one-hot
            try:
                command_one_hot = torch.tensor(command_raw, dtype=torch.float32)
                # Optional validation: Check if it's roughly one-hot
                if not (
                    torch.isclose(torch.sum(command_one_hot), torch.tensor(1.0))
                    or torch.sum(command_one_hot) == 0.0
                ):  # Allow zero vector for invalid commands
                    # print(f"Warning: Builder: Command vector not strictly one-hot or zero: {command_raw}. Using as is.") # Too noisy
                    pass
            except (ValueError, TypeError) as e:
                print(
                    f"Warning: Builder: Could not process command vector {command_raw} (type: {type(command_raw)}): {e}. Using zero vector."
                )
                command_one_hot = torch.zeros(
                    self.NUM_DRIVING_COMMANDS, dtype=torch.float32
                )
        elif isinstance(
            command_raw, DrivingCommand
        ):  # Handle DrivingCommand enum if present
            command_index = (
                command_raw.value
            )  # Assuming enum values are 0, 1, 2, 3 etc.
            if 0 <= command_index < self.NUM_DRIVING_COMMANDS:
                command_one_hot[command_index] = 1.0
            else:
                print(
                    f"Warning: Builder: Invalid DrivingCommand enum value {command_index}. Using zero vector."
                )
        else:
            # Fallback for unexpected format
            print(
                f"Warning: Builder: Unexpected command format {command_raw} (type: {type(command_raw)}). Using zero vector."
            )
            command_one_hot = torch.zeros(
                self.NUM_DRIVING_COMMANDS, dtype=torch.float32
            )

        # Concatenate velocity, acceleration, and one-hot command
        ego_features = torch.cat(
            [velocity, acceleration, command_one_hot], dim=-1
        )  # Shape [2+2+4=8]

        if ego_features.shape[-1] != (2 + 2 + self.NUM_DRIVING_COMMANDS):
            print(
                f"Warning: Builder: Ego features dimension mismatch after concat: {ego_features.shape[-1]}. \
                Expected {2 + 2 + self.NUM_DRIVING_COMMANDS}."
            )
            # Attempt to resize/pad if possible, or handle error

        return {self.get_unique_name(): ego_features}


# --- Custom Target Builder ---


class TrajectoryTargetBuilderGT(AbstractTargetBuilder):
    """
    Target builder for extracting the ground truth future trajectory poses.
    Converts absolute poses to relative poses.
    """

    def __init__(self, trajectory_sampling: TrajectorySampling):
        """
        Initializes the target builder.
        :param trajectory_sampling: trajectory sampling specification (needed to get num_poses).
        """
        super().__init__()
        self._trajectory_sampling = trajectory_sampling
        # Get num_history_frames from trajectory_sampling for context pose
        self._num_history_frames = (
            trajectory_sampling.num_history_frames
            if hasattr(trajectory_sampling, "num_history_frames")
            else 1
        )  # Default to 1 if not specified

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "trajectory_gt"  # Use a clear key (e.g., adding _gt for ground truth)

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        # Get the ground truth future trajectory from the scene
        # scene.get_future_trajectory returns absolute poses
        future_trajectory: Trajectory = scene.get_future_trajectory(
            num_trajectory_frames=self._trajectory_sampling.num_poses
        )

        if not future_trajectory or len(future_trajectory.poses) == 0:
            print(
                f"Warning: {self.get_unique_name()}: No future trajectory poses found. Returning zero placeholder."
            )
            # Return a zero tensor of expected shape (num_future_frames, 3)
            return {
                self.get_unique_name(): torch.zeros(
                    self._trajectory_sampling.num_poses, 3, dtype=torch.float32
                )
            }

        gt_poses_abs = future_trajectory.poses  # N x 3 numpy array (x, y, heading)

        # Get the pose at the current frame to convert absolute to relative
        # The current frame is usually the last frame in the history (index num_history_frames - 1)
        if not scene.frames or len(scene.frames) < self._num_history_frames:
            print(
                f"Warning: {self.get_unique_name()}: Scene has insufficient frames to get current pose. \
                Cannot convert to relative. Returning absolute poses."
            )
            gt_poses_rel = (
                gt_poses_abs  # Fallback to absolute if current pose unavailable
            )
        else:
            current_frame = scene.frames[self._num_history_frames - 1]
            current_ego_pose = current_frame.ego_status.pose  # Pose at current frame

            # Convert absolute ground truth poses to relative poses w.r.t. current ego pose
            try:
                gt_poses_rel = convert_absolute_to_relative_se2_array(
                    current_ego_pose, gt_poses_abs
                )  # N x 3 numpy array
            except RuntimeError as e:  # Replace with the specific exception type
                print(
                    f"Warning: {self.get_unique_name()}: Failed to convert poses to relative: {e}. Returning absolute poses."
                )
                gt_poses_rel = gt_poses_abs  # Fallback on error

        # Convert numpy array to tensor
        gt_poses_tensor = torch.tensor(
            gt_poses_rel, dtype=torch.float32
        )  # [num_future_frames, 3]

        # Ensure the tensor has the correct number of future frames
        if gt_poses_tensor.shape[0] != self._trajectory_sampling.num_poses:
            print(
                f"Warning: {self.get_unique_name()}: Ground truth poses shape mismatch: {gt_poses_tensor.shape[0]} \
                frames found, {self._trajectory_sampling.num_poses} expected. Returning zero placeholder."
            )
            return {
                self.get_unique_name(): torch.zeros(
                    self._trajectory_sampling.num_poses, 3, dtype=torch.float32
                )
            }

        return {self.get_unique_name(): gt_poses_tensor}


# --- Updated IJEPAPlanningAgent Class ---


class IJEPAPlanningAgent(AbstractAgent):
    """
    NAVSIM Agent combining I-JEPA encoding and an MLP planning head,
    designed for compatibility with the standard NAVSIM training script
    using Feature and Target Builders.
    """

    # --- Constants (MUST MATCH TRAINING CONFIGURATION AND BUILDERS) ---
    NUM_FUTURE_FRAMES = 8
    IJEP_DIM = 1280
    EGO_DIM = 8  # Matches output of EgoFeatureBuilder (2 vel + 2 acc + 4 cmd)
    HIDDEN_DIM = 256
    NUM_DRIVING_COMMANDS = 4  # Matches EgoFeatureBuilder constant
    # --- End Constants ---

    def __init__(
        self,
        mlp_weights_path: Optional[str] = None,  # Made optional, can train from scratch
        ijepa_model_id: str = "facebook/ijepa_vith14_1k",  # Default I-JEPA model
        trajectory_sampling: TrajectorySampling = TrajectorySampling(
            time_horizon=4, interval_length=0.5
        ),  # Default sampling
        use_cls_token_if_available: bool = True,
        requires_scene: bool = False,  # Agent doesn't inherently require full scene for inference after init
        learning_rate: float = 1e-4,
        loss_criterion: str = "l1",
    ):
        """
        Initializes the IJEPAPlanningAgent.

        :param mlp_weights_path: Optional path to pre-trained MLP weights. If None, starts from random initialization.
        :param ijepa_model_id: Identifier for the I-JEPA model to load.
        :param trajectory_sampling: Configuration for the output trajectory (defines num_poses for target).
        :param use_cls_token_if_available: Whether to use the CLS token for I-JEPA features.
        :param requires_scene: Whether this agent requires scene data (passed to AbstractAgent).
        :param learning_rate: Learning rate for the optimizer.
        :param loss_criterion: Type of loss function ('l1' or 'mse').
        """
        # Pass trajectory_sampling and requires_scene to super().__init__
        # trajectory_sampling is needed by the base class for compute_trajectory structure
        super().__init__(
            trajectory_sampling=trajectory_sampling, requires_scene=requires_scene
        )

        self._mlp_weights_path_config = mlp_weights_path
        self._ijepa_model_id = ijepa_model_id
        self._use_cls_token = use_cls_token_if_available
        self._learning_rate = learning_rate
        self._loss_criterion_type = loss_criterion

        # Processor and encoder loaded in initialize
        self._processor: AutoProcessor = None
        self._ijepa_encoder: AutoModel = None
        self._feature_extraction_method = "Mean Pooling"  # Determined in initialize()

        # Define the trainable MLP structure as a submodule
        # Input dim is IJEP_DIM + EGO_DIM
        # Output dim is NUM_FUTURE_FRAMES * 3 (x, y, heading)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.IJEP_DIM + self.EGO_DIM, self.HIDDEN_DIM),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(self.HIDDEN_DIM),
            torch.nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(self.HIDDEN_DIM),
            torch.nn.Linear(
                self.HIDDEN_DIM, self.HIDDEN_DIM
            ),  # Added another layer for consistency with EgoStatusMLPAgent
            torch.nn.ReLU(),  # Added ReLU
            torch.nn.LayerNorm(self.HIDDEN_DIM),  # Added LayerNorm
            torch.nn.Linear(
                self.HIDDEN_DIM, self.trajectory_sampling.num_poses * 3
            ),  # Use self.trajectory_sampling
        )

        # Define the loss criterion
        if self._loss_criterion_type.lower() == "l1":
            self.criterion = torch.nn.L1Loss()
        elif self._loss_criterion_type.lower() == "mse":
            self.criterion = torch.nn.MSELoss()
        else:
            raise ValueError(
                f"Unsupported loss criterion: {self._loss_criterion_type}. Choose 'l1' or 'mse'."
            )

        # Set device (can be done here, but models are moved in initialize)
        # self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__  # Use class name

    def initialize(self) -> None:
        """
        Initializes the agent by loading the I-JEPA encoder, processor,
        and optionally loading the pre-trained MLP weights.
        Called by the NAVSIM training/evaluation framework after agent instantiation.
        """
        print(f"Initializing {self.name()}...")
        # Determine device dynamically if not set in __init__
        device = (
            next(self.parameters()).device
            if next(self.parameters(), None) is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Using device: {device}")
        self.to(device)  # Ensure all parameters are on the correct device

        # --- Load Models and Processor ---
        try:
            # Load processor first
            self._processor = AutoProcessor.from_pretrained(
                self._ijepa_model_id, use_fast=True
            )
            # Load I-JEPA encoder
            self._ijepa_encoder = AutoModel.from_pretrained(self._ijepa_model_id).to(
                device
            )
            # Freeze the I-JEPA encoder parameters as we are only training the MLP
            for param in self._ijepa_encoder.parameters():
                param.requires_grad = False
            self._ijepa_encoder.eval()  # Set I-JEPA encoder to evaluation mode
            print(f"Loaded I-JEPA encoder: {self._ijepa_model_id}")
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to load I-JEPA model or processor {self._ijepa_model_id}: {e}"
            ) from e

        # --- Determine feature extraction method ---
        # Perform a dummy forward pass to check model output structure
        with torch.no_grad():
            try:
                dummy_input = torch.zeros(1, 3, 224, 224).to(device)
                outputs = self._ijepa_encoder(pixel_values=dummy_input)

                if (
                    self._use_cls_token
                    and hasattr(outputs, "pooler_output")
                    and outputs.pooler_output is not None
                ):
                    if outputs.pooler_output.shape[-1] == self.IJEP_DIM:
                        self._feature_extraction_method = "Pooler Output (CLS Token)"
                    else:
                        print(
                            f"Warning: I-JEPA pooler_output dimension {outputs.pooler_output.shape[-1]}\
                              does not match expected IJEP_DIM {self.IJEP_DIM}. Defaulting to Mean Pooling."
                        )
                        self._feature_extraction_method = (
                            "Mean Pooling"  # Fallback if dimension mismatch
                        )
                elif hasattr(outputs, "last_hidden_state"):
                    if outputs.last_hidden_state.shape[-1] == self.IJEP_DIM:
                        self._feature_extraction_method = "Mean Pooling"
                    else:
                        print(
                            f"Warning: I-JEPA last_hidden_state dimension {outputs.last_hidden_state.shape[-1]} \
                            does not match expected IJEP_DIM {self.IJEP_DIM}. Cannot use Mean Pooling. Feature extraction may fail."
                        )
                        self._feature_extraction_method = (
                            "Unknown"  # Indicate potential issue
                        )
                else:
                    self._feature_extraction_method = "Unknown"
                    print(
                        "Warning: Could not determine I-JEPA feature extraction method from model outputs."
                    )
            except RuntimeError as e:
                self._feature_extraction_method = (
                    "Mean Pooling"  # Default fallback on error
                )
                print(
                    f"Warning: Error checking I-JEPA output structure: {e}. Defaulting to {self._feature_extraction_method}."
                )

        print(
            f"Using I-JEPA feature extraction method: {self._feature_extraction_method}"
        )

        # --- Load MLP Head weights (optional) ---
        if self._mlp_weights_path_config:
            weights_path = Path(self._mlp_weights_path_config)
            if not weights_path.is_file():
                print(
                    f"Warning: MLP weights file not found at {weights_path}. Starting training from scratch."
                )
            else:
                print(f"Loading MLP weights from: {weights_path}")
                try:
                    # Load state dict into the mlp submodule
                    # Need to handle potential 'module.' prefix if saved from DataParallel
                    state_dict = torch.load(weights_path, map_location=device)
                    # Remove 'module.' prefix if it exists
                    state_dict = {
                        k.replace("module.", ""): v for k, v in state_dict.items()
                    }
                    self.mlp.load_state_dict(state_dict)
                    print("MLP weights loaded successfully.")
                except RuntimeError as e:
                    print(
                        f"Warning: Failed to load MLP weights from {weights_path}: {e}. Starting training from scratch."
                    )
                    # No raise here, just start from default init if loading fails

        # Ensure MLP is on the correct device (already done by self.to(device))
        self.mlp.to(device)

        print(f"{self.name()} initialization complete.")

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        # Required sensor configuration for the SceneLoader/Dataset to provide data
        return SensorConfig(
            cam_f0=True,
            cam_l0=False,
            cam_l1=False,
            cam_l2=False,
            cam_r0=False,
            cam_r1=False,
            cam_r2=False,
            cam_b0=False,
            lidar_pc=False,
        )

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        # Return the builders that extract data for the forward pass
        return [
            CameraImageFeatureBuilder(),  # Extracts raw image data
            EgoFeatureBuilder(),  # Extracts and formats ego features
        ]

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        # Return the builders that extract ground truth targets for loss computation
        return [
            TrajectoryTargetBuilderGT(
                trajectory_sampling=self.trajectory_sampling
            )  # Extracts GT trajectory
        ]

    # This forward pass is called by the AgentLightningModule during training/validation
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        if self._ijepa_encoder is None or self._processor is None or self.mlp is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        # Ensure models are on the correct device
        device = next(self.parameters()).device
        self.mlp.to(device)
        self._ijepa_encoder.to(device)
        # Processor doesn't have a .to() method

        # --- Extract raw data from features dictionary ---
        # The keys must match the unique_names of the FeatureBuilders
        try:
            # This is the raw image tensor from CameraImageFeatureBuilder (CHW, float)
            image_tensor_raw = features["front_camera_image"].to(device)
            # This is the formatted ego features tensor from EgoFeatureBuilder (EGO_DIM)
            ego_features_tensor = features["ego_features"].to(device)
        except KeyError as e:
            raise KeyError(f"Missing expected key in features dictionary. Ensure builders provide '{e}'.") from e
        except RuntimeError as e:
            raise RuntimeError(f"Error moving features to device in forward pass: {e}") from e

        # --- I-JEPA Feature Extraction (using the frozen encoder) ---
        # Need to convert the raw image tensor back to PIL image format or process directly
        # The Hugging Face processor expects PIL Images or batches of pixel values.
        # The builder returns a tensor. Let's convert tensor batch -> list of PIL images
        # and then process with AutoProcessor. This handles resizing, normalization, etc.
        # Note: This processing step within forward adds overhead. Ideally, a FeatureBuilder
        # would handle *all* data preparation up to the point the agent needs it.
        # However, since the I-JEPA model/processor are in the agent, processing here is necessary.

        processed_pixel_values = None
        try:
            # Assume input tensor is BCHW format float [0, 255] or similar.
            # Convert tensor batch (B, C, H, W) to list of HWC numpy uint8 images
            image_batch_np = (
                image_tensor_raw.mul(255).permute(0, 2, 3, 1).byte().cpu().numpy()
            )
            image_list_pil = [Image.fromarray(img) for img in image_batch_np]

            # Process the list of PIL images using the Hugging Face processor
            processor_output = self._processor(
                images=image_list_pil, return_tensors="pt"
            )
            processed_pixel_values = processor_output["pixel_values"].to(device)

            if processed_pixel_values is None:
                raise ValueError("Processor returned None for pixel_values.")

        except (ValueError, RuntimeError, TypeError) as e:
            # If image processing fails, cannot proceed. Return zero prediction or raise error.
            print(
                f"Error during image processing with AutoProcessor in forward pass: {e}. Returning zero prediction batch."
            )
            batch_size = image_tensor_raw.shape[0] if image_tensor_raw.ndim == 4 else 1
            zero_predictions = torch.zeros(
                batch_size,
                self.trajectory_sampling.num_poses,
                3,
                dtype=torch.float32,
                device=device,
            )
            return {"trajectory": zero_predictions}

        # Use torch.no_grad() as the encoder is frozen
        with torch.no_grad():
            try:
                ijepa_outputs = self._ijepa_encoder(pixel_values=processed_pixel_values)

                visual_features = None
                if (
                    self._feature_extraction_method == "Pooler Output (CLS Token)"
                    and hasattr(ijepa_outputs, "pooler_output")
                    and ijepa_outputs.pooler_output is not None
                ):
                    visual_features = ijepa_outputs.pooler_output
                elif (
                    self._feature_extraction_method == "Mean Pooling"
                    and hasattr(ijepa_outputs, "last_hidden_state")
                    and ijepa_outputs.last_hidden_state is not None
                ):
                    visual_features = ijepa_outputs.last_hidden_state.mean(dim=1)
                else:
                    # Fallback
                    if (
                        hasattr(ijepa_outputs, "last_hidden_state")
                        and ijepa_outputs.last_hidden_state is not None
                    ):
                        visual_features = ijepa_outputs.last_hidden_state.mean(dim=1)
                    else:
                        raise ValueError(
                            "Could not extract visual features from I-JEPA output."
                        )

                if visual_features is None:
                    raise ValueError("Visual features extraction resulted in None.")

                if visual_features.shape[-1] != self.IJEP_DIM:
                    raise ValueError(
                        f"Extracted visual features dimension {visual_features.shape[-1]} does not match expected IJEP_DIM {self.IJEP_DIM}."
                    )

            except (ValueError, RuntimeError, TypeError) as e:
                print(
                    f"Error during I-JEPA feature extraction in forward pass: {e}. Returning zero prediction batch."
                )
                batch_size = (
                    processed_pixel_values.shape[0]
                    if processed_pixel_values.ndim == 4
                    else 1
                )
                zero_predictions = torch.zeros(
                    batch_size,
                    self.trajectory_sampling.num_poses,
                    3,
                    dtype=torch.float32,
                    device=device,
                )
                return {"trajectory": zero_predictions}

        # --- MLP Forward Pass ---
        try:
            # Ensure ego features match expected EGO_DIM
            if ego_features_tensor.shape[-1] != self.EGO_DIM:
                # This indicates an issue in the EgoFeatureBuilder or data
                print(
                    f"Error: Ego features dimension {ego_features_tensor.shape[-1]} does not match expected EGO_DIM {self.EGO_DIM} in forward pass. Returning zero prediction batch."
                )
                batch_size = (
                    ego_features_tensor.shape[0] if ego_features_tensor.ndim == 2 else 1
                )
                zero_predictions = torch.zeros(
                    batch_size,
                    self.trajectory_sampling.num_poses,
                    3,
                    dtype=torch.float32,
                    device=device,
                )
                return {"trajectory": zero_predictions}

            # Concatenate visual and ego features
            combined_features = torch.cat([visual_features, ego_features_tensor], dim=1)

            # Pass concatenated features through the MLP submodule
            flat_predictions = self.mlp(combined_features)

            # Reshape flat predictions into trajectory poses (Batch, num_poses, 3)
            # Assuming output is (batch_size, num_poses * 3)
            predicted_relative_poses_tensor = flat_predictions.view(
                -1, self.trajectory_sampling.num_poses, 3
            )

        except (RuntimeError, ValueError) as e:
            print(
                f"Error during MLP forward pass in forward pass: {e}. Returning zero prediction batch."
            )
            batch_size = (
                combined_features.shape[0] if combined_features.ndim == 2 else 1
            )
            zero_predictions = torch.zeros(
                batch_size,
                self.trajectory_sampling.num_poses,
                3,
                dtype=torch.float32,
                device=device,
            )
            return {"trajectory": zero_predictions}

        # --- Return predictions dictionary ---
        # The key name here ('trajectory') should match what compute_loss expects
        predictions = {"trajectory": predicted_relative_poses_tensor}

        return predictions

    # This compute_loss is called by the AgentLightningModule during training/validation
    def compute_loss(
        self,
        features: Dict[
            str, torch.Tensor
        ],  # Included as per AbstractAgent signature, though not used here
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        if self.criterion is None:
            raise RuntimeError("Loss criterion not initialized.")

        # Ensure data is on the correct device
        device = next(self.parameters()).device

        # --- Extract ground truth and predictions from dictionaries ---
        # The keys must match the unique_names of the TargetBuilders and the keys
        # used in the `forward` method's predictions output.
        try:
            ground_truth_trajectory = targets["trajectory_gt"].to(
                device
            )  # Key from TrajectoryTargetBuilderGT
            predicted_trajectory = predictions["trajectory"].to(
                device
            )  # Key from forward method
        except KeyError as e:
            raise KeyError(
                f"Missing expected key in targets or predictions dictionary: {e}. "
                "Ensure builders provide 'trajectory_gt' and forward returns 'trajectory'."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Error moving targets/predictions to device in compute_loss: {e}"
            ) from e

        # --- Compute Loss ---
        # Ensure shapes match for loss computation (Batch, num_future_frames, 3)
        if predicted_trajectory.shape != ground_truth_trajectory.shape:
            print(
                f"Warning: Prediction shape {predicted_trajectory.shape} \
                does not match ground truth shape {ground_truth_trajectory.shape} in compute_loss."
            )
            # Return zero loss or handle error. Returning zero loss avoids crashing batch.
            return torch.tensor(
                0.0, device=device, requires_grad=True
            )  # Return a trainable zero tensor

        loss = self.criterion(predicted_trajectory, ground_truth_trajectory)

        return loss

    # This get_optimizers is called by the AgentLightningModule
    def get_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        if self.mlp is None:
            raise RuntimeError("MLP head not initialized.")

        # Only optimize the parameters of the MLP module
        optimizer = torch.optim.AdamW(self.mlp.parameters(), lr=self._learning_rate)

        # You can return a dictionary here if you had multiple optimizers/schedulers,
        # following PyTorch Lightning's conventions, but a single optimizer is sufficient.
        return optimizer

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Inherited, see superclass."""
        # Return any PyTorch Lightning callbacks specific to this agent's training
        # ModelCheckpoint is typically handled by the main training script config,
        # but you could add custom callbacks here if needed.
        return []  # No specific callbacks required by the agent itself here
