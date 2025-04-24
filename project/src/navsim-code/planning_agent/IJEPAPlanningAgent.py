import torch
import os
import glob
import re
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F  # Import functional for loss
from torch.optim import AdamW
from tqdm import tqdm
from PIL import Image
import numpy as np
from pathlib import Path
from transformers import AutoProcessor, AutoModel
from typing import Dict, List, Union, Tuple

import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import (
    AgentInput,
    Trajectory,
    EgoStatus,
    TrajectorySampling,
    SensorConfig,
)
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)


# Helper function from original PlanningHead
def _auto_resume_path(pattern="checkpoint_epoch*.pth"):
    """
    Automatically finds the path to the latest checkpoint based on epoch number.
    """
    ckpts = glob.glob(pattern)
    if not ckpts:
        return None
    epochs = []
    for p in ckpts:
        m = re.search(r"checkpoint_epoch(\d+)\.pth", p)
        if m:
            epochs.append((int(m.group(1)), p))
    # Sort by epoch number and return the path of the latest
    return max(epochs, key=lambda x: x[0])[1] if epochs else None


class IJEPAPlanningAgent(AbstractAgent):
    """
    A NAVSIM-compatible agent leveraging a frozen I-JEPA encoder for visual feature extraction
    and a trainable MLP head for trajectory prediction. This agent predicts a sequence of future
    poses (x, y, heading) by combining visual features from the I-JEPA model with ego features.
    The MLP head is optimized to output accurate trajectories, ensuring compatibility with
    NAVSIM evaluation scripts and training pipelines.
    """

    # --- Constants (MUST MATCH TRAINING CONFIGURATION) ---
    NUM_FUTURE_FRAMES = 8  # Number of poses in the output trajectory
    IJEP_DIM = 1280  # Dimension of I-JEPA features (1280 for ViT-H/14)
    EGO_DIM = 8  # Dimension of Ego features
    HIDDEN_DIM = 256  # Hidden dimension used in the MLP head
    NUM_DRIVING_COMMANDS = 4  # Number of driving commands expected
    # --- End Constants ---

    def __init__(
        self,
        mlp_weights_path: str,
        ijepa_model_id: str,
        trajectory_sampling: TrajectorySampling,
        use_cls_token_if_available: bool = True,
        requires_scene: bool = False,
        learning_rate: float = 1e-4,  # Added LR for optimizer
        loss_criterion: str = "l1",  # Added loss criterion type
    ):
        """
        Initializes the combined IJEPAPlanningAgent.

        :param mlp_weights_path: Path to the pre-trained MLP weights file (used in initialize).
                                 Can be None if training from scratch.
        :param ijepa_model_id: Identifier for the I-JEPA model to load from Hugging Face.
        :param trajectory_sampling: Configuration for the output trajectory.
        :param use_cls_token_if_available: Whether to use the CLS token for features if available.
        :param requires_scene: Whether this agent requires scene data (passed to AbstractAgent).
        :param learning_rate: Learning rate for the optimizer.
        :param loss_criterion: Type of loss function to use ('l1' or 'mse').
        """
        # Pass trajectory_sampling and requires_scene to super().__init__
        super().__init__(
            trajectory_sampling=trajectory_sampling, requires_scene=requires_scene
        )

        self._mlp_weights_path_config = mlp_weights_path
        self._ijepa_model_id = ijepa_model_id
        self._use_cls_token = use_cls_token_if_available
        self._num_driving_commands = self.NUM_DRIVING_COMMANDS  # Use class constant
        self._learning_rate = learning_rate  # Store LR
        self._loss_criterion_type = loss_criterion  # Store loss type

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"DEBUG: Forcing device to {self._device}") # Keep if needed, but might be noisy

        # Models and Processor - initialized in initialize()
        self._processor: AutoProcessor = None
        self._ijepa_encoder: AutoModel = None
        self._feature_extraction_method = "Mean Pooling"  # Determined in initialize()

        # Define the MLP structure as a submodule
        self.mlp = nn.Sequential(
            nn.Linear(self.IJEP_DIM + self.EGO_DIM, self.HIDDEN_DIM),
            nn.ReLU(),
            nn.LayerNorm(self.HIDDEN_DIM),
            nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM),
            nn.ReLU(),
            nn.LayerNorm(self.HIDDEN_DIM),
            nn.Linear(
                self.HIDDEN_DIM, self.NUM_FUTURE_FRAMES * 3
            ),  # Output 3 coords (x, y, heading) per future frame
        )

        # Define the loss criterion
        if self._loss_criterion_type == "l1":
            self.criterion = nn.L1Loss()
        elif self._loss_criterion_type == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(
                f"Unsupported loss criterion: {self._loss_criterion_type}. Choose 'l1' or 'mse'."
            )

    def name(self) -> str:
        """
        :return: string describing name of this agent.
        """
        return "IJEPAPlanningAgent"

    def get_sensor_config(self) -> SensorConfig:
        """
        :return: Dataclass defining the sensor configuration for lidar and cameras.
        This agent requires the front camera (CAM_F0).
        """
        # Note: For training with NavsimTrajectoryDataset, this config is less critical
        # as the dataset loads data directly. But it's needed for NAVSIM evaluation.
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

    def initialize(self) -> None:
        """
        Initializes the agent by loading the I-JEPA encoder, processor,
        and optionally loading the pre-trained MLP weights.
        """
        print(f"Initializing {self.name()}...")
        print(f"Using device: {self._device}")

        # --- Load Models and Processor ---
        try:
            # Load processor first
            self._processor = AutoProcessor.from_pretrained(
                self._ijepa_model_id, use_fast=True
            )
            # Load I-JEPA encoder
            self._ijepa_encoder = AutoModel.from_pretrained(self._ijepa_model_id).to(
                self._device
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
        except Exception as e:
            raise RuntimeError(f"Error moving features to device: {e}") from e

        # --- Determine feature extraction method ---
        # Perform a dummy forward pass to check model output structure
        with torch.no_grad():
            try:
                dummy_input = torch.zeros(1, 3, 224, 224).to(self._device)
                outputs = self._ijepa_encoder(pixel_values=dummy_input)

                if (
                    self._use_cls_token
                    and hasattr(outputs, "pooler_output")
                    and outputs.pooler_output is not None
                ):
                    # Check dimension consistency
                    if outputs.pooler_output.shape[-1] == self.IJEP_DIM:
                        self._feature_extraction_method = "Pooler Output (CLS Token)"
                    else:
                        print(
                            f"Warning: I-JEPA pooler_output dimension {outputs.pooler_output.shape[-1]} \
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
                    self.mlp.load_state_dict(
                        torch.load(weights_path, map_location=self._device)
                    )
                    print("MLP weights loaded successfully.")
                except (RuntimeError, ValueError) as e:
                    print(
                        f"Warning: Failed to load MLP weights from {weights_path}: {e}. Starting training from scratch."
                    )
                    # No raise here, just start from default init if loading fails

        # Ensure MLP is on the correct device
        self.mlp.to(self._device)

        print(f"{self.name()} initialization complete.")

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Performs the forward pass through the I-JEPA encoder (frozen) and the MLP head.
        This method is used by the NAVSIM/PyTorch Lightning trainer during training
        and validation/testing steps.

        Expected features dictionary keys (depending on feature builders or data preparation):
        - 'image_pixel_values': Tensor containing pre-processed image pixel values (Batch, C, H, W)
        - 'ego_features': Tensor containing ego status features (Batch, Ego_Dim)

        :param features: Dictionary containing input features.
        :return: Dictionary containing model predictions, primarily 'trajectory'.
        """
        if self._ijepa_encoder is None or self._processor is None or self.mlp is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        # Ensure models are on the correct device
        self.mlp.to(self._device)
        self._ijepa_encoder.to(self._device)  # Should already be there after initialize

        # --- Extract data from features dictionary ---
        # These keys depend on how your training data is prepared or what
        # FeatureBuilders would output in a full NAVSIM training setup.
        # Based on your NavsimTrajectoryDataset, we expect pixel values and ego features.
        # Adjust keys if your actual feature builders/data preparation uses different names.
        try:
            # Assuming 'image_pixel_values' contains the tensor from processor output
            pixel_values = features["image_pixel_values"].to(self._device)
            ego_features_tensor = features["ego_features"].to(self._device)
        except KeyError as e:
            raise KeyError(
                f"Missing expected key in features dictionary: {e}. "
                "Ensure your data loading/feature building provides 'image_pixel_values' and 'ego_features'."
            ) from e
        except Exception as e:
            raise RuntimeError(f"Error moving features to device: {e}") from e

        # --- I-JEPA Feature Extraction (using the frozen encoder) ---
        # Use torch.no_grad() even though the encoder is frozen, as a safety measure
        # and for slight performance improvement during training's forward pass.
        with torch.no_grad():
            try:
                ijepa_outputs = self._ijepa_encoder(pixel_values=pixel_values)

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
                    # Perform mean pooling over the sequence dimension (usually dim 1)
                    visual_features = ijepa_outputs.last_hidden_state.mean(dim=1)
                else:
                    # Fallback or if method was 'Unknown'
                    if (
                        hasattr(ijepa_outputs, "last_hidden_state")
                        and ijepa_outputs.last_hidden_state is not None
                    ):
                        # print(f"Warning: Feature extraction method '{self._feature_extraction_method}' failed or unknown. Attempting Mean Pooling fallback.") # Keep silent during training forward
                        visual_features = ijepa_outputs.last_hidden_state.mean(dim=1)
                    else:
                        raise ValueError(
                            "Could not extract visual features from I-JEPA output."
                        )

                if visual_features is None:
                    raise ValueError("Visual features extraction resulted in None.")

                # Add sanity check for feature dimension
                if visual_features.shape[-1] != self.IJEP_DIM:
                    raise ValueError(
                        f"Extracted visual features dimension {visual_features.shape[-1]} does not match expected IJEP_DIM {self.IJEP_DIM}."
                    )

            except Exception as e:
                raise RuntimeError(
                    f"Error during I-JEPA feature extraction in forward pass: {e}"
                ) from e

        # --- MLP Forward Pass ---
        try:
            # Ensure ego features match expected EGO_DIM
            if ego_features_tensor.shape[-1] != self.EGO_DIM:
                raise ValueError(
                    f"Ego features dimension {ego_features_tensor.shape[-1]} does not match expected EGO_DIM {self.EGO_DIM}."
                )

            # Concatenate visual and ego features
            combined_features = torch.cat([visual_features, ego_features_tensor], dim=1)

            # Pass concatenated features through the MLP submodule
            flat_predictions = self.mlp(combined_features)

            # Reshape flat predictions into trajectory poses (Batch, num_poses, 3)
            # Assuming output is (batch_size, num_poses * 3) and ground truth is (batch_size, num_poses, 3)
            predicted_relative_poses_tensor = flat_predictions.view(
                -1, self.NUM_FUTURE_FRAMES, 3
            )

        except Exception as e:
            raise RuntimeError(f"Error during MLP forward pass in forward pass: {e}") from e

        # --- Return predictions dictionary ---
        # The key name here ('trajectory') should match what compute_loss expects
        # and what a TargetBuilder might produce for the ground truth ('trajectory_gt').
        # It's good practice to use consistent naming.
        predictions = {"trajectory": predicted_relative_poses_tensor}

        return predictions

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes the loss between predicted and ground truth trajectories.
        This method is used by the NAVSIM/PyTorch Lightning trainer.

        Expected targets dictionary keys (depending on target builders or data preparation):
        - 'trajectory_gt': Tensor containing ground truth trajectory poses (Batch, num_future_frames, 3)

        Expected predictions dictionary keys (output of self.forward):
        - 'trajectory': Tensor containing predicted trajectory poses (Batch, num_future_frames, 3)

        :param features: Dictionary of input features (can be used for conditional loss, etc., but not needed here).
        :param targets: Dictionary containing ground truth targets.
        :param predictions: Dictionary containing model predictions from self.forward().
        :return: Computed loss tensor.
        """
        if self.criterion is None:
            raise RuntimeError("Loss criterion not initialized.")

        # --- Extract ground truth and predictions from dictionaries ---
        # These keys should be consistent with your data loading/builders
        # and the keys used in the `forward` method's output.
        try:
            ground_truth_trajectory = targets["trajectory_gt"].to(self._device)
            predicted_trajectory = predictions["trajectory"].to(self._device)
        except KeyError as e:
            raise KeyError(
                f"Missing expected key in targets or predictions dictionary: {e}. "
                "Ensure your data loading/target building provides 'trajectory_gt' "
                "and that the forward method returns 'trajectory'."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Error moving targets/predictions to device in compute_loss: {e}"
            ) from e

        # --- Compute Loss ---
        # Ensure shapes match for loss computation
        if predicted_trajectory.shape != ground_truth_trajectory.shape:
            raise ValueError(
                f"Prediction shape {predicted_trajectory.shape} does not match ground truth shape {ground_truth_trajectory.shape} in compute_loss."
            )

        loss = self.criterion(predicted_trajectory, ground_truth_trajectory)

        return loss

    def get_optimizers(
        self,
    ) -> Union[
        torch.optim.Optimizer,
        Dict[str, Union[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]],
    ]:
        """
        Returns the optimizer for the trainable parameters (the MLP head).
        This method is called by the NAVSIM/PyTorch Lightning trainer.

        :return: A torch.optim.Optimizer instance for the MLP parameters.
        """
        if self.mlp is None:
            raise RuntimeError("MLP head not initialized.")

        # Only optimize the parameters of the MLP module
        optimizer = AdamW(self.mlp.parameters(), lr=self._learning_rate)

        # You can return a dictionary here if you had multiple optimizers/schedulers,
        # following PyTorch Lightning's conventions, but a single optimizer is sufficient.
        return optimizer

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """
        :return: List of feature builders required by this agent for training.
        Returning an empty list implies custom data loading or preparation
        before calling the agent's forward pass.
        """
        return []  # Not using NAVSIM's standard feature builders in this example

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """
        :return: List of target builders required by this agent for training.
        Returning an empty list implies custom data loading or preparation
        to provide the ground truth targets.
        """
        return []  # Not using NAVSIM's standard target builders in this example

    def get_training_callbacks(self) -> List[pl.Callback]:
        """
        Returns a list of pytorch-lightning callbacks.
        """
        # Add any relevant PyTorch Lightning callbacks here, e.g., ModelCheckpoint, EarlyStopping
        # return [pl.callbacks.ModelCheckpoint(monitor='val_loss')] # Example
        return []  # No specific callbacks required by the agent itself here

    # --- Removed the standalone `fit` method ---
    # The training loop will be handled by a PyTorch Lightning Trainer
    # and a LightningModule that wraps this agent and calls its methods.
    # If you need the standalone fit utility, keep it but rename it
    # (e.g., `_standalone_train_mlp`) to avoid confusion.
