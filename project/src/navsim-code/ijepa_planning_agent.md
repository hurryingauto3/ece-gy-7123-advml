Okay, building an agent based on the I-JEPA paper within the NAVSIM framework is an interesting challenge. Here's a breakdown of how you could approach it, focusing on leveraging the pre-trained semantic representations from I-JEPA for the downstream planning task in NAVSIM.

**Understanding the Challenge:**

*   **I-JEPA is a Self-Supervised Pre-training Method:** Its primary goal is to learn general-purpose image representations without labels by predicting representations of masked patches.
*   **NAVSIM Agent Goal is Planning:** The agent needs to take sensor input (images, LiDAR, ego status) and predict a future ego-vehicle trajectory.
*   **The Bridge:** You need to use the features learned by a pre-trained I-JEPA model (specifically, its encoder) as input to a *planning module* that predicts the trajectory within the NAVSIM agent structure.

**Approach: Pre-train I-JEPA then Fine-tune for Planning**

This is the most practical approach:

1.  **Pre-train I-JEPA:** Train an I-JEPA model (likely a Vision Transformer, ViT) on a large dataset of images (e.g., ImageNet, or potentially images extracted from NAVSIM scenarios if feasible and permitted). The key output you need from this step is the saved weights of the **I-JEPA Context Encoder**.
2.  **Build a NAVSIM Agent:** Create a new agent class in NAVSIM that *loads* the pre-trained I-JEPA context encoder and uses its output features, along with ego status, to predict the trajectory using a separate "planning head".
3.  **Fine-tune:** Train this NAVSIM agent on the NAVSIM planning task. The loss will be based on the trajectory prediction accuracy, not the I-JEPA prediction objective. You might freeze the I-JEPA encoder initially and only train the planning head, or fine-tune the entire network (encoder + head) with a lower learning rate for the encoder.

**Step-by-Step Implementation in NAVSIM:**

**Step 1: Pre-train I-JEPA (External)**

*   This step happens *outside* the standard NAVSIM training scripts.
*   Follow the I-JEPA paper and any available reference implementations. Use a library like PyTorch Lightning or Timm.
*   Choose a ViT architecture compatible with NAVSIM's data (consider patch size and input resolution).
*   Train on your chosen image dataset using the I-JEPA objective (predicting target representations from context).
*   **Crucially:** Save the state dictionary of the trained **Context Encoder (`f_theta`)**. Let's call the path to this file `PATH_TO_PRETRAINED_IJEPA_ENCODER.pth`.

**Step 2: Define the NAVSIM Agent (`IJEPAPlanningAgent`)**

Create a new file, e.g., `navsim/agents/ijepa_planning_agent.py`:

```python
import torch
import timm # Or your preferred library for ViT models
from torch import nn
from typing import Any, Dict, List, Optional, Union
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
# You'll likely reuse or adapt these from other agents:
from navsim.agents.ego_status_mlp_agent import EgoStatusFeatureBuilder, TrajectoryTargetBuilder

# --- Define a Feature Builder for Camera Input ---
class IJEPACameraFeatureBuilder(AbstractFeatureBuilder):
    """Input feature builder for Camera data suitable for IJEPA's ViT encoder."""

    def __init__(self, image_size=(256, 1024), selected_cameras=["cam_f0"]): # Example size/cameras
        """
        Initializes the feature builder.
        :param image_size: Expected input size (H, W) for the pre-trained ViT.
        :param selected_cameras: List of camera names to use (e.g., ["cam_f0", "cam_l0", "cam_r0"]).
                                  Adjust based on your pre-training and desired input.
        """
        self._image_size = image_size
        self._selected_cameras = selected_cameras
        # Define necessary preprocessing (resizing, normalization)
        # IMPORTANT: This MUST match the preprocessing used during I-JEPA pre-training!
        self._transform = transforms.Compose([
            transforms.Resize(self._image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Example Imagenet stats
        ])

    def get_unique_name(self) -> str:
        return "ijepa_camera_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Extracts and preprocesses camera image(s)."""
        # --- Logic to select, potentially stitch, and preprocess images ---
        # Example: Using only the front camera from the latest timestep
        latest_cameras = agent_input.cameras[-1]
        if "cam_f0" in self._selected_cameras:
             # Assuming image is numpy array HWC, convert to PIL for torchvision
            img_pil = Image.fromarray(latest_cameras.cam_f0.image)
            processed_image = self._transform(img_pil)
        else:
            # Handle cases with other/multiple cameras
            raise NotImplementedError("Camera selection/stitching not fully implemented")
            # Example for stitching (needs PIL conversion and careful handling):
            # l0 = Image.fromarray(latest_cameras.cam_l0.image[...]) # Apply cropping if needed
            # f0 = Image.fromarray(latest_cameras.cam_f0.image[...])
            # r0 = Image.fromarray(latest_cameras.cam_r0.image[...])
            # stitched = ... # Combine images (e.g., np.concatenate then Image.fromarray)
            # processed_image = self._transform(stitched)


        # You might need to handle multiple cameras or stitching depending
        # on how your I-JEPA encoder was pre-trained or how you want to use it.
        # The key is that the output tensor shape matches the encoder's input.
        return {"camera_features": processed_image}


# --- Define the NAVSIM Agent ---
class IJEPAPlanningAgent(AbstractAgent):
    """NAVSIM agent using a pre-trained I-JEPA encoder."""

    def __init__(
        self,
        pretrained_encoder_path: str,
        encoder_arch: str = "vit_base_patch16_224", # Example ViT architecture
        freeze_encoder: bool = True,
        planning_head_dims: List[int] = [512, 256], # Example MLP dimensions
        lr: float = 1e-4,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
        # Add other necessary parameters (e.g., weight decay)
    ):
        super().__init__(trajectory_sampling)
        self._pretrained_encoder_path = pretrained_encoder_path
        self._encoder_arch = encoder_arch
        self._freeze_encoder = freeze_encoder
        self._planning_head_dims = planning_head_dims
        self._lr = lr

        # 1. Define the I-JEPA Encoder
        # Use timm or another library to instantiate the ViT architecture
        # Note: Ensure patch size, image size etc., match the pre-trained model
        self._encoder = timm.create_model(
            self._encoder_arch,
            pretrained=False, # We'll load weights manually
            num_classes=0 # We need features, not classification logits
        )
        # Get the feature dimension output by the encoder (depends on ViT variant)
        encoder_feature_dim = self._encoder.embed_dim # Check this for your specific ViT

        # 2. Define the Planning Head
        # Example: An MLP that takes encoder features + ego status features
        # Input dim: encoder_feature_dim + ego_status_feature_dim (usually 8: vel(2)+accel(2)+cmd(4))
        ego_status_dim = 8 # velocity(2) + acceleration(2) + driving_command(4) - check EgoStatusFeatureBuilder
        planning_input_dim = encoder_feature_dim + ego_status_dim
        head_layers = []
        current_dim = planning_input_dim
        for hidden_dim in self._planning_head_dims:
            head_layers.extend([nn.Linear(current_dim, hidden_dim), nn.ReLU()])
            current_dim = hidden_dim
        # Output layer: num_poses * 3 (x, y, heading)
        head_layers.append(nn.Linear(current_dim, self._trajectory_sampling.num_poses * 3))
        self._planning_head = nn.Sequential(*head_layers)

        # 3. Define which sensors are needed
        # Must align with IJEPACameraFeatureBuilder
        self._sensor_config = SensorConfig(
            cam_f0=[3], # Example: Need only current front camera (index 3 in history)
            cam_l0=False, # Adjust based on IJEPACameraFeatureBuilder
            cam_r0=False, # Adjust based on IJEPACameraFeatureBuilder
            # ... other sensors usually False unless needed by planning head directly
            lidar_pc=False, # I-JEPA typically uses images
        )

    def name(self) -> str:
        return "IJEPAPlanningAgent"

    def initialize(self) -> None:
        """Load pre-trained encoder weights and potentially freeze."""
        print(f"Loading pre-trained I-JEPA encoder weights from: {self._pretrained_encoder_path}")
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._pretrained_encoder_path)
        else:
            state_dict: Dict[str, Any] = torch.load(self._pretrained_encoder_path, map_location=torch.device("cpu"))

        # Adjust keys if necessary (e.g., if saved within a LightningModule)
        # This might need inspection of the saved .pth file keys
        encoder_state_dict = {}
        for k, v in state_dict.items():
             # Example: remove prefix like 'context_encoder.' or 'model.encoder.'
            if k.startswith("context_encoder."):
                 encoder_state_dict[k.replace("context_encoder.", "")] = v
            elif k.startswith("encoder."): # Common timm prefix if saved directly
                 encoder_state_dict[k.replace("encoder.", "")] = v
            elif not (k.startswith("predictor.") or k.startswith("target_encoder.")): # Skip other I-JEPA parts
                 encoder_state_dict[k] = v # Assume it's already the encoder dict

        # Handle potential missing/unexpected keys
        load_result = self._encoder.load_state_dict(encoder_state_dict, strict=False)
        print(f"Encoder weight loading result: {load_result}")
        if load_result.missing_keys:
            print(f"Warning: Missing keys during encoder weight loading: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"Warning: Unexpected keys during encoder weight loading: {load_result.unexpected_keys}")


        if self._freeze_encoder:
            print("Freezing I-JEPA encoder weights.")
            for param in self._encoder.parameters():
                param.requires_grad = False
        else:
            print("I-JEPA encoder weights will be fine-tuned.")

    def get_sensor_config(self) -> SensorConfig:
        return self._sensor_config # Return the config defined in __init__

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Return feature builders for camera and ego status."""
        return [
            IJEPACameraFeatureBuilder(selected_cameras=["cam_f0"]), # Match sensor config
            EgoStatusFeatureBuilder() # Reuse standard ego status builder
        ]

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Return standard trajectory target builder."""
        return [TrajectoryTargetBuilder(trajectory_sampling=self._trajectory_sampling)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass: encode image, combine with status, predict trajectory."""
        camera_data = features["camera_features"]
        ego_status_data = features["ego_status"].to(torch.float32)

        # 1. Get Image Features from I-JEPA Encoder
        # The exact way to get features depends on the ViT implementation (timm)
        # Usually involves forward_features or getting the output before the head
        with torch.no_grad() if self._freeze_encoder else torch.enable_grad():
            # For many timm ViTs, forward_features gives patch embeddings + cls token
            # Or just call the encoder - need to check the specific model output
            # image_features = self._encoder.forward_features(camera_data)
            image_features = self._encoder(camera_data) # Simpler call, check output shape

            # Often, we take the class token or average pool the patch tokens
            # Assuming output is [batch, num_tokens, embed_dim], take CLS token (index 0)
            # Or if no CLS token: image_features = image_features.mean(dim=1)
            if image_features.ndim == 3 and image_features.shape[1] > 1: # Check if patch tokens exist
                 image_cls_features = image_features[:, 0] # Assumes CLS token is first
            elif image_features.ndim == 2: # Already pooled or flattened
                 image_cls_features = image_features
            else:
                 raise ValueError(f"Unexpected I-JEPA encoder output shape: {image_features.shape}")


        # 2. Combine Features
        combined_features = torch.cat([image_cls_features, ego_status_data], dim=-1)

        # 3. Predict Trajectory with Planning Head
        trajectory_flat = self._planning_head(combined_features)
        predicted_trajectory = trajectory_flat.view(-1, self._trajectory_sampling.num_poses, 3)

        # Ensure heading is within [-pi, pi] (optional, depends on loss/activation)
        # predicted_trajectory[..., 2] = torch.tanh(predicted_trajectory[..., 2]) * torch.pi

        return {"trajectory": predicted_trajectory}

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute standard trajectory L1 loss."""
        return torch.nn.functional.l1_loss(predictions["trajectory"], targets["trajectory"])

    def get_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Configure optimizer for trainable parameters (planning head and optionally encoder)."""
        params_to_optimize = list(self._planning_head.parameters())
        if not self._freeze_encoder:
            # Add encoder parameters with potentially lower LR
             params_to_optimize.extend(self._encoder.parameters())
             # Example: Different LR for encoder vs head (more complex setup)
             # optimizer = torch.optim.Adam([
             #     {'params': self._planning_head.parameters(), 'lr': self._lr},
             #     {'params': self._encoder.parameters(), 'lr': self._lr * 0.1} # Fine-tuning LR
             # ])
             # return optimizer

        optimizer = torch.optim.Adam(params_to_optimize, lr=self._lr)
        # You might add a learning rate scheduler here if needed
        # return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Return list of callbacks for training (optional)."""
        # Example: Could add visualization callbacks if needed
        return []
```

**Step 3: Configure the Agent (Hydra)**

Create a config file, e.g., `navsim/planning/script/config/common/agent/ijepa_planning_agent.yaml`:

```yaml
# @package _group_
agent_name: ijepa_planning_agent
_target_: navsim.agents.ijepa_planning_agent.IJEPAPlanningAgent

# --- Parameters for your IJEPAPlanningAgent ---
pretrained_encoder_path: ??? # REQUIRED: Set path to your .pth file
encoder_arch: vit_base_patch16_224  # IMPORTANT: Match the pre-trained model
freeze_encoder: true              # Start by freezing, maybe unfreeze later
planning_head_dims: [512, 256, 128] # Example MLP layers for planning head
lr: 1e-4                          # Learning rate for training
# trajectory_sampling: (defaults are usually okay)
```

**Step 4: Train the Agent**

Use the standard NAVSIM training script (`run_training.py`), selecting your new agent configuration:

```bash
export NAVSIM_DEVKIT_ROOT=/path/to/navsim-devkit
export NAVSIM_PLANNING_ROOT=${NAVSIM_DEVKIT_ROOT}/navsim/planning
export NAVSIM_EXP_ROOT=/path/to/experiments # Your experiment output dir

python ${NAVSIM_PLANNING_ROOT}/script/run_training.py \
    hydra.run.dir=${NAVSIM_EXP_ROOT}/ijepa_training \
    agent=ijepa_planning_agent \
    train_test_split=navtrain \
    worker=single_machine_thread_pool \
    experiment_name=ijepa_planning_train \
    py_func=train \
    +trainer={max_epochs: 50} # Adjust epochs
    # Add overrides for pretrained_encoder_path if not set in yaml
    # agent.pretrained_encoder_path=/path/to/your/encoder.pth
```

**Step 5: Evaluate the Agent**

Use the NAVSIM evaluation scripts (e.g., `run_pdm_score.py`) after training, making sure to point to the checkpoint saved during the training in Step 4.

```bash
export CHECKPOINT=/path/to/experiments/ijepa_training/checkpoints/last.ckpt # Or best checkpoint

# First, ensure metric cache exists (run scripts/evaluation/run_metric_caching.sh if needed)

python ${NAVSIM_PLANNING_ROOT}/script/run_pdm_score.py \
    hydra.run.dir=${NAVSIM_EXP_ROOT}/ijepa_eval \
    agent=ijepa_planning_agent \
    train_test_split=navtest \
    worker=single_machine_thread_pool \
    experiment_name=ijepa_planning_eval \
    py_func=main \
    agent.checkpoint_path=${CHECKPOINT} \
    # Ensure pretrained_encoder_path is also correct if needed at evaluation
    # agent.pretrained_encoder_path=/path/to/your/encoder.pth
```

**Key Considerations and Challenges:**

1.  **Pre-training:** The quality of the pre-trained I-JEPA encoder is critical. Getting this right requires careful implementation and substantial compute resources.
2.  **Encoder Architecture:** Ensure the ViT architecture (patch size, image input size) used in NAVSIM matches the one used for pre-training *exactly*.
3.  **Preprocessing:** Image resizing, normalization, etc., in `IJEPACameraFeatureBuilder` *must* match the preprocessing used during I-JEPA pre-training.
4.  **Feature Extraction:** How you extract features from the ViT encoder (CLS token vs. average pooling) can impact performance. Experimentation might be needed.
5.  **Planning Head:** The design of the planning head (MLP, Transformer Decoder, etc.) is important. An MLP is simpler to start with.
6.  **Fine-tuning Strategy:** Deciding whether to freeze the encoder or fine-tune it (and with what learning rate) is a key hyperparameter. Start by freezing the encoder.
7.  **Input Modalities:** I-JEPA primarily uses images. Integrating LiDAR or map information effectively might require architectural changes to the planning head or fusing features differently. The example above only uses image features + ego status.

This detailed guide provides a solid starting point for integrating I-JEPA's learned representations into a NAVSIM planning agent. Remember that the success heavily depends on the quality of the pre-trained encoder and careful integration into the planning pipeline.