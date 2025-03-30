import torch
import torch.nn as nn
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder, AbstractTargetBuilder
)

class NaiveMLPAgent(AbstractAgent):
    def __init__(self):
        super().__init__()
        # A small MLP that takes a feature vector of size D (e.g. velocity, accel, driving command)
        # and outputs a future trajectory.
        self.mlp = nn.Sequential(
            nn.Linear(4, 128),  # example: (vx, ax, command_left, command_right?), adjust dimension
            nn.ReLU(),
            nn.Linear(128, 30)  # e.g. 10 timesteps * (x, y, heading) = 30 if T=10
        )

    def name(self):
        return "NaiveMLPAgent"

    def initialize(self):
        pass  # load weights if needed

    def get_sensor_config(self) -> SensorConfig:
        """
        We don't need cameras or LiDAR for this naive agent:
        Just set them to False.
        """
        return SensorConfig(
            cam_f0=False, cam_r0=False, cam_l0=False, cam_b0=False,
            lidar_merged=False
        )

    def compute_trajectory(self, agent_input) -> Trajectory:
        """
        Called during evaluation in a non-learning-based agent. 
        We'll do a quick forward pass in eval mode.
        """
        with torch.no_grad():
            # gather velocity, acceleration, command from agent_input.ego
            vx = agent_input.ego.velocity[0].item()  # x-velocity
            ax = agent_input.ego.acceleration[0].item()
            command = agent_input.ego.command  # e.g. 0=left,1=straight,2=right ?

            # produce a small vector
            # be consistent with how you do one-hot or numeric command
            features = torch.tensor([vx, ax, float(command==0), float(command==2)]).unsqueeze(0) 
            pred = self.mlp(features)  # => shape [1, 30]
            pred = pred.view(10, 3)   # => 10 timesteps, each (x, y, heading)
            # Convert to a navsim Trajectory
            trajectory = your_postprocess_to_trajectory(pred)
            return trajectory

    # If you want to do "learning-based agent" training with features/targets:
    def get_feature_builders(self):
        return [EgoStatusFeatureBuilder()]

    def get_target_builders(self):
        return [TrajectoryTargetBuilder()]

    def forward(self, features: dict) -> dict:
        # features["ego_status"]: shape [B, 4]
        x = features["ego_status"]
        out = self.mlp(x)  # => [B, 30]
        out = out.view(out.size(0), 10, 3) # T=10 steps
        return {"trajectory": out}

    def compute_loss(self, features, targets, predictions):
        gt_trajectory = targets["gt_trajectory"]  # => [B, 10, 3]
        pred_trajectory = predictions["trajectory"]  # => [B, 10, 3]
        loss = nn.functional.mse_loss(pred_trajectory, gt_trajectory)
        return loss

    def get_optimizers(self):
        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=1e-4)
        return optimizer

# Feature & Target Builders
class EgoStatusFeatureBuilder(AbstractFeatureBuilder):
    def get_feature_names(self):
        return ["ego_status"]

    def build_features(self, agent_input):
        vx = agent_input.ego.velocity[0].item()
        ax = agent_input.ego.acceleration[0].item()
        command = agent_input.ego.command  # integer: 0=left,1=straight,2=right
        # create a [4]-dim vector
        feature = [vx, ax, float(command==0), float(command==2)]
        return {"ego_status": torch.tensor(feature, dtype=torch.float)}

class TrajectoryTargetBuilder(AbstractTargetBuilder):
    def get_target_names(self):
        return ["gt_trajectory"]

    def build_targets(self, scene, agent_input):
        # produce the ground-truth future trajectory in local coords, e.g. shape [T,3].
        # e.g. scene => you can get the ground-truth future from the reference frames
        gt_traj = your_get_future_trajectory(scene, agent_input)
        return {"gt_trajectory": gt_traj}