"""
UMI RL Controller
Differentiable controller for RL fine-tuning of trained diffusion policy

Use this LATER when you're ready for RL training
For now, use umi_deployment.py for non-differentiable deployment
"""

import torch
import torch.nn as nn
import numpy as np
from umi_controller_base import (
    setup_curobo_ik,
    axis_angle_to_quaternion,
    load_umi_policy
)
from curobo.types.math import Pose


class UMIControllerRL(nn.Module):
    """
    Differentiable UMI controller for RL fine-tuning
    Gradients flow through: Policy → IK → Actions
    """
    
    def __init__(self, checkpoint_path, device="cuda:0", freeze_policy=True):
        """
        Initialize RL-ready controller
        
        Args:
            checkpoint_path: Path to trained .ckpt file
            device: CUDA device (default: "cuda:0")
            freeze_policy: If True, only train RL correction layer (recommended)
        """
        super().__init__()
        
        print("=" * 60)
        print("Initializing UMI RL Controller")
        print("=" * 60)
        
        self.device = device
        
        # Load trained diffusion policy
        self.policy = load_umi_policy(checkpoint_path)
        self.policy = self.policy.to(device)
        
        # Optionally freeze policy weights (train only RL correction)
        if freeze_policy:
            print("Freezing diffusion policy weights (training RL correction only)")
            for param in self.policy.parameters():
                param.requires_grad = False
        else:
            print("Policy weights unfrozen (end-to-end RL training)")
        
        # Setup CuRobo IK solver
        self.ik_solver = setup_curobo_ik(device=device)
        
        # RL correction network
        # Learns residual corrections to diffusion policy outputs
        obs_dim = 768  # ViT embedding dimension
        action_dim = 7  # Cartesian action dimension
        
        self.rl_correction = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)  # Output: correction to Cartesian action
        )
        
        # Initialize small corrections
        nn.init.zeros_(self.rl_correction[-1].weight)
        nn.init.zeros_(self.rl_correction[-1].bias)
        
        # State tracking (NOT trainable)
        self.register_buffer('current_ee_pos', torch.zeros(3))
        self.register_buffer('current_ee_rot', torch.zeros(3))
        self.register_buffer('current_joints', torch.zeros(7))
        
        # Statistics
        self.step_count = 0
        self.ik_failures = 0
        
        print("✓ UMI RL Controller ready")
        print(f"  Trainable params: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def reset_state(self, ee_pos, ee_rot, joint_positions):
        """
        Reset controller state (call at episode start)
        
        Args:
            ee_pos: torch.Tensor([3]) end-effector position
            ee_rot: torch.Tensor([3]) end-effector rotation (axis-angle)
            joint_positions: torch.Tensor([7]) joint positions
        """
        self.current_ee_pos = ee_pos.clone().detach()
        self.current_ee_rot = ee_rot.clone().detach()
        self.current_joints = joint_positions.clone().detach()
        
        self.step_count = 0
        self.ik_failures = 0
    
    def forward(self, obs):
        """
        Forward pass with gradients enabled (for RL training)
        
        Args:
            obs: dict with keys:
                - 'camera0_rgb': torch.Tensor([B, 2, 224, 224, 3])
                - 'robot0_eef_pos': torch.Tensor([B, 2, 3])
                - 'robot0_eef_rot_axis_angle': torch.Tensor([B, 2, 3])
                - 'robot0_gripper_width': torch.Tensor([B, 2, 1])
        
        Returns:
            target_joints: torch.Tensor([B, 7]) target joint positions
            info: dict with auxiliary information
        """
        batch_size = obs['camera0_rgb'].shape[0]
        
        # 1. Run diffusion policy (gradients enabled if not frozen)
        action_dict = self.policy.predict_action(obs)
        cartesian_actions = action_dict['action'][:, 0]  # [B, 7] first action
        
        # 2. Get observation embedding for RL correction
        obs_embedding = self.policy.obs_encoder(obs)  # [B, 768]
        
        # 3. Compute RL correction
        combined = torch.cat([obs_embedding, cartesian_actions], dim=-1)
        correction = self.rl_correction(combined)  # [B, 7]
        
        # 4. Apply correction to policy action
        corrected_actions = cartesian_actions + correction
        
        # 5. Parse actions
        delta_pos = corrected_actions[:, :3]    # [B, 3]
        delta_rot = corrected_actions[:, 3:6]   # [B, 3]
        target_gripper = corrected_actions[:, 6]  # [B]
        
        # 6. Apply deltas to current pose (keep gradients!)
        target_pos = self.current_ee_pos.unsqueeze(0) + delta_pos  # [B, 3]
        target_rot = self.current_ee_rot.unsqueeze(0) + delta_rot  # [B, 3]
        
        # 7. Convert to quaternion (keep gradients!)
        target_quats = []
        for i in range(batch_size):
            quat = axis_angle_to_quaternion(target_rot[i].detach().cpu().numpy())
            target_quats.append(torch.tensor(quat, device=self.device))
        target_quats = torch.stack(target_quats)  # [B, 4]
        
        # 8. Create poses for CuRobo
        goal_poses = Pose(
            position=target_pos,
            quaternion=target_quats
        )
        
        # 9. Solve IK (DIFFERENTIABLE - gradients flow!)
        current_joints_batch = self.current_joints.unsqueeze(0).repeat(batch_size, 1)
        
        ik_result = self.ik_solver.solve_batch(
            goal_poses=goal_poses,
            seed_config=current_joints_batch
        )
        
        target_joints = ik_result.solution  # [B, 7] - gradients preserved!
        
        # 10. Update state (detach for tracking)
        self.current_ee_pos = target_pos[0].detach()
        self.current_ee_rot = target_rot[0].detach()
        self.current_joints = target_joints[0].detach()
        
        self.step_count += 1
        
        # Return info
        info = {
            'cartesian_actions': cartesian_actions,
            'corrections': correction,
            'corrected_actions': corrected_actions,
            'ik_success': ik_result.success,
            'ik_position_error': ik_result.position_error,
            'ik_rotation_error': ik_result.rotation_error,
            'target_gripper': target_gripper,
            'obs_embedding': obs_embedding
        }
        
        return target_joints, info
    
    def get_trainable_params(self):
        """Get parameters for optimizer"""
        return [p for p in self.parameters() if p.requires_grad]
    
    def get_stats(self):
        """Get controller statistics"""
        return {
            'total_steps': self.step_count,
            'ik_failures': self.ik_failures,
            'current_ee_pos': self.current_ee_pos.cpu().numpy(),
            'current_ee_rot': self.current_ee_rot.cpu().numpy(),
            'current_joints': self.current_joints.cpu().numpy()
        }


class PPOAgent:
    """
    Simple PPO agent for RL fine-tuning
    Wraps the UMI controller for training
    """
    
    def __init__(self, controller, learning_rate=3e-4):
        """
        Initialize PPO agent
        
        Args:
            controller: UMIControllerRL instance
            learning_rate: Learning rate for optimizer
        """
        self.controller = controller
        
        # Optimizer (only for RL correction network)
        self.optimizer = torch.optim.Adam(
            controller.get_trainable_params(),
            lr=learning_rate
        )
        
        # PPO hyperparameters
        self.clip_ratio = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        
        print(f"PPO Agent initialized with {len(controller.get_trainable_params())} trainable params")
    
    def compute_loss(self, obs, actions, returns, old_log_probs, advantages):
        """
        Compute PPO loss
        
        Args:
            obs: Observation batch
            actions: Action batch (joint angles)
            returns: Discounted returns
            old_log_probs: Old action log probabilities
            advantages: Advantage estimates
        
        Returns:
            loss: Total PPO loss
            info: Loss breakdown
        """
        # Forward pass
        pred_actions, info = self.controller(obs)
        
        # Compute log probabilities (assuming Gaussian policy)
        action_mean = pred_actions
        action_std = 0.1  # Fixed std for now
        
        action_dist = torch.distributions.Normal(action_mean, action_std)
        log_probs = action_dist.log_prob(actions).sum(dim=-1)
        
        # PPO clip objective
        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        
        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()
        
        # Value loss (predict returns)
        # TODO: Add value network if needed
        value_loss = 0.0
        
        # Entropy bonus (encourage exploration)
        entropy = action_dist.entropy().sum(dim=-1).mean()
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        loss_info = {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss,
            'entropy': entropy.item(),
            'ik_success_rate': info['ik_success'].float().mean().item()
        }
        
        return loss, loss_info
    
    def update(self, obs, actions, returns, old_log_probs, advantages):
        """
        Perform one PPO update
        
        Args:
            obs: Observation batch
            actions: Action batch
            returns: Discounted returns
            old_log_probs: Old log probs
            advantages: Advantage estimates
        
        Returns:
            loss_info: Dictionary with loss breakdown
        """
        self.optimizer.zero_grad()
        
        loss, loss_info = self.compute_loss(
            obs, actions, returns, old_log_probs, advantages
        )
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.controller.get_trainable_params(),
            max_norm=0.5
        )
        
        self.optimizer.step()
        
        return loss_info


if __name__ == "__main__":
    """
    Test RL controller initialization
    This doesn't train - just verifies the setup
    """
    import sys
    
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else \
        "~/universal_manipulation_interface/data/outputs/2026.02.20/" \
        "18.27.00_train_diffusion_unet_timm_umi/checkpoints/epoch=0119-train_loss=0.01056.ckpt"
    
    print("\n" + "=" * 60)
    print("Testing RL Controller Setup")
    print("=" * 60)
    
    # Initialize controller
    controller = UMIControllerRL(
        checkpoint_path=checkpoint_path,
        device="cuda:0",
        freeze_policy=True  # Recommended: only train RL correction
    )
    
    # Initialize PPO agent
    agent = PPOAgent(controller, learning_rate=3e-4)
    
    # Test forward pass
    print("\nTesting forward pass...")
    
    # Dummy observation
    batch_size = 4
    obs = {
        'camera0_rgb': torch.randn(batch_size, 2, 224, 224, 3, device="cuda:0"),
        'robot0_eef_pos': torch.randn(batch_size, 2, 3, device="cuda:0"),
        'robot0_eef_rot_axis_angle': torch.randn(batch_size, 2, 3, device="cuda:0"),
        'robot0_gripper_width': torch.randn(batch_size, 2, 1, device="cuda:0")
    }
    
    # Reset state
    controller.reset_state(
        ee_pos=torch.tensor([0.5, 0.0, 0.4], device="cuda:0"),
        ee_rot=torch.tensor([0.0, 0.0, 0.0], device="cuda:0"),
        joint_positions=torch.zeros(7, device="cuda:0")
    )
    
    # Forward pass
    with torch.no_grad():
        target_joints, info = controller(obs)
    
    print(f"\n✓ Forward pass successful!")
    print(f"  Output shape: {target_joints.shape}")
    print(f"  IK success rate: {info['ik_success'].float().mean().item()*100:.1f}%")
    print(f"  Correction magnitude: {info['corrections'].abs().mean().item():.6f}")
    
    # Test backward pass (verify gradients)
    print("\nTesting backward pass...")
    
    target_joints, info = controller(obs)
    loss = target_joints.sum()  # Dummy loss
    loss.backward()
    
    # Check gradients
    has_grads = any(p.grad is not None for p in controller.get_trainable_params())
    
    if has_grads:
        print("✓ Backward pass successful! Gradients computed.")
    else:
        print("✗ No gradients computed!")
    
    print("\n" + "=" * 60)
    print("RL Controller Test Complete")
    print("=" * 60)
    print("\nController is ready for RL training!")
    print("Next: Implement environment and training loop")