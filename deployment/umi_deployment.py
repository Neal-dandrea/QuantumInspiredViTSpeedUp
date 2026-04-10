"""
UMI Deployment Controller
Non-differentiable deployment of trained diffusion policy with CuRobo IK

Use this for initial deployment and validation (no RL)
"""

import torch
import numpy as np
from umi_controller_base import (
    setup_curobo_ik,
    solve_ik_single,
    prepare_observation,
    load_umi_policy,
    axis_angle_to_quaternion
)


class UMIDeployment:
    """
    Deploy trained UMI diffusion policy with CuRobo IK
    Non-differentiable mode - suitable for deployment
    """
    
    def __init__(self, checkpoint_path, device="cuda:0"):
        """
        Initialize deployment controller
        
        Args:
            checkpoint_path: Path to trained .ckpt file
            device: CUDA device (default: "cuda:0")
        """
        print("=" * 60)
        print("Initializing UMI Deployment Controller")
        print("=" * 60)
        
        self.device = device
        
        # Load trained policy
        self.policy = load_umi_policy(checkpoint_path)
        self.policy = self.policy.to(device)
        
        # Setup CuRobo IK solver
        self.ik_solver = setup_curobo_ik(device=device)
        
        # State tracking
        self.current_ee_pos = np.array([0.3, 0.0, 0.5])  # Initial EE position
        self.current_ee_rot = np.array([0.0, 0.0, 0.0])  # Initial EE rotation (axis-angle)
        self.current_gripper = 0.04  # Initial gripper width (meters)
        self.current_joints = np.zeros(7)  # Initial joint positions
        
        # Observation buffer (needs last 2 observations)
        self.obs_buffer = None
        
        # Statistics
        self.step_count = 0
        self.ik_failures = 0
        
        print("✓ UMI Deployment Controller ready")
    
    def reset(self, ee_pos=None, ee_rot=None, gripper_width=None, joint_positions=None):
        """
        Reset controller state
        
        Args:
            ee_pos: Initial end-effector position (optional)
            ee_rot: Initial end-effector rotation (optional)
            gripper_width: Initial gripper width (optional)
            joint_positions: Initial joint positions (optional)
        """
        if ee_pos is not None:
            self.current_ee_pos = np.array(ee_pos)
        if ee_rot is not None:
            self.current_ee_rot = np.array(ee_rot)
        if gripper_width is not None:
            self.current_gripper = gripper_width
        if joint_positions is not None:
            self.current_joints = np.array(joint_positions)
        
        # Clear observation buffer
        self.obs_buffer = None
        self.step_count = 0
        self.ik_failures = 0
    
    def step(self, rgb_image):
        """
        Run one control step
        
        Args:
            rgb_image: np.array([224, 224, 3]) RGB image from wrist camera
        
        Returns:
            target_joints: np.array([7]) target joint positions, or None if failed
            target_gripper: float, target gripper width
            info: dict with step information
        """
        # Prepare observation
        obs, self.obs_buffer = prepare_observation(
            rgb_image=rgb_image,
            ee_pos=self.current_ee_pos,
            ee_rot=self.current_ee_rot,
            gripper_width=self.current_gripper,
            obs_buffer=self.obs_buffer
        )
        
        # Need 2 observations to start
        if obs is None:
            return None, None, {'status': 'warming_up', 'obs_count': len(self.obs_buffer['camera0_rgb'])}
        
        # Move obs to device
        for key in obs:
            obs[key] = obs[key].to(self.device)
        
        # Run diffusion policy (NO GRADIENTS)
        with torch.no_grad():
            action_dict = self.policy.predict_action(obs)
            # Get first action from sequence [B, 6, 7] -> [7]
            cartesian_action = action_dict['action'][0, 0].cpu().numpy()
        
        # Parse action
        delta_pos = cartesian_action[:3]    # [Δx, Δy, Δz]
        delta_rot = cartesian_action[3:6]   # [Δrx, Δry, Δrz]
        target_gripper = cartesian_action[6]  # gripper_width
        
        # Apply deltas to current pose
        target_pos = self.current_ee_pos + delta_pos
        target_rot = self.current_ee_rot + delta_rot
        
        # Solve IK (NO GRADIENTS)
        target_joints, ik_info = solve_ik_single(
            ik_solver=self.ik_solver,
            target_pos=target_pos,
            target_rot_axis_angle=target_rot,
            current_joints=self.current_joints,
            device=self.device
        )
        
        # Check IK success
        if target_joints is None:
            self.ik_failures += 1
            print(f"⚠ IK failed at step {self.step_count}: "
                  f"pos_err={ik_info['position_error']*1000:.1f}mm, "
                  f"rot_err={np.rad2deg(ik_info['rotation_error']):.1f}deg")
            
            return None, None, {
                'status': 'ik_failed',
                'ik_info': ik_info,
                'target_pos': target_pos,
                'target_rot': target_rot
            }
        
        # Update state
        self.current_ee_pos = target_pos
        self.current_ee_rot = target_rot
        self.current_gripper = target_gripper
        self.current_joints = target_joints
        
        self.step_count += 1
        
        # Return info
        info = {
            'status': 'success',
            'step': self.step_count,
            'ik_failures': self.ik_failures,
            'ik_info': ik_info,
            'cartesian_action': cartesian_action,
            'target_ee_pos': target_pos,
            'target_ee_rot': target_rot,
            'target_gripper': target_gripper
        }
        
        return target_joints, target_gripper, info
    
    def get_stats(self):
        """Get controller statistics"""
        return {
            'total_steps': self.step_count,
            'ik_failures': self.ik_failures,
            'success_rate': (self.step_count - self.ik_failures) / max(self.step_count, 1),
            'current_ee_pos': self.current_ee_pos,
            'current_ee_rot': self.current_ee_rot,
            'current_gripper': self.current_gripper
        }


if __name__ == "__main__":
    # Test controller initialization
    checkpoint_path = "~/universal_manipulation_interface/data/outputs/2026.02.20/18.27.00_train_diffusion_unet_timm_umi/checkpoints/epoch=0119-train_loss=0.01056.ckpt"
    
    controller = UMIDeployment(checkpoint_path)
    
    print("\n" + "=" * 60)
    print("Controller Test")
    print("=" * 60)
    
    # Simulate RGB image
    rgb_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # First step (warming up)
    result = controller.step(rgb_image)
    print(f"Step 1: {result[2]['status']}")
    
    # Second step (should work)
    result = controller.step(rgb_image)
    print(f"Step 2: {result[2]['status']}")
    
    if result[0] is not None:
        print(f"✓ Target joints: {result[0]}")
        print(f"✓ Target gripper: {result[1]:.4f}")
    
    print("\n✓ Controller test complete")