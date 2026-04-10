"""
UMI Deployment with Lula IK
Uses Isaac Sim's built-in Lula instead of CuRobo
"""

import torch
import numpy as np
from scipy.spatial.transform import Rotation


class UMIDeploymentLula:
    """Deploy UMI diffusion policy with Lula IK solver"""
    
    def __init__(self, checkpoint_path, rmpflow, device="cuda:0"):
        """
        Args:
            checkpoint_path: Path to trained .ckpt file
            rmpflow: Lula RmpFlow instance from Isaac Sim
            device: CUDA device
        """
        print("=" * 60)
        print("Initializing UMI Deployment with Lula")
        print("=" * 60)
        
        self.device = device
        
        # Load trained policy
        import sys
        import os
        umi_path = os.path.expanduser("~/universal_manipulation_interface")
        if umi_path not in sys.path:
            sys.path.insert(0, umi_path)
        
        from diffusion_policy.policy.diffusion_unet_timm_policy import DiffusionUnetTimmPolicy
        
        print(f"Loading policy from: {checkpoint_path}")
        self.policy = DiffusionUnetTimmPolicy.load_from_checkpoint(checkpoint_path)
        self.policy.eval()
        self.policy = self.policy.to(device)
        print("✓ UMI policy loaded")
        
        # Lula IK solver
        self.rmpflow = rmpflow
        print("✓ Using Lula IK solver")
        
        # State tracking
        self.current_ee_pos = np.array([0.3, 0.0, 0.5])
        self.current_ee_rot = np.array([0.0, 0.0, 0.0])
        self.current_gripper = 0.04
        self.current_joints = np.zeros(7)
        
        # Observation buffer
        self.obs_buffer = None
        
        # Statistics
        self.step_count = 0
        self.ik_failures = 0
        
        print("✓ UMI Deployment ready")
    
    def reset(self, ee_pos=None, ee_rot=None, gripper_width=None, joint_positions=None):
        """Reset controller state"""
        if ee_pos is not None:
            self.current_ee_pos = np.array(ee_pos)
        if ee_rot is not None:
            self.current_ee_rot = np.array(ee_rot)
        if gripper_width is not None:
            self.current_gripper = gripper_width
        if joint_positions is not None:
            self.current_joints = np.array(joint_positions)
        
        self.obs_buffer = None
        self.step_count = 0
        self.ik_failures = 0
    
    def prepare_observation(self, rgb_image):
        """Prepare observation for policy"""
        if self.obs_buffer is None:
            self.obs_buffer = {
                'camera0_rgb': [],
                'robot0_eef_pos': [],
                'robot0_eef_rot_axis_angle': [],
                'robot0_gripper_width': []
            }
        
        # Add current observation
        self.obs_buffer['camera0_rgb'].append(rgb_image)
        self.obs_buffer['robot0_eef_pos'].append(self.current_ee_pos)
        self.obs_buffer['robot0_eef_rot_axis_angle'].append(self.current_ee_rot)
        self.obs_buffer['robot0_gripper_width'].append(np.array([self.current_gripper]))
        
        # Keep last 2
        for key in self.obs_buffer:
            if len(self.obs_buffer[key]) > 2:
                self.obs_buffer[key] = self.obs_buffer[key][-2:]
        
        # Need exactly 2 observations
        if len(self.obs_buffer['camera0_rgb']) < 2:
            return None
        
        obs = {
            'camera0_rgb': torch.tensor(
                np.stack(self.obs_buffer['camera0_rgb'][-2:]), 
                dtype=torch.float32
            ).unsqueeze(0),
            'robot0_eef_pos': torch.tensor(
                np.stack(self.obs_buffer['robot0_eef_pos'][-2:]),
                dtype=torch.float32
            ).unsqueeze(0),
            'robot0_eef_rot_axis_angle': torch.tensor(
                np.stack(self.obs_buffer['robot0_eef_rot_axis_angle'][-2:]),
                dtype=torch.float32
            ).unsqueeze(0),
            'robot0_gripper_width': torch.tensor(
                np.stack(self.obs_buffer['robot0_gripper_width'][-2:]),
                dtype=torch.float32
            ).unsqueeze(0)
        }
        
        return obs
    
    def step(self, rgb_image):
        """Run one control step with Lula IK"""
        # Prepare observation
        obs = self.prepare_observation(rgb_image)
        
        if obs is None:
            return None, None, {
                'status': 'warming_up',
                'obs_count': len(self.obs_buffer['camera0_rgb'])
            }
        
        # Move to device
        for key in obs:
            obs[key] = obs[key].to(self.device)
        
        # Run policy (NO GRADIENTS)
        with torch.no_grad():
            action_dict = self.policy.predict_action(obs)
            cartesian_action = action_dict['action'][0, 0].cpu().numpy()
        
        # Parse action
        delta_pos = cartesian_action[:3]
        delta_rot = cartesian_action[3:6]
        target_gripper = cartesian_action[6]
        
        # Apply deltas
        target_pos = self.current_ee_pos + delta_pos
        target_rot = self.current_ee_rot + delta_rot
        
        # Convert axis-angle to quaternion for Lula
        rot = Rotation.from_rotvec(target_rot)
        quat = rot.as_quat()  # [x, y, z, w]
        target_quat = np.array([quat[3], quat[0], quat[1], quat[2]])  # [w, x, y, z]
        
        # Solve IK with Lula
        from lula.kinematics import compute_ik
        
        target_joints = compute_ik(
            rmpflow=self.rmpflow,
            target_position=target_pos,
            target_orientation=target_quat,
            seed=self.current_joints
        )
        
        if target_joints is None:
            self.ik_failures += 1
            return None, None, {
                'status': 'ik_failed',
                'target_pos': target_pos,
                'target_rot': target_rot
            }
        
        # Update state
        self.current_ee_pos = target_pos
        self.current_ee_rot = target_rot
        self.current_gripper = target_gripper
        self.current_joints = np.array(target_joints)
        
        self.step_count += 1
        
        info = {
            'status': 'success',
            'step': self.step_count,
            'ik_failures': self.ik_failures,
            'cartesian_action': cartesian_action,
            'target_ee_pos': target_pos,
            'target_ee_rot': target_rot,
            'target_gripper': target_gripper
        }
        
        return np.array(target_joints), target_gripper, info
    
    def get_stats(self):
        """Get statistics"""
        return {
            'total_steps': self.step_count,
            'ik_failures': self.ik_failures,
            'success_rate': (self.step_count - self.ik_failures) / max(self.step_count, 1),
            'current_ee_pos': self.current_ee_pos,
            'current_ee_rot': self.current_ee_rot,
            'current_gripper': self.current_gripper
        }