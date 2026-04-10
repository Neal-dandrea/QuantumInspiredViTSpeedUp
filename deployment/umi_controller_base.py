"""
UMI Controller Base - Shared Utilities
Shared code for both deployment and RL training
"""

import torch
import numpy as np
from scipy.spatial.transform import Rotation
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.util_file import get_robot_configs_path, join_path, load_yaml


def setup_curobo_ik(device="cuda:0", num_seeds=20):
    """
    Setup CuRobo IK solver for Franka Panda
    
    Args:
        device: CUDA device (default: "cuda:0")
        num_seeds: Number of IK seeds to try (default: 20)
    
    Returns:
        IKSolver instance
    """
    print("Initializing CuRobo IK solver...")
    
    # Setup tensor device
    tensor_args = TensorDeviceType(device=torch.device(device))
    
    # Load Franka config
    robot_cfg_path = get_robot_configs_path()
    franka_cfg_file = join_path(robot_cfg_path, "franka.yml")
    robot_cfg = load_yaml(franka_cfg_file)
    
    # Create IK solver config
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        tensor_args=tensor_args,
        num_seeds=num_seeds,
        position_threshold=0.005,  # 5mm
        rotation_threshold=0.05,   # ~3 degrees
    )
    
    # Create IK solver
    ik_solver = IKSolver(ik_config)
    
    print("✓ CuRobo IK solver ready")
    return ik_solver


def axis_angle_to_quaternion(axis_angle):
    """
    Convert axis-angle rotation to quaternion
    
    Args:
        axis_angle: np.array([rx, ry, rz]) in radians
    
    Returns:
        quaternion: np.array([w, x, y, z])
    """
    if isinstance(axis_angle, torch.Tensor):
        axis_angle = axis_angle.cpu().numpy()
    
    # Handle batch dimension
    if len(axis_angle.shape) == 1:
        # Single axis-angle
        rot = Rotation.from_rotvec(axis_angle)
        quat = rot.as_quat()  # [x, y, z, w]
        return np.array([quat[3], quat[0], quat[1], quat[2]])  # [w, x, y, z]
    else:
        # Batch of axis-angles
        rot = Rotation.from_rotvec(axis_angle)
        quat = rot.as_quat()  # [N, 4] in [x, y, z, w]
        # Convert to [w, x, y, z]
        return np.concatenate([quat[:, 3:4], quat[:, :3]], axis=1)


def quaternion_to_axis_angle(quaternion):
    """
    Convert quaternion to axis-angle rotation
    
    Args:
        quaternion: np.array([w, x, y, z])
    
    Returns:
        axis_angle: np.array([rx, ry, rz]) in radians
    """
    if isinstance(quaternion, torch.Tensor):
        quaternion = quaternion.cpu().numpy()
    
    # Handle batch dimension
    if len(quaternion.shape) == 1:
        # Single quaternion [w, x, y, z]
        quat_scipy = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])  # [x, y, z, w]
        rot = Rotation.from_quat(quat_scipy)
        return rot.as_rotvec()
    else:
        # Batch of quaternions
        quat_scipy = np.concatenate([quaternion[:, 1:], quaternion[:, 0:1]], axis=1)  # [x, y, z, w]
        rot = Rotation.from_quat(quat_scipy)
        return rot.as_rotvec()


def solve_ik_single(ik_solver, target_pos, target_rot_axis_angle, current_joints, device="cuda:0"):
    """
    Solve IK for a single target pose (non-differentiable)
    
    Args:
        ik_solver: CuRobo IKSolver instance
        target_pos: np.array([x, y, z]) in meters
        target_rot_axis_angle: np.array([rx, ry, rz]) in radians
        current_joints: np.array([7]) current joint positions
        device: CUDA device
    
    Returns:
        target_joints: np.array([7]) joint positions, or None if failed
        info: dict with IK solve info
    """
    # Convert to quaternion
    target_quat = axis_angle_to_quaternion(target_rot_axis_angle)
    
    # Create pose
    goal_pose = Pose(
        position=torch.tensor([target_pos], device=device, dtype=torch.float32),
        quaternion=torch.tensor([target_quat], device=device, dtype=torch.float32)
    )
    
    # Solve IK (non-differentiable)
    with torch.no_grad():
        result = ik_solver.solve_single(
            goal_pose=goal_pose,
            seed_config=torch.tensor(current_joints, device=device, dtype=torch.float32)
        )
    
    info = {
        'success': result.success[0].item(),
        'position_error': result.position_error[0].item(),
        'rotation_error': result.rotation_error[0].item(),
    }
    
    if result.success[0]:
        return result.solution[0].cpu().numpy(), info
    else:
        return None, info


def solve_ik_batch(ik_solver, target_positions, target_rot_axis_angles, current_joints_batch, device="cuda:0"):
    """
    Solve IK for a batch of target poses (non-differentiable)
    
    Args:
        ik_solver: CuRobo IKSolver instance
        target_positions: np.array([B, 3]) positions in meters
        target_rot_axis_angles: np.array([B, 3]) rotations in radians
        current_joints_batch: np.array([B, 7]) current joint positions
        device: CUDA device
    
    Returns:
        target_joints: np.array([B, 7]) joint positions
        success_mask: np.array([B]) boolean success mask
    """
    batch_size = len(target_positions)
    
    # Convert to quaternions
    target_quats = axis_angle_to_quaternion(target_rot_axis_angles)
    
    # Create batch poses
    goal_poses = Pose(
        position=torch.tensor(target_positions, device=device, dtype=torch.float32),
        quaternion=torch.tensor(target_quats, device=device, dtype=torch.float32)
    )
    
    # Solve IK (non-differentiable)
    with torch.no_grad():
        result = ik_solver.solve_batch(
            goal_poses=goal_poses,
            seed_config=torch.tensor(current_joints_batch, device=device, dtype=torch.float32)
        )
    
    return result.solution.cpu().numpy(), result.success.cpu().numpy()


def prepare_observation(rgb_image, ee_pos, ee_rot, gripper_width, obs_buffer=None):
    """
    Prepare observation dictionary for UMI policy
    
    Args:
        rgb_image: np.array([224, 224, 3]) RGB image
        ee_pos: np.array([3]) end-effector position
        ee_rot: np.array([3]) end-effector rotation (axis-angle)
        gripper_width: float, gripper width in meters
        obs_buffer: Optional buffer to maintain observation history
    
    Returns:
        obs: dict ready for policy input
        updated_buffer: updated observation buffer
    """
    if obs_buffer is None:
        obs_buffer = {
            'camera0_rgb': [],
            'robot0_eef_pos': [],
            'robot0_eef_rot_axis_angle': [],
            'robot0_gripper_width': []
        }
    
    # Add current observation
    obs_buffer['camera0_rgb'].append(rgb_image)
    obs_buffer['robot0_eef_pos'].append(ee_pos)
    obs_buffer['robot0_eef_rot_axis_angle'].append(ee_rot)
    obs_buffer['robot0_gripper_width'].append(np.array([gripper_width]))
    
    # Keep last 2 observations
    for key in obs_buffer:
        if len(obs_buffer[key]) > 2:
            obs_buffer[key] = obs_buffer[key][-2:]
    
    # Format for policy (needs exactly 2 observations)
    if len(obs_buffer['camera0_rgb']) < 2:
        return None, obs_buffer
    
    obs = {
        'camera0_rgb': torch.tensor(
            np.stack(obs_buffer['camera0_rgb'][-2:]), 
            dtype=torch.float32
        ).unsqueeze(0),  # [1, 2, 224, 224, 3]
        
        'robot0_eef_pos': torch.tensor(
            np.stack(obs_buffer['robot0_eef_pos'][-2:]),
            dtype=torch.float32
        ).unsqueeze(0),  # [1, 2, 3]
        
        'robot0_eef_rot_axis_angle': torch.tensor(
            np.stack(obs_buffer['robot0_eef_rot_axis_angle'][-2:]),
            dtype=torch.float32
        ).unsqueeze(0),  # [1, 2, 3]
        
        'robot0_gripper_width': torch.tensor(
            np.stack(obs_buffer['robot0_gripper_width'][-2:]),
            dtype=torch.float32
        ).unsqueeze(0)   # [1, 2, 1]
    }
    
    return obs, obs_buffer


def load_umi_policy(checkpoint_path):
    """
    Load trained UMI diffusion policy
    
    Args:
        checkpoint_path: Path to .ckpt file
    
    Returns:
        policy: Loaded policy in eval mode
    """
    import sys
    import os
    
    # Add UMI to path
    umi_path = os.path.expanduser("~/universal_manipulation_interface")
    if umi_path not in sys.path:
        sys.path.insert(0, umi_path)
    
    from diffusion_policy.policy.diffusion_unet_timm_policy import DiffusionUnetTimmPolicy
    
    print(f"Loading policy from: {checkpoint_path}")
    policy = DiffusionUnetTimmPolicy.load_from_checkpoint(checkpoint_path)
    policy.eval()  # Set to evaluation mode
    
    print("✓ UMI policy loaded")
    return policy