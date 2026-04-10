#!/usr/bin/env python3
"""
Test CuRobo basic IK solving for Franka Panda
Run this OUTSIDE Isaac Sim first to verify CuRobo works
"""

import torch
import numpy as np

print("=" * 60)
print("Testing CuRobo Installation")
print("=" * 60)

try:
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import RobotConfig
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml
    print("✓ CuRobo imports successful")
except ImportError as e:
    print(f"✗ CuRobo import failed: {e}")
    print("\nInstall with: pip install curobo")
    exit(1)

# Check CUDA
if torch.cuda.is_available():
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    device = "cuda:0"
else:
    print("⚠ CUDA not available, using CPU (will be slow)")
    device = "cpu"

# Setup tensor device
tensor_args = TensorDeviceType(device=torch.device(device))

# Load Franka robot config
print("\nLoading Franka robot configuration...")
try:
    robot_cfg_path = get_robot_configs_path()
    print(f"Robot configs path: {robot_cfg_path}")
    
    # Load Franka config
    franka_cfg_file = join_path(robot_cfg_path, "franka.yml")
    robot_cfg = load_yaml(franka_cfg_file)
    print(f"✓ Loaded Franka config from: {franka_cfg_file}")
    
except Exception as e:
    print(f"✗ Failed to load Franka config: {e}")
    print("\nTrying alternative method...")
    
    # Alternative: use direct config
    robot_cfg = RobotConfig.from_basic(
        "franka_panda",
        tensor_args=tensor_args
    )
    print("✓ Using built-in Franka config")

# Create IK solver config
print("\nInitializing IK solver...")
ik_config = IKSolverConfig.load_from_robot_config(
    robot_cfg,
    tensor_args=tensor_args,
    num_seeds=20,  # Number of random seeds to try
    position_threshold=0.005,  # 5mm position tolerance
    rotation_threshold=0.05,   # ~3° rotation tolerance
)

# Create IK solver
ik_solver = IKSolver(ik_config)
print("✓ IK solver initialized")

# Test IK solve
print("\n" + "=" * 60)
print("Testing IK Solve")
print("=" * 60)

# Target pose: reach forward
target_position = torch.tensor([[0.5, 0.0, 0.4]], device=device, dtype=torch.float32)
# Quaternion [w, x, y, z] - pointing down
target_quaternion = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device, dtype=torch.float32)

goal_pose = Pose(
    position=target_position,
    quaternion=target_quaternion
)

# Initial joint configuration (home position)
seed_config = torch.tensor(
    [[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]],  # Franka home pose
    device=device,
    dtype=torch.float32
)

print(f"\nTarget position: {target_position[0].cpu().numpy()}")
print(f"Target quaternion: {target_quaternion[0].cpu().numpy()}")
print(f"Seed config: {seed_config[0].cpu().numpy()}")

# Solve IK
print("\nSolving IK...")
result = ik_solver.solve_single(
    goal_pose=goal_pose,
    seed_config=seed_config[0]
)

if result.success[0]:
    solution = result.solution[0].cpu().numpy()
    position_error = result.position_error[0].item()
    rotation_error = result.rotation_error[0].item()
    
    print("\n✓ IK SOLUTION FOUND!")
    print(f"  Joint angles (rad): {solution}")
    print(f"  Joint angles (deg): {np.rad2deg(solution)}")
    print(f"  Position error: {position_error*1000:.2f} mm")
    print(f"  Rotation error: {np.rad2deg(rotation_error):.2f} deg")
else:
    print("\n✗ IK solution failed")
    print(f"  Position error: {result.position_error[0].item()*1000:.2f} mm")
    print(f"  Rotation error: {np.rad2deg(result.rotation_error[0].item()):.2f} deg")

# Test batch IK (for RL later)
print("\n" + "=" * 60)
print("Testing Batch IK (for future RL)")
print("=" * 60)

batch_size = 4
batch_positions = torch.tensor([
    [0.5, 0.0, 0.4],
    [0.5, 0.2, 0.4],
    [0.5, -0.2, 0.4],
    [0.3, 0.0, 0.5],
], device=device, dtype=torch.float32)

batch_quaternions = torch.tensor([
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
], device=device, dtype=torch.float32)

batch_goal_poses = Pose(
    position=batch_positions,
    quaternion=batch_quaternions
)

batch_seed = seed_config.repeat(batch_size, 1)

print(f"\nSolving {batch_size} IK problems simultaneously...")
batch_result = ik_solver.solve_batch(
    goal_poses=batch_goal_poses,
    seed_config=batch_seed
)

successful = batch_result.success.sum().item()
print(f"✓ {successful}/{batch_size} solutions found")

for i in range(batch_size):
    if batch_result.success[i]:
        pos_err = batch_result.position_error[i].item() * 1000
        print(f"  Solution {i+1}: pos_error={pos_err:.2f}mm ✓")
    else:
        print(f"  Solution {i+1}: FAILED ✗")

print("\n" + "=" * 60)
print("CuRobo Test Complete!")
print("=" * 60)
print("\nNext step: Integrate with Isaac Sim scene")