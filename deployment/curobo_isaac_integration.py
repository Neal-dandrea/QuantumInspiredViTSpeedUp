#!/usr/bin/env python3
"""
Isaac Sim + CuRobo Integration
Deploy your trained UMI policy with CuRobo IK solver

Run from Isaac Sim Python:
~/.local/share/ov/pkg/isaac_sim-5.1.0/python.sh curobo_isaac_integration.py
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import torch
import numpy as np
from scipy.spatial.transform import Rotation
import cv2

# Isaac Sim imports
from isaacsim.core import World
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.articulations import Articulation

# CuRobo imports
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.util_file import get_robot_configs_path, join_path, load_yaml

print("=" * 60)
print("CuRobo + Isaac Sim Integration")
print("=" * 60)

# Initialize World
world = World()
world.scene.add_default_ground_plane()

# Get your existing Franka robot
# Adjust this path to match YOUR scene
franka_prim_path = "/World/Franka"  # Change if different

# Check if Franka exists
stage = get_current_stage()
franka_prim = get_prim_at_path(franka_prim_path)

if not franka_prim.IsValid():
    print(f"✗ Franka not found at {franka_prim_path}")
    print("\nAvailable prims:")
    from pxr import Usd
    for prim in stage.Traverse():
        print(f"  {prim.GetPath()}")
    simulation_app.close()
    exit(1)

print(f"✓ Found Franka at {franka_prim_path}")

# Add Franka to scene
franka = world.scene.add(
    Articulation(
        prim_path=franka_prim_path,
        name="franka"
    )
)

# Initialize CuRobo IK Solver
print("\nInitializing CuRobo IK solver...")
tensor_args = TensorDeviceType(device=torch.device("cuda:0"))

# Load Franka config
robot_cfg_path = get_robot_configs_path()
franka_cfg_file = join_path(robot_cfg_path, "franka.yml")
robot_cfg = load_yaml(franka_cfg_file)

ik_config = IKSolverConfig.load_from_robot_config(
    robot_cfg,
    tensor_args=tensor_args,
    num_seeds=20,
    position_threshold=0.005,
    rotation_threshold=0.05
)

ik_solver = IKSolver(ik_config)
print("✓ CuRobo IK solver ready")

# Helper functions
def axis_angle_to_quaternion(axis_angle):
    """Convert axis-angle [rx, ry, rz] to quaternion [w, x, y, z]"""
    rot = Rotation.from_rotvec(axis_angle)
    quat = rot.as_quat()  # Returns [x, y, z, w]
    return np.array([quat[3], quat[0], quat[1], quat[2]])  # Return [w, x, y, z]

def solve_ik(target_pos, target_rot_axis_angle, current_joints):
    """
    Solve IK using CuRobo
    
    Args:
        target_pos: [x, y, z] in meters
        target_rot_axis_angle: [rx, ry, rz] in radians
        current_joints: [7] current joint positions
    
    Returns:
        target_joints: [7] joint positions, or None if failed
    """
    # Convert to quaternion
    target_quat = axis_angle_to_quaternion(target_rot_axis_angle)
    
    # Create pose
    goal_pose = Pose(
        position=torch.tensor([target_pos], device="cuda:0", dtype=torch.float32),
        quaternion=torch.tensor([target_quat], device="cuda:0", dtype=torch.float32)
    )
    
    # Solve IK (non-differentiable mode)
    with torch.no_grad():
        result = ik_solver.solve_single(
            goal_pose=goal_pose,
            seed_config=torch.tensor(current_joints, device="cuda:0", dtype=torch.float32)
        )
    
    if result.success[0]:
        return result.solution[0].cpu().numpy()
    else:
        print(f"⚠ IK failed: pos_err={result.position_error[0].item()*1000:.1f}mm")
        return None

# Reset simulation
world.reset()
print("\n✓ Simulation initialized")

# Get initial state
initial_joints = franka.get_joint_positions()
print(f"\nInitial joint positions: {initial_joints[:7]}")

# Test IK with simple motion
print("\n" + "=" * 60)
print("Testing CuRobo IK in Isaac Sim")
print("=" * 60)

# Current end-effector state
current_ee_pos = np.array([0.5, 0.0, 0.4])  # Initial position
current_ee_rot = np.array([0.0, 0.0, 0.0])  # Initial rotation (axis-angle)
current_joints = initial_joints[:7]

# Test motion: move right, then forward, then up
test_motions = [
    ("Move RIGHT", np.array([0.0, 0.2, 0.0]), np.array([0.0, 0.0, 0.0])),
    ("Move FORWARD", np.array([0.1, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
    ("Move UP", np.array([0.0, 0.0, 0.1]), np.array([0.0, 0.0, 0.0])),
]

for motion_name, delta_pos, delta_rot in test_motions:
    print(f"\n{motion_name}:")
    
    # Apply delta
    target_pos = current_ee_pos + delta_pos
    target_rot = current_ee_rot + delta_rot
    
    print(f"  Target: pos={target_pos}, rot={target_rot}")
    
    # Solve IK
    target_joints = solve_ik(target_pos, target_rot, current_joints)
    
    if target_joints is not None:
        print(f"  ✓ IK solution found")
        
        # Apply to robot
        franka.set_joint_positions(target_joints)
        
        # Step simulation
        for _ in range(60):  # 1 second at 60 Hz
            world.step(render=True)
        
        # Update state
        current_ee_pos = target_pos
        current_ee_rot = target_rot
        current_joints = target_joints
        
    else:
        print(f"  ✗ IK solution failed, skipping")

print("\n" + "=" * 60)
print("CuRobo Integration Test Complete!")
print("=" * 60)
print("\nIf you saw the robot move, CuRobo is working!")
print("\nNext steps:")
print("  1. Integrate your trained UMI policy")
print("  2. Add wrist camera")
print("  3. Run full handover task")

# Keep simulation running
print("\nSimulation running. Close window to exit.")
while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()