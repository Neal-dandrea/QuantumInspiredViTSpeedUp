#!/usr/bin/env python3
"""
Full UMI Deployment in Isaac Sim
Runs trained diffusion policy with CuRobo IK on your Franka + K-cup scene

Run with:
~/.local/share/ov/pkg/isaac-sim-5.1.0/python.sh isaac_sim_main.py
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import sys
import os
import torch
import numpy as np
import cv2

# Add deployment directory to path
deployment_path = os.path.join(os.path.expanduser("~"), "universal_manipulation_interface", "deployment")
if deployment_path not in sys.path:
    sys.path.insert(0, deployment_path)

# Isaac Sim imports
from isaacsim.core import World
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.articulations import Articulation

# UMI deployment
from umi_deployment import UMIDeployment

print("=" * 70)
print(" UMI DIFFUSION POLICY DEPLOYMENT IN ISAAC SIM")
print("=" * 70)

# ============================================================================
# CONFIGURATION - ADJUST THESE FOR YOUR SCENE
# ============================================================================

# Path to your trained checkpoint
CHECKPOINT_PATH = os.path.expanduser(
    "~/universal_manipulation_interface/data/outputs/2026.02.20/"
    "18.27.00_train_diffusion_unet_timm_umi/checkpoints/epoch=0119-train_loss=0.01056.ckpt"
)

# Paths in your Isaac Sim scene
FRANKA_PRIM_PATH = "/World/Franka"  # Adjust if different
CAMERA_PRIM_PATH = "/World/Franka/panda_hand/camera"  # Wrist camera (adjust if needed)

# Control parameters
CONTROL_FREQUENCY = 10  # Hz (policy runs at 10 Hz)
MAX_STEPS = 500  # Maximum control steps
RENDER_FREQUENCY = 60  # Hz (rendering frequency)

# ============================================================================
# INITIALIZE SCENE
# ============================================================================

print("\nInitializing Isaac Sim scene...")
world = World()

# Check if scene already has ground plane, table, objects
stage = get_current_stage()

# Find Franka
franka_prim = get_prim_at_path(FRANKA_PRIM_PATH)
if not franka_prim.IsValid():
    print(f"\n✗ ERROR: Franka not found at {FRANKA_PRIM_PATH}")
    print("\nAvailable prims in scene:")
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if "Franka" in prim_path or "franka" in prim_path:
            print(f"  {prim_path}")
    simulation_app.close()
    exit(1)

print(f"✓ Found Franka at {FRANKA_PRIM_PATH}")

# Add Franka to scene
franka = world.scene.add(
    Articulation(
        prim_path=FRANKA_PRIM_PATH,
        name="franka"
    )
)

# ============================================================================
# SETUP CAMERA
# ============================================================================

def get_wrist_camera_image():
    """
    Get RGB image from wrist camera
    
    Returns:
        rgb_image: np.array([224, 224, 3]) uint8
    """
    # TODO: Replace with actual camera capture
    # For now, create dummy image
    # You'll need to implement this based on your camera setup
    
    # Option 1: If you have a Camera sensor
    # from isaacsim.sensor import Camera
    # camera = Camera(CAMERA_PRIM_PATH)
    # rgb = camera.get_rgba()[:, :, :3]
    
    # Option 2: Synthetic camera (placeholder)
    rgb_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    return rgb_image

print("⚠ WARNING: Using dummy camera images")
print("   You need to implement get_wrist_camera_image() for real deployment")

# ============================================================================
# LOAD UMI POLICY
# ============================================================================

print(f"\nLoading UMI policy from checkpoint...")
controller = UMIDeployment(
    checkpoint_path=CHECKPOINT_PATH,
    device="cuda:0"
)

# ============================================================================
# INITIALIZE ROBOT STATE
# ============================================================================

world.reset()
print("\n✓ Simulation initialized")

# Get initial robot state
initial_joints = franka.get_joint_positions()[:7]
print(f"Initial joint positions: {initial_joints}")

# Initialize controller with robot state
# TODO: Get actual end-effector pose from FK
initial_ee_pos = np.array([0.5, 0.0, 0.4])  # Approximate
initial_ee_rot = np.array([0.0, 0.0, 0.0])  # Approximate
initial_gripper = 0.04  # Approximate

controller.reset(
    ee_pos=initial_ee_pos,
    ee_rot=initial_ee_rot,
    gripper_width=initial_gripper,
    joint_positions=initial_joints
)

print("\n✓ Controller initialized")

# ============================================================================
# CONTROL LOOP
# ============================================================================

print("\n" + "=" * 70)
print(" STARTING CONTROL LOOP")
print("=" * 70)
print(f"Control frequency: {CONTROL_FREQUENCY} Hz")
print(f"Max steps: {MAX_STEPS}")
print("\nPress Ctrl+C or close window to stop")
print("=" * 70 + "\n")

step_count = 0
control_dt = 1.0 / CONTROL_FREQUENCY
render_steps_per_control = int(RENDER_FREQUENCY / CONTROL_FREQUENCY)

try:
    while simulation_app.is_running() and step_count < MAX_STEPS:
        
        # Get camera image
        rgb_image = get_wrist_camera_image()
        
        # Run controller
        target_joints, target_gripper, info = controller.step(rgb_image)
        
        # Check status
        if info['status'] == 'warming_up':
            print(f"Step {step_count}: Warming up (need 2 observations)...")
        
        elif info['status'] == 'ik_failed':
            print(f"Step {step_count}: IK FAILED - skipping")
            # Keep current position
        
        elif info['status'] == 'success':
            # Apply joint positions
            franka.set_joint_positions(target_joints)
            
            # Apply gripper (convert width to finger positions)
            # Franka gripper: each finger moves half the width
            finger_positions = np.array([target_gripper/2, target_gripper/2])
            # TODO: Apply gripper positions if you have gripper control
            # franka.gripper.set_joint_positions(finger_positions)
            
            # Print progress
            if step_count % 10 == 0:
                stats = controller.get_stats()
                print(f"Step {step_count:3d} | "
                      f"Success rate: {stats['success_rate']*100:.1f}% | "
                      f"EE pos: [{stats['current_ee_pos'][0]:.3f}, "
                      f"{stats['current_ee_pos'][1]:.3f}, "
                      f"{stats['current_ee_pos'][2]:.3f}] | "
                      f"Gripper: {stats['current_gripper']*100:.1f}cm")
        
        # Step simulation (render multiple times for smooth visualization)
        for _ in range(render_steps_per_control):
            world.step(render=True)
        
        step_count += 1

except KeyboardInterrupt:
    print("\n\nControl loop interrupted by user")

# ============================================================================
# CLEANUP
# ============================================================================

print("\n" + "=" * 70)
print(" FINAL STATISTICS")
print("=" * 70)

final_stats = controller.get_stats()
print(f"Total steps: {final_stats['total_steps']}")
print(f"IK failures: {final_stats['ik_failures']}")
print(f"Success rate: {final_stats['success_rate']*100:.1f}%")
print(f"Final EE position: {final_stats['current_ee_pos']}")
print(f"Final gripper width: {final_stats['current_gripper']*100:.1f} cm")

print("\n" + "=" * 70)
print(" DEPLOYMENT COMPLETE")
print("=" * 70)

# Keep window open for inspection
print("\nSimulation window will stay open. Close to exit.")
while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()