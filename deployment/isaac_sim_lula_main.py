#!/usr/bin/env python3
"""
UMI Deployment with Lula IK (Isaac Sim built-in)
No CuRobo needed - uses Isaac Sim's native Lula solver

Run with:
~/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh isaac_sim_lula_main.py
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import sys
import os
import numpy as np

# Add deployment to path
deployment_path = os.path.join(os.path.expanduser("~"), "universal_manipulation_interface", "deployment")
if deployment_path not in sys.path:
    sys.path.insert(0, deployment_path)

# Isaac Sim imports - FIXED FOR 5.1.0
from isaacsim.core.api import World
from isaacsim.core.api import Articulation  # FIXED: Import from api directly
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.nucleus import get_assets_root_path

# Lula imports
try:
    from lula import RmpFlow, RobotDescription
except ImportError:
    print("⚠ Lula not found in standard location")
    print("Trying alternative import...")
    try:
        from isaacsim.robot_motion.lula import RmpFlow, RobotDescription
    except ImportError:
        print("✗ Could not import Lula")
        print("Lula may need to be imported differently in Isaac Sim 5.1.0")
        simulation_app.close()
        exit(1)

# UMI deployment
from lula_deployment import UMIDeploymentLula

print("=" * 70)
print(" UMI DEPLOYMENT WITH LULA IK")
print("=" * 70)

# Configuration
CHECKPOINT_PATH = os.path.expanduser(
    "~/universal_manipulation_interface/data/outputs/2026.02.20/"
    "18.27.00_train_diffusion_unet_timm_umi/checkpoints/epoch=0119-train_loss=0.01056.ckpt"
)

FRANKA_PRIM_PATH = "/World/Franka"
CONTROL_FREQUENCY = 10  # Hz
MAX_STEPS = 500

# Initialize scene
print("\nInitializing scene...")
world = World()
stage = get_current_stage()

# Find Franka
franka_prim = get_prim_at_path(FRANKA_PRIM_PATH)
if not franka_prim.IsValid():
    print(f"✗ Franka not found at {FRANKA_PRIM_PATH}")
    print("\nAvailable prims:")
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if "Franka" in prim_path or "franka" in prim_path.lower():
            print(f"  {prim_path}")
    simulation_app.close()
    exit(1)

print(f"✓ Found Franka at {FRANKA_PRIM_PATH}")

# Add Franka to scene
franka = world.scene.add(Articulation(prim_path=FRANKA_PRIM_PATH, name="franka"))

# Setup Lula IK
print("\nInitializing Lula IK solver...")

# Load Franka robot description
assets_root = get_assets_root_path()
robot_description_path = assets_root + "/Isaac/Robots/Franka/franka_descriptor.yaml"
rmpflow_config_path = assets_root + "/Isaac/Robots/Franka/rmpflow/franka_rmpflow_config.yaml"

print(f"Robot description: {robot_description_path}")
print(f"RMPflow config: {rmpflow_config_path}")

robot_description = RobotDescription.load(robot_description_path)
rmpflow = RmpFlow(robot_description)
rmpflow.load_from_config(rmpflow_config_path)

print("✓ Lula IK solver ready")

# Load UMI policy
print("\nLoading UMI policy...")
controller = UMIDeploymentLula(
    checkpoint_path=CHECKPOINT_PATH,
    rmpflow=rmpflow,
    device="cuda:0"
)

# Initialize
world.reset()
print("\n✓ Simulation initialized")

initial_joints = franka.get_joint_positions()[:7]
print(f"Initial joints: {initial_joints}")

controller.reset(
    ee_pos=np.array([0.5, 0.0, 0.4]),
    ee_rot=np.array([0.0, 0.0, 0.0]),
    gripper_width=0.04,
    joint_positions=initial_joints
)

# Control loop
print("\n" + "=" * 70)
print(" STARTING CONTROL")
print("=" * 70)

step_count = 0
render_steps = int(60 / CONTROL_FREQUENCY)

def get_camera_image():
    """TODO: Replace with actual camera"""
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

print("⚠ WARNING: Using dummy camera images")
print("   Implement get_camera_image() for real deployment\n")

try:
    while simulation_app.is_running() and step_count < MAX_STEPS:
        # Get image
        rgb_image = get_camera_image()
        
        # Run controller
        target_joints, target_gripper, info = controller.step(rgb_image)
        
        if info['status'] == 'warming_up':
            print(f"Step {step_count}: Warming up...")
        elif info['status'] == 'ik_failed':
            print(f"Step {step_count}: IK FAILED")
        elif info['status'] == 'success':
            franka.set_joint_positions(target_joints)
            
            if step_count % 10 == 0:
                stats = controller.get_stats()
                print(f"Step {step_count:3d} | "
                      f"Success: {stats['success_rate']*100:.1f}% | "
                      f"EE: [{stats['current_ee_pos'][0]:.3f}, "
                      f"{stats['current_ee_pos'][1]:.3f}, "
                      f"{stats['current_ee_pos'][2]:.3f}]")
        
        # Render
        for _ in range(render_steps):
            world.step(render=True)
        
        step_count += 1

except KeyboardInterrupt:
    print("\nStopped by user")

# Stats
print("\n" + "=" * 70)
print(" FINAL STATS")
print("=" * 70)
stats = controller.get_stats()
print(f"Total steps: {stats['total_steps']}")
print(f"IK failures: {stats['ik_failures']}")
print(f"Success rate: {stats['success_rate']*100:.1f}%")

print("\nKeeping window open. Close to exit.")
while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()