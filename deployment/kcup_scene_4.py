"""
UMI K-Cup Scene v4 - Fixed IK + Local Dependencies
===================================================

Run with:
    ~/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh ~/Downloads/kcup_scene_v4.py
"""

import numpy as np
import os
import sys
import time

# Add local dependencies FIRST (before any imports that might need them)
sys.path.insert(0, '/tmp/umi_deps')

# ===========================================================================
# START ISAAC SIM FIRST
# ===========================================================================
print("="*60)
print("Starting Isaac Sim...")
print("="*60)

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

# Now import Isaac Sim modules
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCylinder
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.sensors.camera import Camera
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
from pxr import UsdGeom, Gf, UsdPhysics
from scipy.spatial.transform import Rotation as R
import torch

# ===========================================================================
# CONFIGURATION
# ===========================================================================
# Scene dimensions - TABLE MOVED CLOSER to robot
TABLE_LENGTH = 0.66
TABLE_WIDTH = 1.00  # Narrower table so robot can reach across
TABLE_THICKNESS = 0.03

# Robot pedestal - robot is elevated
PEDESTAL_HEIGHT = 0.50

# Table position - closer to robot!
TABLE_CENTER_X = 0.45  # Much closer (was 0.58)
TABLE_CENTER_Y = 0.0
TABLE_HEIGHT = PEDESTAL_HEIGHT + 0.42  # Table surface slightly below robot base
TABLE_SURFACE_Z = TABLE_HEIGHT

# Objects
PLATE_DIAMETER = 0.10
PLATE_DEPTH = 0.02
KCUP_RADIUS = 0.022
KCUP_HEIGHT = 0.044

# Control settings
CONTROL_FREQ = 10
ACTION_HORIZON = 8
OBS_HORIZON = 2

# ===========================================================================
# CREATE WORLD
# ===========================================================================
print("Creating world...")
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

from isaacsim.core.utils.stage import get_current_stage
stage = get_current_stage()

# ===========================================================================
# ROBOT PEDESTAL
# ===========================================================================
print("Adding robot pedestal...")
pedestal_cube = UsdGeom.Cube.Define(stage, "/World/Pedestal")
pedestal_cube.CreateSizeAttr(1.0)
ped_xform = UsdGeom.Xformable(pedestal_cube.GetPrim())
ped_xform.ClearXformOpOrder()
ped_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, PEDESTAL_HEIGHT / 2))
ped_xform.AddScaleOp().Set(Gf.Vec3d(0.30, 0.30, PEDESTAL_HEIGHT))
pedestal_cube.CreateDisplayColorAttr([(0.3, 0.3, 0.35)])
UsdPhysics.CollisionAPI.Apply(pedestal_cube.GetPrim())

# ===========================================================================
# ROBOT
# ===========================================================================
print("Adding Franka on pedestal...")
franka = world.scene.add(
    Franka(
        prim_path="/World/Franka",
        name="franka",
        position=np.array([0.0, 0.0, PEDESTAL_HEIGHT]),
    )
)

# ===========================================================================
# TABLE
# ===========================================================================
print("Adding table...")
TABLE_CENTER_Z = TABLE_HEIGHT - TABLE_THICKNESS / 2

table_cube = UsdGeom.Cube.Define(stage, "/World/Table/Top")
table_cube.CreateSizeAttr(1.0)
xform = UsdGeom.Xformable(table_cube.GetPrim())
xform.ClearXformOpOrder()
xform.AddTranslateOp().Set(Gf.Vec3d(TABLE_CENTER_X, TABLE_CENTER_Y, TABLE_CENTER_Z))
xform.AddScaleOp().Set(Gf.Vec3d(TABLE_LENGTH, TABLE_WIDTH, TABLE_THICKNESS))
table_cube.CreateDisplayColorAttr([(0.6, 0.4, 0.2)])
UsdPhysics.CollisionAPI.Apply(table_cube.GetPrim())

# Table legs
LEG_SIZE = [0.04, 0.04, TABLE_HEIGHT - TABLE_THICKNESS]
leg_positions = [
    [TABLE_CENTER_X - TABLE_LENGTH/2 + 0.04, -TABLE_WIDTH/2 + 0.04, LEG_SIZE[2]/2],
    [TABLE_CENTER_X + TABLE_LENGTH/2 - 0.04, -TABLE_WIDTH/2 + 0.04, LEG_SIZE[2]/2],
    [TABLE_CENTER_X - TABLE_LENGTH/2 + 0.04,  TABLE_WIDTH/2 - 0.04, LEG_SIZE[2]/2],
    [TABLE_CENTER_X + TABLE_LENGTH/2 - 0.04,  TABLE_WIDTH/2 - 0.04, LEG_SIZE[2]/2],
]
for i, pos in enumerate(leg_positions):
    leg = UsdGeom.Cube.Define(stage, f"/World/Table/Leg{i}")
    leg.CreateSizeAttr(1.0)
    leg_xform = UsdGeom.Xformable(leg.GetPrim())
    leg_xform.ClearXformOpOrder()
    leg_xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
    leg_xform.AddScaleOp().Set(Gf.Vec3d(*LEG_SIZE))
    leg.CreateDisplayColorAttr([(0.4, 0.25, 0.1)])
    UsdPhysics.CollisionAPI.Apply(leg.GetPrim())

# ===========================================================================
# RED PLATE (target)
# ===========================================================================
print("Adding red plate...")
PLATE_X = TABLE_CENTER_X + 0.15
PLATE_Y = -0.20
PLATE_Z = TABLE_SURFACE_Z + PLATE_DEPTH / 2

plate_cube = UsdGeom.Cube.Define(stage, "/World/Plate")
plate_cube.CreateSizeAttr(1.0)
plate_xform = UsdGeom.Xformable(plate_cube.GetPrim())
plate_xform.ClearXformOpOrder()
plate_xform.AddTranslateOp().Set(Gf.Vec3d(PLATE_X, PLATE_Y, PLATE_Z))
plate_xform.AddScaleOp().Set(Gf.Vec3d(PLATE_DIAMETER, PLATE_DIAMETER, PLATE_DEPTH))
plate_cube.CreateDisplayColorAttr([(0.8, 0.1, 0.1)])
UsdPhysics.CollisionAPI.Apply(plate_cube.GetPrim())

# ===========================================================================
# K-CUP
# ===========================================================================
print("Adding K-cup...")

def get_random_kcup_position():
    while True:
        x = TABLE_CENTER_X + np.random.uniform(-0.15, 0.15)
        y = np.random.uniform(-0.15, 0.15)
        dist = np.sqrt((x - PLATE_X)**2 + (y - PLATE_Y)**2)
        if dist > PLATE_DIAMETER/2 + 0.05:
            break
    z = TABLE_SURFACE_Z + KCUP_HEIGHT / 2 + 0.001
    return np.array([x, y, z])

kcup_pos = get_random_kcup_position()
kcup = world.scene.add(
    DynamicCylinder(
        prim_path="/World/KCup",
        name="kcup",
        position=kcup_pos,
        radius=float(KCUP_RADIUS),
        height=float(KCUP_HEIGHT),
        color=np.array([0.2, 0.6, 0.3]),
        mass=0.012,
    )
)
print(f"K-cup at: [{kcup_pos[0]:.3f}, {kcup_pos[1]:.3f}]")

# ===========================================================================
# CAMERA
# ===========================================================================
print("Adding wrist camera...")
camera = Camera(
    prim_path="/World/Franka/panda_hand/wrist_camera",
    resolution=(224, 224),
    frequency=10,
)

# ===========================================================================
# RESET WORLD
# ===========================================================================
print("Resetting world...")
world.reset()

# Position camera
camera_prim = stage.GetPrimAtPath("/World/Franka/panda_hand/wrist_camera")
if camera_prim:
    xform = UsdGeom.Xformable(camera_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0.06, 0, 0.04))
    xform.AddRotateXYZOp().Set(Gf.Vec3d(-100, 0, 0))
camera.initialize()

# ===========================================================================
# IK SOLVER
# ===========================================================================
print("Setting up IK...")
ISAAC_PATH = os.path.expanduser("~/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64")
ik_solver = LulaKinematicsSolver(
    robot_description_path=f"{ISAAC_PATH}/exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/franka/rmpflow/robot_descriptor.yaml",
    urdf_path=f"{ISAAC_PATH}/exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/franka/lula_franka_gen.urdf"
)

# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================
HOME = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04])

def smooth_move(target, steps=60):
    current = franka.get_joint_positions()
    for i in range(steps):
        alpha = (i + 1) / steps
        interp = current + alpha * (target - current)
        franka.get_articulation_controller().apply_action(ArticulationAction(joint_positions=interp))
        world.step(render=True)

def move_to_xyz(x, y, z):
    _, current_ori = franka.end_effector.get_world_pose()
    current_joints = franka.get_joint_positions()[:7]
    result, success = ik_solver.compute_inverse_kinematics(
        frame_name="panda_hand",
        warm_start=current_joints,
        target_position=np.array([x, y, z]),
        target_orientation=current_ori,
    )
    if success:
        target = franka.get_joint_positions().copy()
        target[:7] = result
        smooth_move(target, steps=40)
        return True
    print(f"IK failed for [{x:.3f}, {y:.3f}, {z:.3f}]")
    return False

def set_gripper(width):
    current = franka.get_joint_positions()
    target = current.copy()
    target[7] = width
    target[8] = width
    smooth_move(target, steps=20)

# ===========================================================================
# GO HOME
# ===========================================================================
print("Moving to home...")
smooth_move(HOME)

# ===========================================================================
# PRINT INFO
# ===========================================================================
ee_pos, _ = franka.end_effector.get_world_pose()

print("\n" + "="*60)
print("SCENE READY")
print("="*60)
print(f"Robot base at Z = {PEDESTAL_HEIGHT:.2f}m (on pedestal)")
print(f"Table surface at Z = {TABLE_SURFACE_Z:.2f}m")
print(f"EE position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
print(f"K-cup at: [{kcup_pos[0]:.3f}, {kcup_pos[1]:.3f}]")
print(f"Plate at: [{PLATE_X:.3f}, {PLATE_Y:.3f}]")
print("\nManual controls:")
print("  w/s/a/d/q/e - move XYZ")
print("  o/c - open/close gripper")
print("  1 - above kcup, 2 - above plate")
print("  r - randomize kcup, h - home")
print("  p - load policy, g - run policy, stop - stop policy")
print("  i - print info, x - exit")
print("="*60 + "\n")

# ===========================================================================
# POLICY (delayed loading)
# ===========================================================================
policy = None
policy_loaded = False

def load_policy():
    global policy, policy_loaded
    
    if policy_loaded:
        print("Policy already loaded!")
        return True
    
    print("\n" + "="*60)
    print("Loading policy... (this may take a minute)")
    print("="*60)
    
    try:
        os.environ['HF_HOME'] = '/tmp/hf_cache'
        os.environ['TORCH_HOME'] = '/tmp/torch_cache'
        os.environ['XDG_CACHE_HOME'] = '/tmp/cache'
        
        UMI_ROOT = os.path.expanduser("~/universal_manipulation_interface")
        if UMI_ROOT not in sys.path:
            sys.path.insert(0, UMI_ROOT)
        
        from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import TrainDiffusionUnetHybridWorkspace
        
        CHECKPOINT_PATH = "/tmp/model.ckpt"
        if not os.path.exists(CHECKPOINT_PATH):
            print(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
            return False
        
        ckpt = torch.load(CHECKPOINT_PATH, map_location='cuda', weights_only=False)
        workspace = TrainDiffusionUnetHybridWorkspace(ckpt['cfg'])
        workspace.load_payload(ckpt, exclude_keys=None, include_keys=None)
        
        policy = workspace.model
        policy.eval()
        policy.cuda()
        
        policy_loaded = True
        print("Policy loaded successfully!")
        print("="*60 + "\n")
        return True
        
    except Exception as e:
        print(f"ERROR loading policy: {e}")
        import traceback
        traceback.print_exc()
        return False

# ===========================================================================
# POLICY HELPERS
# ===========================================================================
def rotation_6d_to_matrix(rot_6d):
    a1, a2 = rot_6d[:3], rot_6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)

def matrix_to_rotation_6d(matrix):
    return matrix[:, :2].flatten()

def quat_to_rot6d(quat_wxyz):
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    rot_matrix = R.from_quat(quat_xyzw).as_matrix()
    return matrix_to_rotation_6d(rot_matrix)

def preprocess_image(image):
    img = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    return np.transpose(img, (2, 0, 1))

def get_robot_obs():
    ee_pos, ee_quat = franka.end_effector.get_world_pose()
    rot_6d = quat_to_rot6d(ee_quat)
    joints = franka.get_joint_positions()
    gripper_width = joints[7] + joints[8]
    return ee_pos, ee_quat, rot_6d, gripper_width

def apply_policy_action(action, current_pos, current_quat):
    delta_pos = action[0:3]
    delta_rot_6d = action[3:9]
    target_gripper = np.clip(action[9], 0.0, 0.08)
    
    target_pos = current_pos + delta_pos
    
    quat_xyzw = np.array([current_quat[1], current_quat[2], current_quat[3], current_quat[0]])
    current_rot = R.from_quat(quat_xyzw)
    delta_rot_matrix = rotation_6d_to_matrix(delta_rot_6d)
    delta_rot = R.from_matrix(delta_rot_matrix)
    target_rot = delta_rot * current_rot
    target_quat_xyzw = target_rot.as_quat()
    target_quat = np.array([target_quat_xyzw[3], target_quat_xyzw[0], target_quat_xyzw[1], target_quat_xyzw[2]])
    
    current_joints = franka.get_joint_positions()[:7]
    result, success = ik_solver.compute_inverse_kinematics(
        frame_name="panda_hand",
        warm_start=current_joints,
        target_position=target_pos,
        target_orientation=target_quat,
    )
    
    if success:
        target_joints = franka.get_joint_positions().copy()
        target_joints[:7] = result
        target_joints[7] = target_gripper / 2
        target_joints[8] = target_gripper / 2
        return target_joints, True
    return None, False

# ===========================================================================
# MAIN LOOP
# ===========================================================================
import threading

cmd = None
running = True
policy_running = False
obs_buffer = []
action_buffer = None
action_idx = 0
step_count = 0
last_control_time = time.time()
start_rot_matrix = None

STEP = 0.03
HOVER_Z = TABLE_SURFACE_Z + 0.15

def input_loop():
    global cmd, running
    while running:
        try:
            c = input().strip().lower()
            if c:
                cmd = c
        except:
            running = False

threading.Thread(target=input_loop, daemon=True).start()

while simulation_app.is_running() and running:
    if cmd:
        c = cmd
        cmd = None
        
        ee_pos, ee_quat, _, gripper_width = get_robot_obs()
        
        if c == 'x':
            running = False
            
        elif c == 'w':
            move_to_xyz(ee_pos[0] + STEP, ee_pos[1], ee_pos[2])
        elif c == 's':
            move_to_xyz(ee_pos[0] - STEP, ee_pos[1], ee_pos[2])
        elif c == 'a':
            move_to_xyz(ee_pos[0], ee_pos[1] + STEP, ee_pos[2])
        elif c == 'd':
            move_to_xyz(ee_pos[0], ee_pos[1] - STEP, ee_pos[2])
        elif c == 'q':
            move_to_xyz(ee_pos[0], ee_pos[1], ee_pos[2] + STEP)
        elif c == 'e':
            move_to_xyz(ee_pos[0], ee_pos[1], ee_pos[2] - STEP)
            
        elif c == 'o':
            set_gripper(0.04)
            print("Gripper open")
        elif c == 'c':
            set_gripper(0.0)
            print("Gripper closed")
            
        elif c == '1':
            kp, _ = kcup.get_world_pose()
            print(f"Moving above K-cup at [{kp[0]:.3f}, {kp[1]:.3f}]")
            move_to_xyz(kp[0], kp[1], HOVER_Z)
        elif c == '2':
            print(f"Moving above plate at [{PLATE_X:.3f}, {PLATE_Y:.3f}]")
            move_to_xyz(PLATE_X, PLATE_Y, HOVER_Z)
        elif c == 'h':
            smooth_move(HOME)
            print("Home")
        elif c == 'r':
            policy_running = False
            new_pos = get_random_kcup_position()
            kcup.set_world_pose(position=new_pos)
            print(f"K-cup at: [{new_pos[0]:.3f}, {new_pos[1]:.3f}]")
            
        elif c == 'p':
            load_policy()
        elif c == 'g':
            if not policy_loaded:
                print("Load policy first with 'p'!")
            else:
                policy_running = True
                obs_buffer = []
                action_buffer = None
                action_idx = 0
                step_count = 0
                _, eq, _, _ = get_robot_obs()
                qxyzw = np.array([eq[1], eq[2], eq[3], eq[0]])
                start_rot_matrix = R.from_quat(qxyzw).as_matrix()
                print(">>> POLICY RUNNING - Type 'stop' to stop <<<")
        elif c == 'stop':
            policy_running = False
            print(">>> POLICY STOPPED <<<")
            
        elif c == 'i':
            print(f"EE: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}], gripper: {gripper_width:.3f}")
    
    # Policy control
    if policy_running and policy is not None:
        current_time = time.time()
        
        if current_time - last_control_time >= 1.0 / CONTROL_FREQ:
            last_control_time = current_time
            step_count += 1
            
            rgba = camera.get_rgba()
            if rgba is None:
                world.step(render=True)
                continue
            rgb = rgba[:, :, :3]
            
            ee_pos, ee_quat, rot_6d, gripper_width = get_robot_obs()
            
            if start_rot_matrix is not None:
                qxyzw = np.array([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]])
                current_rot_matrix = R.from_quat(qxyzw).as_matrix()
                rel_rot_matrix = current_rot_matrix @ start_rot_matrix.T
                rot_6d_wrt_start = matrix_to_rotation_6d(rel_rot_matrix)
            else:
                rot_6d_wrt_start = rot_6d
            
            img = preprocess_image(rgb)
            
            obs_buffer.append({
                'img': img,
                'pos': ee_pos.copy(),
                'rot': rot_6d.copy(),
                'rot_wrt_start': rot_6d_wrt_start.copy(),
                'gripper': gripper_width,
                'quat': ee_quat.copy(),
            })
            
            if len(obs_buffer) > OBS_HORIZON:
                obs_buffer = obs_buffer[-OBS_HORIZON:]
            while len(obs_buffer) < OBS_HORIZON:
                obs_buffer.insert(0, obs_buffer[0])
            
            if action_buffer is None or action_idx >= ACTION_HORIZON:
                obs_dict = {
                    'camera0_rgb': torch.from_numpy(
                        np.stack([o['img'] for o in obs_buffer])
                    ).float().unsqueeze(0).cuda(),
                    'robot0_eef_pos': torch.from_numpy(
                        np.stack([o['pos'] for o in obs_buffer])
                    ).float().unsqueeze(0).cuda(),
                    'robot0_eef_rot_axis_angle': torch.from_numpy(
                        np.stack([o['rot'] for o in obs_buffer])
                    ).float().unsqueeze(0).cuda(),
                    'robot0_eef_rot_axis_angle_wrt_start': torch.from_numpy(
                        np.stack([o['rot_wrt_start'] for o in obs_buffer])
                    ).float().unsqueeze(0).cuda(),
                    'robot0_gripper_width': torch.from_numpy(
                        np.stack([[o['gripper']] for o in obs_buffer])
                    ).float().unsqueeze(0).cuda(),
                }
                
                with torch.no_grad():
                    action_out = policy.predict_action(obs_dict)
                    action_buffer = action_out['action'].squeeze(0).cpu().numpy()
                    action_idx = 0
                
                print(f"Step {step_count}: Replanned")
            
            if action_buffer is not None and action_idx < len(action_buffer):
                action = action_buffer[action_idx]
                action_idx += 1
                
                obs = obs_buffer[-1]
                target_joints, success = apply_policy_action(action, obs['pos'], obs['quat'])
                
                if success:
                    franka.get_articulation_controller().apply_action(
                        ArticulationAction(joint_positions=target_joints)
                    )
                
                if step_count % 10 == 0:
                    print(f"  pos=[{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}], grip={gripper_width:.3f}")
    
    world.step(render=True)

print("Closing...")
simulation_app.close()