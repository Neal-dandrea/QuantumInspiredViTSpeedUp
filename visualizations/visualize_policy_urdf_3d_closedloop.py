"""
3D Side-by-Side Comparison with CLOSED-LOOP Policy Predictions
Simulates real deployment where policy gets new observations each step

Usage:
    python visualize_policy_urdf_3d_closedloop.py
"""

import numpy as np
import sys
import os
import xml.etree.ElementTree as ET

# Add library paths
sys.path.insert(0, '/tmp/python_libs')
from imagecodecs.numcodecs import register_codecs
register_codecs()

import pyvista as pv
from stl import mesh as stl_mesh
import zarr
from scipy.spatial.transform import Rotation as R

print("=" * 70)
print(" 3D POLICY COMPARISON - CLOSED-LOOP SIMULATION")
print("=" * 70)

# ============================================================================
# CONFIGURATION
# ============================================================================

URDF_PATH = '/tmp/gripper_cad/umi_gripper_fixed.urdf'
DATA_PATH = '/home/wadeab/universal_manipulation_interface/data/session_001/data'
CHECKPOINT_PATH = "/tmp/latest.ckpt"
NORMALIZER_PATH = "/tmp/normalizer.pkl"
DEMO_IDX = 0
NUM_POSES = 8
OUTPUT_FILE = '/home/wadeab/universal_manipulation_interface/policy_comparison_closedloop_3d.png'

COLOR_GT_PATH = 'red'
COLOR_PRED_PATH = 'blue'
COLOR_GT_GRIPPER = [0.7, 0.2, 0.2, 0.5]
COLOR_PRED_GRIPPER = [0.2, 0.4, 0.7, 0.5]

# ============================================================================
# URDF PARSING (same as before)
# ============================================================================

def parse_urdf_transform(origin_elem):
    """Parse xyz and rpy from URDF origin element"""
    if origin_elem is None:
        return np.eye(4)
    
    xyz = origin_elem.get('xyz', '0 0 0').split()
    rpy = origin_elem.get('rpy', '0 0 0').split()
    
    xyz = np.array([float(x) for x in xyz])
    rpy = np.array([float(x) for x in rpy])
    
    rot = R.from_euler('xyz', rpy).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = xyz
    
    return transform


def load_urdf_gripper(urdf_path):
    """Load gripper from URDF with proper assembly"""
    print(f"\nLoading URDF: {urdf_path}")
    
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    link_meshes = {}
    
    for link in root.findall('.//link'):
        link_name = link.get('name')
        visual = link.find('.//visual')
        
        if visual is not None:
            mesh_elem = visual.find('.//mesh')
            if mesh_elem is not None:
                mesh_file = mesh_elem.get('filename')
                
                if os.path.exists(mesh_file):
                    try:
                        stl = stl_mesh.Mesh.from_file(mesh_file)
                        vertices = stl.vectors.reshape(-1, 3)
                        unique_verts, indices = np.unique(vertices, axis=0, return_inverse=True)
                        faces = indices.reshape(-1, 3)
                        
                        faces_formatted = []
                        for face in faces:
                            faces_formatted.extend([3, face[0], face[1], face[2]])
                        
                        mesh = pv.PolyData(unique_verts, faces_formatted)
                        origin = visual.find('.//origin')
                        local_transform = parse_urdf_transform(origin)
                        
                        link_meshes[link_name] = {
                            'mesh': mesh,
                            'local_transform': local_transform
                        }
                        
                        print(f"  ✓ Loaded {link_name}: {mesh.n_points} vertices")
                        
                    except Exception as e:
                        print(f"  ⚠ Failed {link_name}: {e}")
    
    joints = {}
    for joint in root.findall('.//joint'):
        joint_name = joint.get('name')
        parent = joint.find('parent').get('link')
        child = joint.find('child').get('link')
        origin = joint.find('origin')
        
        transform = parse_urdf_transform(origin)
        
        joints[joint_name] = {
            'parent': parent,
            'child': child,
            'transform': transform
        }
    
    def get_global_transform(link_name, transforms_cache={}):
        if link_name in transforms_cache:
            return transforms_cache[link_name]
        
        if link_name == 'world' or link_name == 'base_link':
            transforms_cache[link_name] = np.eye(4)
            return np.eye(4)
        
        for joint_name, joint_data in joints.items():
            if joint_data['child'] == link_name:
                parent_transform = get_global_transform(joint_data['parent'], transforms_cache)
                global_transform = parent_transform @ joint_data['transform']
                transforms_cache[link_name] = global_transform
                return global_transform
        
        return np.eye(4)
    
    assembled_gripper = None
    
    for link_name, link_data in link_meshes.items():
        mesh = link_data['mesh'].copy()
        
        local_T = link_data['local_transform']
        points_homog = np.hstack([mesh.points, np.ones((mesh.n_points, 1))])
        points_transformed = (local_T @ points_homog.T).T[:, :3]
        mesh.points = points_transformed
        
        global_T = get_global_transform(link_name)
        points_homog = np.hstack([mesh.points, np.ones((mesh.n_points, 1))])
        points_transformed = (global_T @ points_homog.T).T[:, :3]
        mesh.points = points_transformed
        
        if assembled_gripper is None:
            assembled_gripper = mesh
        else:
            assembled_gripper = assembled_gripper.merge(mesh)
    
    print(f"\n✓ Assembled gripper: {assembled_gripper.n_points} total vertices")
    
    center = assembled_gripper.center
    assembled_gripper.points -= center
    
    return assembled_gripper


# ============================================================================
# POLICY LOADING
# ============================================================================

def load_trained_policy():
    """Load the trained UMI diffusion policy"""
    print("\nLoading trained policy...")
    
    import torch
    import dill
    import hydra
    
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    payload = torch.load(open(CHECKPOINT_PATH, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    
    if 'state_dicts' in payload:
        state_dict = payload['state_dicts']['model']
    elif 'state_dict' in payload:
        state_dict = payload['state_dict']
    else:
        raise ValueError("Cannot find state_dict in checkpoint!")
    
    import pickle
    with open(NORMALIZER_PATH, 'rb') as f:
        normalizer = pickle.load(f)
    
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(state_dict)
    policy.set_normalizer(normalizer)
    policy.eval()
    policy = policy.to(DEVICE)
    
    print(f"✓ Policy loaded on {DEVICE}")
    
    return policy, DEVICE


# ============================================================================
# CLOSED-LOOP PREDICTION
# ============================================================================

def run_closedloop_predictions(policy, device, gt_positions, gt_rotations):
    """
    Run policy in CLOSED-LOOP mode:
    - Use actual camera images from demo
    - Use PREDICTED positions/rotations as observations (not ground truth)
    - Policy corrects based on what it predicts it sees
    """
    import torch
    from scipy.spatial.transform import Rotation as Rot
    
    print("\nRunning CLOSED-LOOP policy predictions...")
    print("  (Policy uses its own predicted state, not ground truth)")
    
    data = zarr.open(DATA_PATH, 'r')
    DEMO_LENGTH = 224
    start_idx = DEMO_IDX * DEMO_LENGTH
    end_idx = start_idx + DEMO_LENGTH
    
    obs_rgb = data['camera0_rgb'][start_idx:end_idx]
    obs_gripper = data['robot0_gripper_width'][start_idx:end_idx]
    
    # Initialize with ground truth starting position
    pred_positions = [gt_positions[0], gt_positions[1]]
    pred_rotations = [gt_rotations[0], gt_rotations[1]]
    
    for t in range(2, len(obs_rgb)):
        # Use PREDICTED positions/rotations as observations (closed-loop!)
        current_pos = np.array([pred_positions[-2], pred_positions[-1]])
        current_rot = np.array([pred_rotations[-2], pred_rotations[-1]])
        
        # Convert axis-angle to rotation 6d
        rot_6d_list = []
        for aa in current_rot:
            rot_mat = Rot.from_rotvec(aa).as_matrix()
            rot_6d = rot_mat[:, :2].T.reshape(-1)
            rot_6d_list.append(rot_6d)
        rot_6d = np.stack(rot_6d_list)
        rot_wrt_start = rot_6d.copy()
        
        # Prepare observation with PREDICTED state
        rgb_tensor = torch.tensor(obs_rgb[t-1:t+1]).unsqueeze(0).float()
        rgb_tensor = rgb_tensor.permute(0, 1, 4, 2, 3)
        
        obs = {
            'camera0_rgb': rgb_tensor.to(device),
            'robot0_eef_pos': torch.tensor(current_pos).unsqueeze(0).float().to(device),
            'robot0_eef_rot_axis_angle': torch.tensor(rot_6d).unsqueeze(0).float().to(device),
            'robot0_gripper_width': torch.tensor(obs_gripper[t-1:t+1]).unsqueeze(0).float().to(device),
            'robot0_eef_rot_axis_angle_wrt_start': torch.tensor(rot_wrt_start).unsqueeze(0).float().to(device)
        }
        
        with torch.no_grad():
            action_dict = policy.predict_action(obs)
            pred_action = action_dict['action'][0, 0].cpu().numpy()
        
        # Convert rotation_6d back to axis-angle
        pred_pos_delta = pred_action[:3]
        pred_rot_6d = pred_action[3:9]
        
        # Reconstruct rotation matrix from 6d
        rot_6d_reshaped = pred_rot_6d.reshape(2, 3).T
        col0 = rot_6d_reshaped[:, 0]
        col1 = rot_6d_reshaped[:, 1]
        col0 = col0 / np.linalg.norm(col0)
        col1 = col1 - np.dot(col0, col1) * col0
        col1 = col1 / np.linalg.norm(col1)
        col2 = np.cross(col0, col1)
        rot_mat = np.stack([col0, col1, col2], axis=1)
        
        pred_rot_delta = Rot.from_matrix(rot_mat).as_rotvec()
        
        # Apply action to get new predicted state
        new_pos = pred_positions[-1] + pred_pos_delta
        
        current_rot_mat = Rot.from_rotvec(pred_rotations[-1])
        delta_rot_mat = Rot.from_rotvec(pred_rot_delta)
        new_rot = (current_rot_mat * delta_rot_mat).as_rotvec()
        
        pred_positions.append(new_pos)
        pred_rotations.append(new_rot)
        
        if t % 50 == 0:
            print(f"  Progress: {t-2}/{len(obs_rgb)-2} timesteps")
    
    pred_positions = np.array(pred_positions)
    pred_rotations = np.array(pred_rotations)
    
    print(f"✓ Generated {len(pred_positions)} predicted poses (CLOSED-LOOP)")
    
    return pred_positions, pred_rotations


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def rotation_matrix_from_axis_angle(axis_angle):
    """Convert axis-angle to rotation matrix"""
    return R.from_rotvec(axis_angle).as_matrix()


def transform_gripper(gripper, position, rotation_matrix):
    """Apply rotation and translation to gripper"""
    gripper_copy = gripper.copy()
    points_rotated = (rotation_matrix @ gripper_copy.points.T).T
    gripper_copy.points = points_rotated + position
    return gripper_copy


def load_demo_data(demo_idx):
    """Load ground truth trajectory"""
    print(f"\nLoading demo {demo_idx}...")
    data = zarr.open(DATA_PATH, 'r')
    
    DEMO_LENGTH = 224
    start_idx = demo_idx * DEMO_LENGTH
    end_idx = start_idx + DEMO_LENGTH
    
    positions = np.array(data['robot0_eef_pos'][start_idx:end_idx])
    rotations = np.array(data['robot0_eef_rot_axis_angle'][start_idx:end_idx])
    
    print(f"✓ Loaded {len(positions)} timesteps")
    return positions, rotations


# ============================================================================
# MAIN VISUALIZATION
# ============================================================================

def create_comparison_visualization():
    """Create side-by-side comparison with CLOSED-LOOP predictions"""
    
    gripper = load_urdf_gripper(URDF_PATH)
    
    gt_positions, gt_rotations = load_demo_data(DEMO_IDX)
    
    policy, device = load_trained_policy()
    pred_positions, pred_rotations = run_closedloop_predictions(policy, device, gt_positions, gt_rotations)
    
    # Stats
    trajectory_error = np.linalg.norm(pred_positions - gt_positions, axis=1)
    mean_error = np.mean(trajectory_error)
    max_error = np.max(trajectory_error)
    
    print(f"\nTrajectory Statistics (CLOSED-LOOP):")
    print(f"  Mean position error: {mean_error*1000:.1f} mm")
    print(f"  Max position error:  {max_error*1000:.1f} mm")
    
    # Create visualization
    print("\nCreating visualization...")
    plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=[2400, 1200])
    
    # ========== LEFT: GROUND TRUTH ==========
    plotter.subplot(0, 0)
    plotter.add_text("Ground Truth Trajectory", position='upper_edge', 
                     font_size=24, color='black')
    
    gt_polyline = pv.PolyData(gt_positions)
    gt_polyline.lines = np.hstack([[len(gt_positions)] + list(range(len(gt_positions)))])
    plotter.add_mesh(gt_polyline, color=COLOR_GT_PATH, line_width=8)
    
    plotter.add_mesh(pv.Sphere(radius=0.01, center=gt_positions[0]), color='green')
    plotter.add_mesh(pv.Sphere(radius=0.01, center=gt_positions[-1]), color='orange')
    
    pose_indices = np.linspace(0, len(gt_positions)-1, NUM_POSES, dtype=int)
    
    for i, idx in enumerate(pose_indices):
        pos = gt_positions[idx]
        rot = rotation_matrix_from_axis_angle(gt_rotations[idx])
        gripper_transformed = transform_gripper(gripper, pos, rot)
        
        alpha = 0.3 + 0.3 * (i / (NUM_POSES - 1))
        plotter.add_mesh(gripper_transformed, color=COLOR_GT_GRIPPER[:3], 
                        opacity=alpha, show_edges=True, edge_color='darkred', line_width=0.5)
    
    plotter.add_axes()
    plotter.camera_position = 'iso'
    
    # ========== RIGHT: CLOSED-LOOP PREDICTION ==========
    plotter.subplot(0, 1)
    plotter.add_text("Policy (CLOSED-LOOP)", position='upper_edge', font_size=24, color='black')
    
    pred_polyline = pv.PolyData(pred_positions)
    pred_polyline.lines = np.hstack([[len(pred_positions)] + list(range(len(pred_positions)))])
    plotter.add_mesh(pred_polyline, color=COLOR_PRED_PATH, line_width=8)
    
    plotter.add_mesh(gt_polyline, color='gray', line_width=2, opacity=0.2)
    
    plotter.add_mesh(pv.Sphere(radius=0.01, center=pred_positions[0]), color='green')
    plotter.add_mesh(pv.Sphere(radius=0.01, center=pred_positions[-1]), color='orange')
    
    for i, idx in enumerate(pose_indices):
        pos = pred_positions[idx]
        rot = rotation_matrix_from_axis_angle(pred_rotations[idx])
        gripper_transformed = transform_gripper(gripper, pos, rot)
        
        alpha = 0.3 + 0.3 * (i / (NUM_POSES - 1))
        plotter.add_mesh(gripper_transformed, color=COLOR_PRED_GRIPPER[:3],
                        opacity=alpha, show_edges=True, edge_color='darkblue', line_width=0.5)
    
    plotter.add_axes()
    plotter.camera_position = 'iso'
    plotter.link_views()
    
    # Save
    print(f"\nSaving to {OUTPUT_FILE}...")
    plotter.screenshot(OUTPUT_FILE)
    
    print("✓ Complete!")
    print("\n" + "=" * 70)
    print(f" OUTPUT: {OUTPUT_FILE}")
    print("=" * 70)
    print(f"  Demo: {DEMO_IDX}")
    print(f"  Timesteps: {len(pred_positions)}")
    print(f"  Mean position error: {mean_error*1000:.1f} mm")
    print(f"  Max position error: {max_error*1000:.1f} mm")
    print(f"  Gripper poses shown: {NUM_POSES} per trajectory")
    print("\n✅ CLOSED-LOOP: Policy uses its own predictions as observations!")
    print("   This simulates real deployment behavior.")
    
    return OUTPUT_FILE


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    try:
        output = create_comparison_visualization()
        print(f"\n🎉 SUCCESS! CLOSED-LOOP visualization complete.")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()