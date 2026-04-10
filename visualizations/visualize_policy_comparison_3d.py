"""
3D Side-by-Side Comparison: Ground Truth vs Policy Prediction
Shows UMI gripper at multiple poses along both trajectories

Usage:
    python visualize_policy_comparison_3d.py
"""

import numpy as np
import sys
import os

# Add library paths
sys.path.insert(0, '/tmp/python_libs')
from imagecodecs.numcodecs import register_codecs
register_codecs()

import pyvista as pv
from stl import mesh as stl_mesh
import zarr

print("=" * 70)
print(" 3D POLICY COMPARISON WITH UMI GRIPPER")
print("=" * 70)

# ============================================================================
# CONFIGURATION
# ============================================================================

GRIPPER_STL_DIR = '/tmp/gripper_cad'
DATA_PATH = '/home/wadeab/universal_manipulation_interface/data/session_001/data'
DEMO_IDX = 0  # Which demo to visualize
NUM_POSES = 8  # Number of gripper poses to show along each trajectory
OUTPUT_FILE = '/home/wadeab/universal_manipulation_interface/policy_comparison_3d.png'

# Colors
COLOR_GT_PATH = 'red'
COLOR_PRED_PATH = 'blue'
COLOR_GT_GRIPPER = [0.8, 0.2, 0.2, 0.4]  # RGBA - semi-transparent red
COLOR_PRED_GRIPPER = [0.2, 0.4, 0.8, 0.4]  # RGBA - semi-transparent blue

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_gripper_stl():
    """Load and combine all gripper STL parts"""
    print("\nLoading gripper STL files...")
    
    stl_files = [
        'base_of_gripper.stl',
        'main_connector.stl',
        'gripper_mount.stl',
        'finger_holder_left.stl',
        'finger_holder_right.stl',
        'gripper_finger_slider_left.stl',
        'gripper_finger_slider_right.stl',
        'soft_gripper_finger.stl',
    ]
    
    all_meshes = []
    
    for stl_file in stl_files:
        filepath = os.path.join(GRIPPER_STL_DIR, stl_file)
        if os.path.exists(filepath):
            try:
                mesh = stl_mesh.Mesh.from_file(filepath)
                
                # Convert to PyVista format
                vertices = mesh.vectors.reshape(-1, 3)
                
                # Remove duplicate vertices
                unique_vertices, indices = np.unique(vertices, axis=0, return_inverse=True)
                
                # Create faces
                faces = indices.reshape(-1, 3)
                faces_formatted = []
                for face in faces:
                    faces_formatted.extend([3, face[0], face[1], face[2]])
                
                pv_mesh = pv.PolyData(unique_vertices, faces_formatted)
                all_meshes.append(pv_mesh)
                
                print(f"  ✓ Loaded {stl_file}")
            except Exception as e:
                print(f"  ⚠ Failed to load {stl_file}: {e}")
        else:
            print(f"  ⚠ Not found: {stl_file}")
    
    # Merge all meshes
    if all_meshes:
        gripper = all_meshes[0]
        for mesh in all_meshes[1:]:
            gripper = gripper.merge(mesh)
        
        print(f"\n✓ Combined gripper: {gripper.n_points} vertices, {gripper.n_cells} faces")
        
        # Center gripper
        center = gripper.center
        gripper.points -= center
        
        # Scale to meters (if needed - adjust based on your CAD units)
        # Uncomment if gripper is in mm:
        # gripper.points *= 0.001
        
        return gripper
    else:
        raise ValueError("No gripper meshes loaded!")


def rotation_matrix_from_axis_angle(axis_angle):
    """Convert axis-angle to rotation matrix"""
    from scipy.spatial.transform import Rotation as R
    return R.from_rotvec(axis_angle).as_matrix()


def transform_gripper(gripper, position, rotation_matrix):
    """Apply rotation and translation to gripper"""
    gripper_copy = gripper.copy()
    
    # Rotate
    points_rotated = (rotation_matrix @ gripper_copy.points.T).T
    
    # Translate
    gripper_copy.points = points_rotated + position
    
    return gripper_copy


def load_demo_data(demo_idx):
    """Load ground truth trajectory from demonstration"""
    print(f"\nLoading demo {demo_idx}...")
    
    data = zarr.open(DATA_PATH, 'r')
    
    DEMO_LENGTH = 224
    start_idx = demo_idx * DEMO_LENGTH
    end_idx = start_idx + DEMO_LENGTH
    
    positions = np.array(data['robot0_eef_pos'][start_idx:end_idx])
    rotations = np.array(data['robot0_eef_rot_axis_angle'][start_idx:end_idx])
    
    print(f"✓ Loaded {len(positions)} timesteps")
    
    return positions, rotations


def compute_predicted_trajectory(positions):
    """
    Simulate policy predictions by adding small noise to ground truth
    
    In real usage, replace this with actual policy rollout results
    """
    print("\nGenerating predicted trajectory...")
    print("  (Using simulated predictions - replace with actual policy output)")
    
    # Add small cumulative drift to simulate prediction errors
    noise = np.random.randn(*positions.shape) * 0.003  # 3mm std noise
    cumulative_noise = np.cumsum(noise, axis=0)
    
    predicted_positions = positions + cumulative_noise
    
    return predicted_positions


# ============================================================================
# MAIN VISUALIZATION
# ============================================================================

def create_comparison_visualization():
    """Create side-by-side comparison of GT vs predicted trajectories"""
    
    # Load gripper
    gripper = load_gripper_stl()
    
    # Load demo data
    gt_positions, gt_rotations = load_demo_data(DEMO_IDX)
    
    # Get predicted trajectory (replace with actual policy predictions!)
    pred_positions = compute_predicted_trajectory(gt_positions)
    pred_rotations = gt_rotations.copy()  # Assuming same rotations for now
    
    # Calculate trajectory statistics
    trajectory_error = np.linalg.norm(pred_positions - gt_positions, axis=1)
    mean_error = np.mean(trajectory_error)
    max_error = np.max(trajectory_error)
    
    print(f"\nTrajectory Statistics:")
    print(f"  Mean position error: {mean_error*1000:.1f} mm")
    print(f"  Max position error:  {max_error*1000:.1f} mm")
    
    # Create plotter with two viewports
    print("\nCreating visualization...")
    plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=[2400, 1200])
    
    # ========== LEFT PANEL: GROUND TRUTH ==========
    plotter.subplot(0, 0)
    plotter.add_text("Ground Truth Trajectory", position='upper_edge', 
                     font_size=24, color='black', font='arial')
    
    # Plot GT trajectory
    gt_polyline = pv.PolyData(gt_positions)
    gt_polyline.lines = np.hstack([[len(gt_positions)] + list(range(len(gt_positions)))])
    plotter.add_mesh(gt_polyline, color=COLOR_GT_PATH, line_width=8, 
                     label='Ground Truth Path')
    
    # Start/end markers
    plotter.add_mesh(pv.Sphere(radius=0.01, center=gt_positions[0]), 
                     color='green', label='Start')
    plotter.add_mesh(pv.Sphere(radius=0.01, center=gt_positions[-1]), 
                     color='orange', label='Goal')
    
    # Add grippers at key poses
    pose_indices = np.linspace(0, len(gt_positions)-1, NUM_POSES, dtype=int)
    
    for i, idx in enumerate(pose_indices):
        pos = gt_positions[idx]
        rot = rotation_matrix_from_axis_angle(gt_rotations[idx])
        
        gripper_transformed = transform_gripper(gripper, pos, rot)
        
        # Gradient color
        alpha = 0.3 + 0.4 * (i / (NUM_POSES - 1))
        color = COLOR_GT_GRIPPER[:3] + [alpha]
        
        plotter.add_mesh(gripper_transformed, color=color, opacity=alpha,
                        show_edges=True, edge_color='darkred', line_width=0.3)
    
    plotter.add_axes()
    plotter.camera_position = 'iso'
    plotter.add_legend(size=(0.15, 0.15), loc='lower right')
    
    # ========== RIGHT PANEL: POLICY PREDICTION ==========
    plotter.subplot(0, 1)
    plotter.add_text("Policy Prediction", position='upper_edge',
                     font_size=24, color='black', font='arial')
    
    # Plot predicted trajectory
    pred_polyline = pv.PolyData(pred_positions)
    pred_polyline.lines = np.hstack([[len(pred_positions)] + list(range(len(pred_positions)))])
    plotter.add_mesh(pred_polyline, color=COLOR_PRED_PATH, line_width=8,
                     label='Predicted Path')
    
    # Show ground truth as faint reference
    plotter.add_mesh(gt_polyline, color='gray', line_width=2, opacity=0.2,
                     label='Ground Truth (ref)')
    
    # Start/end markers
    plotter.add_mesh(pv.Sphere(radius=0.01, center=pred_positions[0]),
                     color='green', label='Start')
    plotter.add_mesh(pv.Sphere(radius=0.01, center=pred_positions[-1]),
                     color='orange', label='Predicted Goal')
    
    # Add grippers at key poses
    for i, idx in enumerate(pose_indices):
        pos = pred_positions[idx]
        rot = rotation_matrix_from_axis_angle(pred_rotations[idx])
        
        gripper_transformed = transform_gripper(gripper, pos, rot)
        
        # Gradient color
        alpha = 0.3 + 0.4 * (i / (NUM_POSES - 1))
        color = COLOR_PRED_GRIPPER[:3] + [alpha]
        
        plotter.add_mesh(gripper_transformed, color=color, opacity=alpha,
                        show_edges=True, edge_color='darkblue', line_width=0.3)
    
    plotter.add_axes()
    plotter.camera_position = 'iso'
    plotter.add_legend(size=(0.15, 0.15), loc='lower right')
    
    # Link cameras so both views rotate together
    plotter.link_views()
    
    # Save
    print(f"\nSaving to {OUTPUT_FILE}...")
    plotter.screenshot(OUTPUT_FILE)
    
    print("✓ Visualization complete!")
    print("\n" + "=" * 70)
    print(f" OUTPUT: {OUTPUT_FILE}")
    print("=" * 70)
    print(f"\nStatistics:")
    print(f"  Demo: {DEMO_IDX}")
    print(f"  Timesteps: {len(gt_positions)}")
    print(f"  Mean error: {mean_error*1000:.1f} mm")
    print(f"  Max error: {max_error*1000:.1f} mm")
    print(f"  Gripper poses shown: {NUM_POSES} per trajectory")
    
    return OUTPUT_FILE


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    try:
        output = create_comparison_visualization()
        print(f"\n✅ SUCCESS! Open {output} to view the comparison.")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()