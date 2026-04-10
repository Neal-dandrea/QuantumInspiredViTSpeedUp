import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import zarr

data = zarr.open("data/session_001/data", 'r')
start_idx = 0
end_idx = 224

gt_positions = np.array(data['robot0_eef_pos'][start_idx:end_idx])
gt_rotations = np.array(data['robot0_eef_rot_axis_angle'][start_idx:end_idx])

print(f"Loaded {len(gt_positions)} positions and rotations")

# Compute rotation errors over trajectory
rotation_magnitudes = []
for rot in gt_rotations:
    angle = np.linalg.norm(rot)  # Magnitude of axis-angle
    rotation_magnitudes.append(np.degrees(angle))

rotation_magnitudes = np.array(rotation_magnitudes)

print(f"Rotation range: {rotation_magnitudes.min():.2f}° to {rotation_magnitudes.max():.2f}°")

# Visualization
fig = plt.figure(figsize=(16, 8))

# Plot 1: Orientations along trajectory
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title('Gripper Orientations Along Trajectory', fontsize=14, fontweight='bold')

sample_indices = range(0, len(gt_positions), 8)
for idx in sample_indices:
    pos = gt_positions[idx]
    rot_mat = R.from_rotvec(gt_rotations[idx]).as_matrix()
    
    # Draw RGB = XYZ axes
    arrow_length = 0.05
    colors = ['r', 'g', 'b']
    for i, color in enumerate(colors):
        direction = rot_mat[:, i]
        ax1.quiver(pos[0], pos[1], pos[2],
                   direction[0], direction[1], direction[2],
                   length=arrow_length, color=color, alpha=0.7, 
                   arrow_length_ratio=0.3, linewidth=2)

ax1.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 
         'k-', alpha=0.4, linewidth=2, label='Trajectory')
ax1.set_xlabel('X (m)', fontsize=11)
ax1.set_ylabel('Y (m)', fontsize=11)
ax1.set_zlabel('Z (m)', fontsize=11)
ax1.legend()

# Plot 2: Rotation magnitude over time
ax2 = fig.add_subplot(122)
ax2.plot(rotation_magnitudes, 'b-', linewidth=2)
ax2.set_xlabel('Timestep', fontsize=12)
ax2.set_ylabel('Rotation Magnitude (degrees)', fontsize=12)
ax2.set_title('Gripper Rotation Over Time', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axhline(rotation_magnitudes.mean(), color='r', linestyle='--', 
            label=f'Mean: {rotation_magnitudes.mean():.2f}°', linewidth=2)
ax2.legend()

plt.tight_layout()
plt.savefig('rotation_ground_truth_3d.png', dpi=150, bbox_inches='tight')
print("✓ Saved: rotation_ground_truth_3d.png")
print("(Shows ground truth orientations - no policy needed)")
