"""
Comprehensive Policy Visualization
Analyze trained UMI diffusion policy without simulation

Checks:
1. Action prediction accuracy on training data
2. Trajectory smoothness and consistency  
3. Error analysis per action dimension
4. Closed-loop stability
5. Multi-demo comparison

Usage:
    python visualize_trained_policy.py
"""

import os
import sys
import zarr
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

from scipy.spatial.transform import Rotation as R

# Add UMI to path
umi_path = os.path.expanduser("~/universal_manipulation_interface")
if umi_path not in sys.path:
    sys.path.insert(0, umi_path)

# After the matplotlib import (around line 28)
import sys
sys.path.insert(0, '/tmp/python_libs')
from imagecodecs.numcodecs import register_codecs
register_codecs()

from diffusion_policy.policy.diffusion_unet_timm_policy import DiffusionUnetTimmPolicy

import dill

print("=" * 70)
print(" UMI POLICY VISUALIZATION")
print("=" * 70)

# Configuration
CHECKPOINT_PATH = "/tmp/latest.ckpt"

DATA_PATH = os.path.expanduser(
    "~/universal_manipulation_interface/data/session_001/data"
)

NUM_DEMOS_TO_ANALYZE = 5
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load policy - UMI's METHOD
print(f"\nLoading policy from: {CHECKPOINT_PATH}")
print(f"Checkpoint exists: {os.path.exists(CHECKPOINT_PATH)}")

# Load with dill (UMI's way)
payload = torch.load(open(CHECKPOINT_PATH, 'rb'), pickle_module=dill)
print("✓ Checkpoint loaded")

# Extract policy from payload
if 'cfg' in payload:
    cfg = payload['cfg']
    print("Found config in checkpoint")

if 'state_dicts' in payload:
    state_dict = payload['state_dicts']['model']
elif 'state_dict' in payload:
    state_dict = payload['state_dict']
else:
    print("Checkpoint keys:", payload.keys())
    raise ValueError("Cannot find state_dict in checkpoint!")

# Load normalizer
normalizer_path = "/tmp/normalizer.pkl"

print(f"Loading normalizer from: {normalizer_path}")

import pickle
with open(normalizer_path, 'rb') as f:
    normalizer = pickle.load(f)

# Initialize policy from config
import hydra
policy = hydra.utils.instantiate(cfg.policy)

# Load state dict
policy.load_state_dict(state_dict)

# Set normalizer
policy.set_normalizer(normalizer)

print("✓ Policy loaded")
policy.eval()
policy = policy.to(DEVICE)
print("✓ Policy ready")

# Load data
print(f"\nLoading data from: {DATA_PATH}")
data = zarr.open(DATA_PATH, 'r')
print(f"✓ Data loaded")
# NEW
print(f"  Total demos: {len(data['camera0_rgb'])}")
print(f"  Demo length: {data['camera0_rgb'][0].shape[0]} timesteps")

def predict_on_demo(policy, demo_idx):
    """Run policy on one demonstration"""
    from scipy.spatial.transform import Rotation as R
    
    DEMO_LENGTH = 224
    start_idx = demo_idx * DEMO_LENGTH
    end_idx = start_idx + DEMO_LENGTH
    
    obs_rgb = data['camera0_rgb'][start_idx:end_idx]
    obs_ee_pos = data['robot0_eef_pos'][start_idx:end_idx]
    obs_ee_rot = data['robot0_eef_rot_axis_angle'][start_idx:end_idx]
    obs_gripper = data['robot0_gripper_width'][start_idx:end_idx]
    
    # Compute true actions as deltas
    pos_deltas = np.diff(obs_ee_pos, axis=0, prepend=obs_ee_pos[0:1])
    rot_deltas = np.diff(obs_ee_rot, axis=0, prepend=obs_ee_rot[0:1])
    
    true_actions = np.concatenate([
        pos_deltas,
        rot_deltas,
        obs_gripper
    ], axis=1)
    
    predicted_actions = []
    
    for t in range(2, len(obs_rgb)):
        # Convert axis-angle to rotation 6d
        rot_aa = obs_ee_rot[t-1:t+1]
        rot_6d_list = []
        for aa in rot_aa:
            rot_mat = R.from_rotvec(aa).as_matrix()
            rot_6d = rot_mat[:, :2].T.reshape(-1)
            rot_6d_list.append(rot_6d)
        rot_6d = np.stack(rot_6d_list)
        rot_wrt_start = rot_6d.copy()
        
        rgb_tensor = torch.tensor(obs_rgb[t-1:t+1]).unsqueeze(0).float()
        rgb_tensor = rgb_tensor.permute(0, 1, 4, 2, 3)
        
        obs = {
            'camera0_rgb': rgb_tensor.to(DEVICE),
            'robot0_eef_pos': torch.tensor(obs_ee_pos[t-1:t+1]).unsqueeze(0).float().to(DEVICE),
            'robot0_eef_rot_axis_angle': torch.tensor(rot_6d).unsqueeze(0).float().to(DEVICE),
            'robot0_gripper_width': torch.tensor(obs_gripper[t-1:t+1]).unsqueeze(0).float().to(DEVICE),
            'robot0_eef_rot_axis_angle_wrt_start': torch.tensor(rot_wrt_start).unsqueeze(0).float().to(DEVICE)
        }
        
        with torch.no_grad():
            action_dict = policy.predict_action(obs)
            pred_action = action_dict['action'][0, 0].cpu().numpy()  # (10,) [pos(3) + rot6d(6) + gripper(1)]
        
        # Convert rotation_6d back to axis-angle for comparison
        pred_pos = pred_action[:3]
        pred_rot_6d = pred_action[3:9]
        pred_gripper = pred_action[9:10]
        
        # Reconstruct rotation matrix from 6d
        rot_6d_reshaped = pred_rot_6d.reshape(2, 3).T  # (3, 2)
        col0 = rot_6d_reshaped[:, 0]
        col1 = rot_6d_reshaped[:, 1]
        col0 = col0 / np.linalg.norm(col0)
        col1 = col1 - np.dot(col0, col1) * col0
        col1 = col1 / np.linalg.norm(col1)
        col2 = np.cross(col0, col1)
        rot_mat = np.stack([col0, col1, col2], axis=1)
        
        # Convert to axis-angle
        pred_rot_aa = R.from_matrix(rot_mat).as_rotvec()
        
        # Reassemble as 7D action
        pred_action_7d = np.concatenate([pred_pos, pred_rot_aa, pred_gripper])
        predicted_actions.append(pred_action_7d)
    
    predicted_actions = np.array(predicted_actions)
    true_actions = true_actions[2:]
    obs_ee_pos = obs_ee_pos[2:]
    
    return {
        'predicted_actions': predicted_actions,
        'true_actions': true_actions,
        'ee_positions': obs_ee_pos
    }
    
# Analyze multiple demos
print(f"\nAnalyzing {NUM_DEMOS_TO_ANALYZE} demonstrations...")
results = []
for demo_idx in range(NUM_DEMOS_TO_ANALYZE):
    print(f"  Demo {demo_idx + 1}/{NUM_DEMOS_TO_ANALYZE}...", end=' ')
    result = predict_on_demo(policy, demo_idx)
    results.append(result)
    
    # Compute error
    error = np.linalg.norm(result['predicted_actions'] - result['true_actions'], axis=1)
    print(f"Mean error: {error.mean():.6f}")

print("\n✓ Analysis complete")

# Create comprehensive visualization
print("\nGenerating visualizations...")

fig = plt.figure(figsize=(20, 12))
gs = GridSpec(3, 4, figure=fig)

# Plot 1: 3D trajectory comparison (Demo 0)
ax1 = fig.add_subplot(gs[0, :2], projection='3d')
result = results[0]
pred_pos = np.cumsum(result['predicted_actions'][:, :3], axis=0) + result['ee_positions'][0]
true_pos = result['ee_positions']

ax1.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], 
         'b-', label='Policy Prediction', linewidth=2)
ax1.plot(true_pos[:, 0], true_pos[:, 1], true_pos[:, 2],
         'r--', label='Ground Truth', linewidth=2, alpha=0.7)
ax1.scatter(true_pos[0, 0], true_pos[0, 1], true_pos[0, 2],
           c='g', s=200, marker='o', label='Start')
ax1.scatter(true_pos[-1, 0], true_pos[-1, 1], true_pos[-1, 2],
           c='orange', s=200, marker='*', label='Goal')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_zlabel('Z (m)')
ax1.set_title('3D End-Effector Trajectory (Demo 1)', fontsize=14, fontweight='bold')
ax1.legend()

# Plot 2: Action error over time (Demo 0)
ax2 = fig.add_subplot(gs[0, 2:])
error = np.linalg.norm(result['predicted_actions'] - result['true_actions'], axis=1)
ax2.plot(error, 'g-', linewidth=2)
ax2.axhline(error.mean(), color='r', linestyle='--', label=f'Mean: {error.mean():.6f}')
ax2.fill_between(range(len(error)), 0, error, alpha=0.3, color='g')
ax2.set_xlabel('Timestep')
ax2.set_ylabel('L2 Error')
ax2.set_title('Action Prediction Error Over Time', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Per-dimension action comparison
ax3 = fig.add_subplot(gs[1, :2])
dim_names = ['Δx', 'Δy', 'Δz', 'Δrx', 'Δry', 'Δrz', 'gripper']
timesteps = range(len(result['predicted_actions']))

for i in range(3):  # Just position for clarity
    ax3.plot(timesteps, result['predicted_actions'][:, i], 
             label=f'Pred {dim_names[i]}', linestyle='-', linewidth=2)
    ax3.plot(timesteps, result['true_actions'][:, i],
             label=f'True {dim_names[i]}', linestyle='--', linewidth=1.5, alpha=0.7)

ax3.set_xlabel('Timestep')
ax3.set_ylabel('Action Value')
ax3.set_title('Position Action Components', fontsize=14, fontweight='bold')
ax3.legend(ncol=2)
ax3.grid(True, alpha=0.3)

# Plot 4: Rotation actions
ax4 = fig.add_subplot(gs[1, 2:])
for i in range(3, 6):
    ax4.plot(timesteps, result['predicted_actions'][:, i],
             label=f'Pred {dim_names[i]}', linestyle='-', linewidth=2)
    ax4.plot(timesteps, result['true_actions'][:, i],
             label=f'True {dim_names[i]}', linestyle='--', linewidth=1.5, alpha=0.7)

ax4.set_xlabel('Timestep')
ax4.set_ylabel('Action Value')
ax4.set_title('Rotation Action Components', fontsize=14, fontweight='bold')
ax4.legend(ncol=2)
ax4.grid(True, alpha=0.3)

# Plot 5: Multi-demo error distribution
ax5 = fig.add_subplot(gs[2, :2])
all_errors = []
for i, result in enumerate(results):
    error = np.linalg.norm(result['predicted_actions'] - result['true_actions'], axis=1)
    all_errors.append(error)
    ax5.plot(error, alpha=0.6, label=f'Demo {i+1}')

ax5.set_xlabel('Timestep')
ax5.set_ylabel('L2 Error')
ax5.set_title('Prediction Error Across Demonstrations', fontsize=14, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Error statistics
ax6 = fig.add_subplot(gs[2, 2:])
mean_errors = [np.mean(e) for e in all_errors]
max_errors = [np.max(e) for e in all_errors]
std_errors = [np.std(e) for e in all_errors]

x = np.arange(len(results))
width = 0.25

ax6.bar(x - width, mean_errors, width, label='Mean Error', color='blue', alpha=0.7)
ax6.bar(x, max_errors, width, label='Max Error', color='red', alpha=0.7)
ax6.bar(x + width, std_errors, width, label='Std Error', color='green', alpha=0.7)

ax6.set_xlabel('Demo')
ax6.set_ylabel('Error')
ax6.set_title('Error Statistics Per Demo', fontsize=14, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels([f'Demo {i+1}' for i in range(len(results))])
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('policy_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: policy_comprehensive_analysis.png")

# Print summary statistics
print("\n" + "=" * 70)
print(" SUMMARY STATISTICS")
print("=" * 70)

overall_mean_error = np.mean([np.mean(e) for e in all_errors])
overall_max_error = np.max([np.max(e) for e in all_errors])

print(f"\nOverall Performance:")
print(f"  Mean prediction error: {overall_mean_error:.6f}")
print(f"  Max prediction error: {overall_max_error:.6f}")

print(f"\nPer-demo statistics:")
for i, (result, errors) in enumerate(zip(results, all_errors)):
    print(f"  Demo {i+1}:")
    print(f"    Mean error: {np.mean(errors):.6f}")
    print(f"    Max error: {np.max(errors):.6f}")
    print(f"    Std error: {np.std(errors):.6f}")

print("\n" + "=" * 70)
print(" INTERPRETATION")
print("=" * 70)

if overall_mean_error < 0.01:
    print("\n✅ EXCELLENT: Policy predictions are very close to demonstrations")
    print("   → Policy learned the task well")
    print("   → Issues in Isaac Sim are likely simulation-related, not policy")
elif overall_mean_error < 0.05:
    print("\n✓ GOOD: Policy predictions are reasonable")
    print("   → Policy captured main patterns")
    print("   → Some simulation issues expected, but policy is functional")
else:
    print("\n⚠ WARNING: High prediction error")
    print("   → Policy may not have learned the task properly")
    print("   → Consider retraining with:")
    print("      - More demonstrations")
    print("      - Different hyperparameters")
    print("      - Longer training")

print("\n" + "=" * 70)
print(" NEXT STEPS")
print("=" * 70)

if overall_mean_error < 0.05:
    print("\n1. ✅ Policy is trained well → Focus on Isaac Sim deployment")
    print("2. Debug simulation issues:")
    print("   - IK solver accuracy")
    print("   - Action scaling/normalization")
    print("   - Camera calibration")
    print("   - Gripper geometry (if in camera view)")
else:
    print("\n1. ⚠ Policy needs improvement → Retrain before deployment")
    print("2. Check training data quality")
    print("3. Verify normalization statistics")
    print("4. Try different network architectures")

print("\n✓ Analysis complete!")
print(f"Results saved to: policy_comprehensive_analysis.png")