"""
Comprehensive Policy Visualization - ALL DEMONSTRATIONS
Analyze trained UMI diffusion policy without simulation

Analyzes ALL 381 demonstrations to get comprehensive policy assessment
Expected runtime: 2-3 hours

Usage:
    python visualize_trained_policy_ALL_DEMOS.py
"""

import os
import sys
import zarr
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import time

from scipy.spatial.transform import Rotation as R

# Add UMI to path
umi_path = os.path.expanduser("~/universal_manipulation_interface")
if umi_path not in sys.path:
    sys.path.insert(0, umi_path)

# Register imagecodecs for JPEG-XL support
import sys
sys.path.insert(0, '/tmp/python_libs')
from imagecodecs.numcodecs import register_codecs
register_codecs()

from diffusion_policy.policy.diffusion_unet_timm_policy import DiffusionUnetTimmPolicy

import dill

print("=" * 70)
print(" UMI POLICY VISUALIZATION - ALL DEMONSTRATIONS")
print("=" * 70)

# Configuration
CHECKPOINT_PATH = "/tmp/latest.ckpt"

DATA_PATH = os.path.expanduser(
    "~/universal_manipulation_interface/data/session_001/data"
)

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

# Calculate number of complete demos
DEMO_LENGTH = 224
TOTAL_TIMESTEPS = len(data['robot0_eef_pos'])
NUM_DEMOS_TO_ANALYZE = TOTAL_TIMESTEPS // DEMO_LENGTH

print(f"  Total timesteps: {TOTAL_TIMESTEPS}")
print(f"  Demo length: {DEMO_LENGTH} timesteps")
print(f"  Complete demos: {NUM_DEMOS_TO_ANALYZE}")
print(f"\n⏱ Estimated runtime: 2-3 hours")
print(f"Progress will be printed every 10 demos\n")


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


# Analyze ALL demos
print(f"Analyzing ALL {NUM_DEMOS_TO_ANALYZE} demonstrations...")
print("=" * 70)

start_time = time.time()
results = []
all_errors = []
mean_errors_list = []

for demo_idx in range(NUM_DEMOS_TO_ANALYZE):
    # Progress update every 10 demos
    if demo_idx % 10 == 0:
        elapsed = time.time() - start_time
        demos_per_sec = demo_idx / elapsed if elapsed > 0 else 0
        remaining = (NUM_DEMOS_TO_ANALYZE - demo_idx) / demos_per_sec if demos_per_sec > 0 else 0
        
        print(f"Progress: {demo_idx}/{NUM_DEMOS_TO_ANALYZE} ({100*demo_idx/NUM_DEMOS_TO_ANALYZE:.1f}%) | "
              f"Elapsed: {elapsed/60:.1f}min | "
              f"ETA: {remaining/60:.1f}min")
    
    result = predict_on_demo(policy, demo_idx)
    results.append(result)
    
    # Compute error
    error = np.linalg.norm(result['predicted_actions'] - result['true_actions'], axis=1)
    all_errors.append(error)
    mean_errors_list.append(np.mean(error))

print(f"\n✓ Analysis complete! Total time: {(time.time() - start_time)/60:.1f} minutes")

# Print comprehensive statistics
print("\n" + "=" * 70)
print(" COMPREHENSIVE POLICY ANALYSIS (ALL DEMOS)")
print("=" * 70)

overall_mean_error = np.mean(mean_errors_list)
overall_median_error = np.median(mean_errors_list)
overall_std_error = np.std(mean_errors_list)
overall_max_error = np.max([np.max(e) for e in all_errors])
overall_min_error = np.min(mean_errors_list)

print(f"\n📊 Global Statistics:")
print(f"  Mean error across all demos:   {overall_mean_error:.6f}")
print(f"  Median error:                  {overall_median_error:.6f}")
print(f"  Std deviation:                 {overall_std_error:.6f}")
print(f"  Min mean error (best demo):    {overall_min_error:.6f}")
print(f"  Max error (single timestep):   {overall_max_error:.6f}")

# Error distribution
print(f"\n📈 Error Distribution ({NUM_DEMOS_TO_ANALYZE} demos):")
excellent = sum(1 for e in mean_errors_list if e < 0.01)
good = sum(1 for e in mean_errors_list if 0.01 <= e < 0.02)
acceptable = sum(1 for e in mean_errors_list if 0.02 <= e < 0.05)
poor = sum(1 for e in mean_errors_list if e >= 0.05)

print(f"  Excellent  (< 0.01):  {excellent:4d} demos ({100*excellent/NUM_DEMOS_TO_ANALYZE:5.1f}%)")
print(f"  Good    (0.01-0.02):  {good:4d} demos ({100*good/NUM_DEMOS_TO_ANALYZE:5.1f}%)")
print(f"  Accept  (0.02-0.05):  {acceptable:4d} demos ({100*acceptable/NUM_DEMOS_TO_ANALYZE:5.1f}%)")
print(f"  Poor       (> 0.05):  {poor:4d} demos ({100*poor/NUM_DEMOS_TO_ANALYZE:5.1f}%)")

# Identify worst demos
worst_demo_indices = np.argsort(mean_errors_list)[-10:][::-1]
print(f"\n⚠️  Top 10 Worst Performing Demos:")
for rank, idx in enumerate(worst_demo_indices, 1):
    print(f"  {rank:2d}. Demo {idx:3d}: Mean error = {mean_errors_list[idx]:.6f}, "
          f"Max error = {np.max(all_errors[idx]):.6f}")

# Identify best demos
best_demo_indices = np.argsort(mean_errors_list)[:10]
print(f"\n✅ Top 10 Best Performing Demos:")
for rank, idx in enumerate(best_demo_indices, 1):
    print(f"  {rank:2d}. Demo {idx:3d}: Mean error = {mean_errors_list[idx]:.6f}")

# Percentiles
print(f"\n📊 Error Percentiles:")
p25 = np.percentile(mean_errors_list, 25)
p50 = np.percentile(mean_errors_list, 50)
p75 = np.percentile(mean_errors_list, 75)
p95 = np.percentile(mean_errors_list, 95)
p99 = np.percentile(mean_errors_list, 99)

print(f"  25th percentile: {p25:.6f}")
print(f"  50th percentile: {p50:.6f}")
print(f"  75th percentile: {p75:.6f}")
print(f"  95th percentile: {p95:.6f}")
print(f"  99th percentile: {p99:.6f}")

print("\n" + "=" * 70)
print(" INTERPRETATION")
print("=" * 70)

if overall_mean_error < 0.01:
    print("\n✅ EXCELLENT: Policy predictions are very close to demonstrations")
    print(f"   → {100*excellent/NUM_DEMOS_TO_ANALYZE:.1f}% of demos have error < 0.01")
    print("   → Policy learned the task very well")
    print("   → Issues in Isaac Sim are likely simulation-related, not policy")
elif overall_mean_error < 0.02:
    print("\n✓ GOOD: Policy predictions are reasonable")
    print(f"   → {100*(excellent+good)/NUM_DEMOS_TO_ANALYZE:.1f}% of demos have error < 0.02")
    print("   → Policy captured main patterns")
    print("   → Some simulation issues expected, but policy is functional")
elif overall_mean_error < 0.05:
    print("\n⚠ ACCEPTABLE: Policy is functional but could be better")
    print(f"   → {100*poor/NUM_DEMOS_TO_ANALYZE:.1f}% of demos have error > 0.05")
    print("   → Policy works but may struggle with precision")
    print("   → Consider retraining or check data quality")
else:
    print("\n❌ POOR: Policy predictions are not accurate enough")
    print(f"   → {100*poor/NUM_DEMOS_TO_ANALYZE:.1f}% of demos have error > 0.05")
    print("   → Policy did not learn the task properly")
    print("   → Retrain with more data or different hyperparameters")

print("\n" + "=" * 70)
print(" NEXT STEPS")
print("=" * 70)

if overall_mean_error < 0.02:
    print("\n1. ✅ Policy is trained well → Focus on Isaac Sim deployment")
    print("2. Debug simulation issues:")
    print("   - IK solver accuracy")
    print("   - Action scaling/normalization")
    print("   - Camera calibration")
    print("   - Gripper geometry (if in camera view)")
    print(f"\n3. Investigate the {poor} poor-performing demos (error > 0.05)")
    print("   - Check if they have unusual motions or data issues")
else:
    print("\n1. ⚠ Policy needs improvement → Retrain before deployment")
    print("2. Check training data quality")
    print("3. Verify normalization statistics")
    print("4. Try different network architectures or more training epochs")

# Create simplified visualization (plotting all 381 would be messy)
print("\n" + "=" * 70)
print(" GENERATING SUMMARY VISUALIZATIONS")
print("=" * 70)

fig = plt.figure(figsize=(20, 12))
gs = GridSpec(3, 3, figure=fig)

# Plot 1: 3D trajectory for best demo
ax1 = fig.add_subplot(gs[0, 0], projection='3d')
best_idx = best_demo_indices[0]
result = results[best_idx]
pred_pos = np.cumsum(result['predicted_actions'][:, :3], axis=0) + result['ee_positions'][0]
true_pos = result['ee_positions']

ax1.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], 
         'b-', label='Policy', linewidth=2)
ax1.plot(true_pos[:, 0], true_pos[:, 1], true_pos[:, 2],
         'r--', label='Ground Truth', linewidth=2, alpha=0.7)
ax1.scatter(true_pos[0, 0], true_pos[0, 1], true_pos[0, 2],
           c='g', s=100, marker='o')
ax1.scatter(true_pos[-1, 0], true_pos[-1, 1], true_pos[-1, 2],
           c='orange', s=100, marker='*')
ax1.set_title(f'Best Demo (#{best_idx}, error={mean_errors_list[best_idx]:.6f})', fontweight='bold')
ax1.legend()

# Plot 2: 3D trajectory for worst demo
ax2 = fig.add_subplot(gs[0, 1], projection='3d')
worst_idx = worst_demo_indices[0]
result = results[worst_idx]
pred_pos = np.cumsum(result['predicted_actions'][:, :3], axis=0) + result['ee_positions'][0]
true_pos = result['ee_positions']

ax2.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], 
         'b-', label='Policy', linewidth=2)
ax2.plot(true_pos[:, 0], true_pos[:, 1], true_pos[:, 2],
         'r--', label='Ground Truth', linewidth=2, alpha=0.7)
ax2.scatter(true_pos[0, 0], true_pos[0, 1], true_pos[0, 2],
           c='g', s=100, marker='o')
ax2.scatter(true_pos[-1, 0], true_pos[-1, 1], true_pos[-1, 2],
           c='orange', s=100, marker='*')
ax2.set_title(f'Worst Demo (#{worst_idx}, error={mean_errors_list[worst_idx]:.6f})', fontweight='bold')
ax2.legend()

# Plot 3: 3D trajectory for median demo
ax3 = fig.add_subplot(gs[0, 2], projection='3d')
median_idx = np.argsort(mean_errors_list)[NUM_DEMOS_TO_ANALYZE//2]
result = results[median_idx]
pred_pos = np.cumsum(result['predicted_actions'][:, :3], axis=0) + result['ee_positions'][0]
true_pos = result['ee_positions']

ax3.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], 
         'b-', label='Policy', linewidth=2)
ax3.plot(true_pos[:, 0], true_pos[:, 1], true_pos[:, 2],
         'r--', label='Ground Truth', linewidth=2, alpha=0.7)
ax3.scatter(true_pos[0, 0], true_pos[0, 1], true_pos[0, 2],
           c='g', s=100, marker='o')
ax3.scatter(true_pos[-1, 0], true_pos[-1, 1], true_pos[-1, 2],
           c='orange', s=100, marker='*')
ax3.set_title(f'Median Demo (#{median_idx}, error={mean_errors_list[median_idx]:.6f})', fontweight='bold')
ax3.legend()

# Plot 4: Error distribution histogram
ax4 = fig.add_subplot(gs[1, :2])
ax4.hist(mean_errors_list, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
ax4.axvline(overall_mean_error, color='r', linestyle='--', linewidth=2, label=f'Mean: {overall_mean_error:.6f}')
ax4.axvline(overall_median_error, color='g', linestyle='--', linewidth=2, label=f'Median: {overall_median_error:.6f}')
ax4.set_xlabel('Mean Error Per Demo')
ax4.set_ylabel('Frequency')
ax4.set_title(f'Error Distribution Across All {NUM_DEMOS_TO_ANALYZE} Demos', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Cumulative distribution
ax5 = fig.add_subplot(gs[1, 2])
sorted_errors = np.sort(mean_errors_list)
cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
ax5.plot(sorted_errors, cumulative, linewidth=2, color='navy')
ax5.axhline(95, color='r', linestyle='--', alpha=0.5, label='95th percentile')
ax5.axvline(p95, color='r', linestyle='--', alpha=0.5)
ax5.set_xlabel('Mean Error')
ax5.set_ylabel('Cumulative % of Demos')
ax5.set_title('Cumulative Error Distribution', fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend()

# Plot 6: Error vs Demo index (to see if there are patterns)
ax6 = fig.add_subplot(gs[2, :])
ax6.scatter(range(NUM_DEMOS_TO_ANALYZE), mean_errors_list, alpha=0.5, s=10, color='blue')
ax6.axhline(overall_mean_error, color='r', linestyle='--', label=f'Mean: {overall_mean_error:.6f}')
ax6.axhline(0.01, color='g', linestyle=':', alpha=0.5, label='Excellent threshold (0.01)')
ax6.axhline(0.05, color='orange', linestyle=':', alpha=0.5, label='Acceptable threshold (0.05)')
ax6.set_xlabel('Demo Index')
ax6.set_ylabel('Mean Error')
ax6.set_title(f'Prediction Error Across All {NUM_DEMOS_TO_ANALYZE} Demonstrations', fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('policy_ALL_DEMOS_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: policy_ALL_DEMOS_analysis.png")

print("\n" + "=" * 70)
print(" ✅ COMPREHENSIVE ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nResults saved to: policy_ALL_DEMOS_analysis.png")
print(f"Total demos analyzed: {NUM_DEMOS_TO_ANALYZE}")
print(f"Overall mean error: {overall_mean_error:.6f}")
print(f"\nYour policy is {'EXCELLENT' if overall_mean_error < 0.01 else 'GOOD' if overall_mean_error < 0.02 else 'ACCEPTABLE' if overall_mean_error < 0.05 else 'POOR'}!")