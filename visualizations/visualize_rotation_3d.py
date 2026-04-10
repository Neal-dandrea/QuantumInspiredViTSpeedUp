import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import zarr
import torch
import pickle
import dill
import hydra

data = zarr.open("data/session_001/data", 'r')
start_idx = 0
end_idx = 224

gt_positions = np.array(data['robot0_eef_pos'][start_idx:end_idx])
gt_rotations = np.array(data['robot0_eef_rot_axis_angle'][start_idx:end_idx])
gt_images = data['camera0_rgb'][start_idx:end_idx]
gt_gripper = np.array(data['robot0_gripper_width'][start_idx:end_idx])

print("Loading policy...")
payload = torch.load('/tmp/latest.ckpt', pickle_module=dill)
cfg = payload['cfg']
state_dict = payload.get('state_dicts', {}).get('model') or payload.get('state_dict')
with open('/tmp/normalizer.pkl', 'rb') as f:
    normalizer = pickle.load(f)

policy = hydra.utils.instantiate(cfg.policy)
policy.load_state_dict(state_dict)
policy.set_normalizer(normalizer)
policy.eval()
policy = policy.cuda()

def preprocess_image(rgb):
    return rgb.transpose(2, 0, 1).astype(np.float32) / 255.0

def matrix_to_rotation_6d(rot_matrix):
    return rot_matrix[:, :2].T.reshape(-1)

predicted_rotations = []
OBS_HORIZON = 2

print("Computing predictions...")
for t in range(len(gt_positions)):
    obs_start = max(0, t - OBS_HORIZON + 1)
    obs_imgs = [preprocess_image(np.array(gt_images[i])) for i in range(obs_start, t+1)]
    obs_pos = [gt_positions[i] for i in range(obs_start, t+1)]
    obs_rot = [gt_rotations[i] for i in range(obs_start, t+1)]
    obs_gripper = [gt_gripper[i] for i in range(obs_start, t+1)]
    
    while len(obs_imgs) < OBS_HORIZON:
        obs_imgs.insert(0, obs_imgs[0])
        obs_pos.insert(0, obs_pos[0])
        obs_rot.insert(0, obs_rot[0])
        obs_gripper.insert(0, obs_gripper[0])
    
    obs_rot_6d = []
    for rot_aa in obs_rot:
        rot_mat = R.from_rotvec(rot_aa).as_matrix()
        rot_6d = matrix_to_rotation_6d(rot_mat)
        obs_rot_6d.append(rot_6d)
    
    start_rot_mat = R.from_rotvec(gt_rotations[0]).as_matrix()
    obs_rot_wrt_start = []
    for rot_aa in obs_rot:
        current_rot_mat = R.from_rotvec(rot_aa).as_matrix()
        rel_rot_mat = current_rot_mat @ start_rot_mat.T
        rot_6d_rel = matrix_to_rotation_6d(rel_rot_mat)
        obs_rot_wrt_start.append(rot_6d_rel)
    
    obs_dict = {
        'camera0_rgb': torch.from_numpy(np.stack(obs_imgs)).float().unsqueeze(0).cuda(),
        'robot0_eef_pos': torch.from_numpy(np.stack(obs_pos)).float().unsqueeze(0).cuda(),
        'robot0_eef_rot_axis_angle': torch.from_numpy(np.stack(obs_rot_6d)).float().unsqueeze(0).cuda(),
        'robot0_eef_rot_axis_angle_wrt_start': torch.from_numpy(np.stack(obs_rot_wrt_start)).float().unsqueeze(0).cuda(),
        'robot0_gripper_width': torch.from_numpy(np.stack([[g] for g in obs_gripper])).float().unsqueeze(0).cuda(),
    }
    
    with torch.no_grad():
        action = policy.predict_action(obs_dict)['action'].squeeze(0).cpu().numpy()
    
    pred_rot_6d = action[0, 3:9]
    pred_rot_6d_reshaped = pred_rot_6d.reshape(2, 3).T
    col0 = pred_rot_6d_reshaped[:, 0]
    col1 = pred_rot_6d_reshaped[:, 1]
    col0 = col0 / np.linalg.norm(col0)
    col1 = col1 - np.dot(col0, col1) * col0
    col1 = col1 / np.linalg.norm(col1)
    col2 = np.cross(col0, col1)
    pred_rot_matrix = np.stack([col0, col1, col2], axis=1)
    pred_rot_aa = R.from_matrix(pred_rot_matrix).as_rotvec()
    predicted_rotations.append(pred_rot_aa)

predicted_rotations = np.array(predicted_rotations)

rotation_errors = []
for gt_rot, pred_rot in zip(gt_rotations, predicted_rotations):
    gt_R = R.from_rotvec(gt_rot)
    pred_R = R.from_rotvec(pred_rot)
    error_R = pred_R.inv() * gt_R
    angle_error = error_R.magnitude()
    rotation_errors.append(np.degrees(angle_error))

rotation_errors = np.array(rotation_errors)
print(f"Mean rotation error: {rotation_errors.mean():.2f} degrees")

fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title('Ground Truth Orientations')

for idx in range(0, len(gt_positions), 8):
    pos = gt_positions[idx]
    rot_mat = R.from_rotvec(gt_rotations[idx]).as_matrix()
    for i, color in enumerate(['r', 'g', 'b']):
        direction = rot_mat[:, i]
        ax1.quiver(pos[0], pos[1], pos[2], direction[0], direction[1], direction[2],
                   length=0.05, color=color, alpha=0.6)

ax1.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 'k-', alpha=0.3)

ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title('Policy Predicted Orientations')

for idx in range(0, len(gt_positions), 8):
    pos = gt_positions[idx]
    rot_mat = R.from_rotvec(predicted_rotations[idx]).as_matrix()
    for i, color in enumerate(['r', 'g', 'b']):
        direction = rot_mat[:, i]
        ax2.quiver(pos[0], pos[1], pos[2], direction[0], direction[1], direction[2],
                   length=0.05, color=color, alpha=0.6)

ax2.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 'k-', alpha=0.3)

plt.savefig('rotation_visualization_3d.png', dpi=150)
print("Saved: rotation_visualization_3d.png")
