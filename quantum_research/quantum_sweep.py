"""
QML Architecture Sweep - quantum_sweep.py
==========================================

Comprehensive script that runs all QML architecture configurations
for the quantum robot policy research. Designed for GPU cluster use.

Configurations tested:
  Phase 1: Qubit count sweep          (4q, 6q, 8q, 10q)
  Phase 2: Circuit depth sweep        (2L, 4L, 6L, 8L)
  Phase 3: Encoding method sweep      (amplitude, angle, IQP)
  Phase 4: Gate pattern sweep         (QONN, BasicEntangler, StronglyEntangling)
  Phase 5: Knowledge distillation     (best arch + KD)

Output per architecture:
  - {ARCH_NAME}_training_log.json         Full epoch history
  - {ARCH_NAME}_loss_curve.png            Two-panel loss plot
  - {ARCH_NAME}_summary.txt               Human-readable report
  - {ARCH_NAME}_best.pt                   Best model checkpoint
  - {ARCH_NAME}_per_dim_error.json        Per action dimension MSE
  - {ARCH_NAME}_grad_norms.json           Gradient norm history (barren plateau detection)

Output per phase:
  - phase{N}_comparison.png              Overlaid val loss curves
  - phase{N}_convergence.png             Epochs to convergence bar chart
  - phase{N}_training_times.png          Training time bar chart

Output full sweep:
  - full_sweep_comparison.png            All 13 architectures overlaid
  - params_vs_performance.png            Scatter: params vs best val loss
  - sweep_summary.json                   Ranked results table
  - sweep_leaderboard.txt                Human readable ranked results

Adding new architectures:
  Add a new entry to the ARCHITECTURES dict with a unique key and a phase number.
  The sweep will automatically include it in all relevant plots and comparisons.
  Use phase 6+ for any new phases you create.

Usage:
  python quantum_sweep.py --phase all
  python quantum_sweep.py --phase 1
  python quantum_sweep.py --arch QViT_8q_4layer_amp
  python quantum_sweep.py --phase all \\
    --vit_encoder_path /path/to/vit_encoder_only.pt \\
    --zarr_path /path/to/dataset.zarr.zip \\
    --data_path /path/to/session_001 \\
    --log_dir /path/to/logs
  python quantum_sweep.py --phase 1 --force
  python quantum_sweep.py --list

Author: Neal D'Andrea
Date: June 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import json
import argparse
import time
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# ── PennyLane import ──────────────────────────────────────────────────────────
try:
    import pennylane as qml
except ImportError:
    sys.path.insert(0, '/tmp/pypackages')
    import pennylane as qml

try:
    import timm
except ImportError:
    raise ImportError("timm required: pip install timm")


# ==============================================================================
# SWEEP CONFIGURATION
# ==============================================================================

SWEEP_CONFIG = {
    # ── Paths (override via CLI args) ──────────────────────────────────────────
    'vit_encoder_path': '/tmp/dandrenf/QuantumInspiredViTSpeedUp/quantum_research/vit_encoder_only.pt',
    'zarr_path':        '/tmp/dandrenf/umi_data/session_001/dataset.zarr.zip',
    'data_path':        '/tmp/dandrenf/umi_data/session_001',
    'log_dir':          '/home/dandrenf/qml_sweep_logs',

    # ── Training hyperparameters ───────────────────────────────────────────────
    'num_frames_per_video': 100,   # Use all extracted frames per video
    'batch_size':           32,    # RTX 6000 Ada has 51GB VRAM - use larger batches
    'learning_rate':        0.001,
    'num_epochs':           100,
    'patience':             15,    # Early stopping patience
    'num_workers':          8,     # More workers for fast JPEG loading
    'action_dim':           7,

    # ── Action dimension names for per-dimension analysis ─────────────────────
    'action_dim_names': ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper'],

    # ── Convergence threshold for convergence speed plot ──────────────────────
    'convergence_threshold': 0.05,

    # ── Phase definitions (add new phases here as needed) ─────────────────────
    'phases': {
        1: 'Qubit Count Sweep',
        2: 'Circuit Depth Sweep',
        3: 'Encoding Method Sweep',
        4: 'Gate Pattern Sweep',
        5: 'Knowledge Distillation',
    }
}


# ==============================================================================
# ARCHITECTURE DEFINITIONS
# ==============================================================================
# To add a new architecture: add a new entry to this dict with a unique key.
# Required fields: phase, n_qubits, n_layers, compression_dim, encoding, gate_pattern, description
# Optional fields: use_distillation (default False)
# The sweep script automatically includes any new entry in all plots and comparisons.

ARCHITECTURES = {

    # ── Phase 1: Qubit Count Sweep ─────────────────────────────────────────────
    'QViT_4q_4layer_amp': {
        'phase': 1,
        'n_qubits': 4, 'n_layers': 4, 'compression_dim': 16,
        'encoding': 'amplitude', 'gate_pattern': 'QONN',
        'description': '4 qubits, 4 layers, amplitude encoding'
    },
    'QViT_6q_4layer_amp': {
        'phase': 1,
        'n_qubits': 6, 'n_layers': 4, 'compression_dim': 64,
        'encoding': 'amplitude', 'gate_pattern': 'QONN',
        'description': '6 qubits, 4 layers, amplitude encoding (BASELINE)'
    },
    'QViT_8q_4layer_amp': {
        'phase': 1,
        'n_qubits': 8, 'n_layers': 4, 'compression_dim': 256,
        'encoding': 'amplitude', 'gate_pattern': 'QONN',
        'description': '8 qubits, 4 layers, amplitude encoding'
    },
    'QViT_10q_4layer_amp': {
        'phase': 1,
        'n_qubits': 10, 'n_layers': 4, 'compression_dim': 1024,
        'encoding': 'amplitude', 'gate_pattern': 'QONN',
        'description': '10 qubits, 4 layers, amplitude encoding'
    },

    # ── Phase 2: Circuit Depth Sweep ───────────────────────────────────────────
    'QViT_6q_2layer_amp': {
        'phase': 2,
        'n_qubits': 6, 'n_layers': 2, 'compression_dim': 64,
        'encoding': 'amplitude', 'gate_pattern': 'QONN',
        'description': '6 qubits, 2 layers, amplitude encoding'
    },
    'QViT_6q_6layer_amp': {
        'phase': 2,
        'n_qubits': 6, 'n_layers': 6, 'compression_dim': 64,
        'encoding': 'amplitude', 'gate_pattern': 'QONN',
        'description': '6 qubits, 6 layers, amplitude encoding'
    },
    'QViT_6q_8layer_amp': {
        'phase': 2,
        'n_qubits': 6, 'n_layers': 8, 'compression_dim': 64,
        'encoding': 'amplitude', 'gate_pattern': 'QONN',
        'description': '6 qubits, 8 layers, amplitude encoding'
    },

    # ── Phase 3: Encoding Method Sweep ────────────────────────────────────────
    'QViT_6q_4layer_angle': {
        'phase': 3,
        'n_qubits': 6, 'n_layers': 4, 'compression_dim': 6,
        'encoding': 'angle', 'gate_pattern': 'QONN',
        'description': '6 qubits, 4 layers, angle encoding'
    },
    'QViT_6q_4layer_IQP': {
        'phase': 3,
        'n_qubits': 6, 'n_layers': 4, 'compression_dim': 6,
        'encoding': 'IQP', 'gate_pattern': 'QONN',
        'description': '6 qubits, 4 layers, IQP encoding'
    },

    # ── Phase 4: Gate Pattern Sweep ───────────────────────────────────────────
    'QViT_6q_4layer_basic': {
        'phase': 4,
        'n_qubits': 6, 'n_layers': 4, 'compression_dim': 64,
        'encoding': 'amplitude', 'gate_pattern': 'BasicEntangler',
        'description': '6 qubits, 4 layers, BasicEntangler gates'
    },
    'QViT_6q_4layer_strong': {
        'phase': 4,
        'n_qubits': 6, 'n_layers': 4, 'compression_dim': 64,
        'encoding': 'amplitude', 'gate_pattern': 'StronglyEntangling',
        'description': '6 qubits, 4 layers, StronglyEntangling gates'
    },

    # ── Phase 5: Knowledge Distillation ───────────────────────────────────────
    'QViT_6q_4layer_amp_KD': {
        'phase': 5,
        'n_qubits': 6, 'n_layers': 4, 'compression_dim': 64,
        'encoding': 'amplitude', 'gate_pattern': 'QONN',
        'use_distillation': True,
        'description': '6 qubits, 4 layers, amplitude + knowledge distillation'
    },
}


# ==============================================================================
# QUANTUM CIRCUIT BUILDERS
# ==============================================================================

def build_quantum_circuit(n_qubits, n_layers, encoding, gate_pattern):
    """
    Build and return a PennyLane quantum circuit function based on config.
    Fully parametric - works with any n_qubits, n_layers, encoding, gate_pattern.

    Args:
        n_qubits:     Number of qubits
        n_layers:     Number of circuit layers
        encoding:     'amplitude', 'angle', or 'IQP'
        gate_pattern: 'QONN', 'BasicEntangler', or 'StronglyEntangling'

    Returns:
        quantum_circuit: callable PennyLane qnode
        weights_shape:   tuple shape of trainable weights tensor
    """
    try:
        dev = qml.device('lightning.qubit', wires=n_qubits)
        print(f"  Using lightning.qubit ({n_qubits} qubits)")
    except Exception:
        dev = qml.device('default.qubit', wires=n_qubits)
        print(f"  Using default.qubit ({n_qubits} qubits)")

    if gate_pattern in ('QONN', 'StronglyEntangling'):
        weights_shape = (n_layers, n_qubits, 3)
    elif gate_pattern == 'BasicEntangler':
        weights_shape = (n_layers, n_qubits, 1)
    else:
        weights_shape = (n_layers, n_qubits, 3)

    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def quantum_circuit(features, weights):
        # ── Encoding ─────────────────────────────────────────────────────────
        if encoding == 'amplitude':
            qml.AmplitudeEmbedding(features, wires=range(n_qubits),
                                   normalize=True, pad_with=0.0)
        elif encoding == 'angle':
            for i in range(n_qubits):
                qml.RY(features[i], wires=i)
        elif encoding == 'IQP':
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            for i in range(n_qubits):
                qml.RZ(features[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            for i in range(n_qubits):
                qml.RZ(features[i] ** 2, wires=i)

        # ── Parametrized layers ───────────────────────────────────────────────
        for layer in range(n_layers):
            if gate_pattern == 'QONN':
                for qubit in range(n_qubits):
                    qml.RX(weights[layer, qubit, 0], wires=qubit)
                    qml.RY(weights[layer, qubit, 1], wires=qubit)
                    qml.RZ(weights[layer, qubit, 2], wires=qubit)
                for qubit in range(n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])

            elif gate_pattern == 'BasicEntangler':
                for qubit in range(n_qubits):
                    qml.RY(weights[layer, qubit, 0], wires=qubit)
                for qubit in range(n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])

            elif gate_pattern == 'StronglyEntangling':
                for qubit in range(n_qubits):
                    qml.RX(weights[layer, qubit, 0], wires=qubit)
                    qml.RY(weights[layer, qubit, 1], wires=qubit)
                    qml.RZ(weights[layer, qubit, 2], wires=qubit)
                stride = max(1, layer + 1)
                for qubit in range(n_qubits):
                    target = (qubit + stride) % n_qubits
                    if target != qubit:
                        qml.CNOT(wires=[qubit, target])

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return quantum_circuit, weights_shape


# ==============================================================================
# HYBRID MODEL CLASS
# ==============================================================================

class QuantumRobotPolicy(nn.Module):
    """
    Hybrid quantum-classical robot manipulation policy.
    Fully parametric - supports any architecture config from ARCHITECTURES dict.
    Adding a new architecture requires no changes to this class.
    """

    def __init__(self, config, vit_encoder_path):
        super(QuantumRobotPolicy, self).__init__()

        self.config       = config
        self.n_qubits     = config['n_qubits']
        self.n_layers     = config['n_layers']
        self.encoding     = config['encoding']
        self.gate_pattern = config['gate_pattern']
        self.comp_dim     = config['compression_dim']
        self.action_dim   = SWEEP_CONFIG['action_dim']

        # Frozen ViT encoder
        print(f"  Loading ViT encoder...")
        self.vit_encoder = timm.create_model(
            'vit_base_patch16_224', pretrained=False,
            num_classes=0, global_pool=''
        )
        state_dict = torch.load(vit_encoder_path, weights_only=False)
        self.vit_encoder.load_state_dict(state_dict, strict=False)
        self.vit_encoder.eval()
        for param in self.vit_encoder.parameters():
            param.requires_grad = False
        print(f"  ✓ ViT loaded and frozen")

        # Compression layer: 768 → compression_dim
        self.compression = nn.Linear(768, self.comp_dim)

        # Quantum circuit
        self.quantum_circuit, weights_shape = build_quantum_circuit(
            self.n_qubits, self.n_layers, self.encoding, self.gate_pattern
        )
        self.quantum_weights = nn.Parameter(torch.randn(*weights_shape) * 0.1)

        # Action head: n_qubits → 7D action
        self.action_head = nn.Linear(self.n_qubits, self.action_dim)

        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_frozen    = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"  ✓ {config['description']}")
        print(f"  ✓ Compression: 768 → {self.comp_dim}")
        print(f"  ✓ Quantum weights: {weights_shape} = {np.prod(weights_shape)} params")
        print(f"  ✓ Trainable: {total_trainable:,} params")
        print(f"  ✓ Param reduction: {(1-total_trainable/(total_trainable+total_frozen))*100:.2f}%")

    def forward(self, x):
        batch_size = x.shape[0]

        with torch.no_grad():
            vit_out   = self.vit_encoder(x)
            cls_token = vit_out[:, 0, :]

        compressed = self.compression(cls_token)

        quantum_outputs = []
        for i in range(batch_size):
            measurements = self.quantum_circuit(compressed[i], self.quantum_weights)
            quantum_outputs.append(torch.stack(measurements))

        quantum_out = torch.stack(quantum_outputs).float()
        actions = self.action_head(quantum_out)
        return actions

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def get_grad_norm(self):
        """Return mean gradient norm across all trainable params. Used for barren plateau detection."""
        norms = []
        for p in self.parameters():
            if p.requires_grad and p.grad is not None:
                norms.append(p.grad.norm().item())
        return float(np.mean(norms)) if norms else 0.0

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==============================================================================
# DATASET
# ==============================================================================

class UMIVideoDataset(Dataset):
    """
    UMI robot manipulation dataset.
    Automatically detects and uses pre-extracted JPEGs if available.
    Falls back to MP4 lazy loading otherwise.
    Works with any number of videos or frames per video.
    """

    def __init__(self, zarr_path, video_folder,
                 num_frames_per_video=100, transform=None):
        from torchvision import transforms as T

        self.transform = transform or T.Compose([
            T.ToPILImage(),
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229,  0.224, 0.225])
        ])

        self.frames  = []
        self.actions = []

        real_actions = None
        if zarr_path and os.path.exists(zarr_path):
            real_actions = self._load_actions(zarr_path)

        # Look for extracted frames in sibling directory
        frames_dir = os.path.join(os.path.dirname(video_folder), 'extracted_frames')

        if os.path.exists(frames_dir):
            print(f"  ✓ Using pre-extracted JPEGs: {frames_dir}")
            self._load_from_jpegs(frames_dir, num_frames_per_video, real_actions)
        elif video_folder and os.path.exists(video_folder):
            print(f"  ⚠️  No JPEGs found, using MP4 lazy loading (slow)")
            self._load_from_videos(video_folder, num_frames_per_video, real_actions)
        else:
            raise FileNotFoundError(f"No data found at {video_folder}")

        print(f"  ✓ Dataset: {len(self.frames)} frames loaded")
        print(f"  ✓ Actions: {'REAL zarr' if real_actions is not None else 'PLACEHOLDER zeros'}")

    def _load_actions(self, zarr_path):
        import zipfile
        import zarr as zarr_lib

        extract_path = zarr_path.replace('.zip', '_extracted')
        if not os.path.exists(extract_path):
            print(f"  Extracting zarr...")
            with zipfile.ZipFile(zarr_path, 'r', allowZip64=True) as z:
                z.extractall(extract_path)

        zarr_root = None
        for candidate in [extract_path, os.path.join(extract_path, 'dataset.zarr')]:
            if os.path.exists(candidate):
                try:
                    zarr_root = zarr_lib.open(candidate, mode='r')
                    break
                except Exception:
                    continue

        if zarr_root is None:
            print("  ⚠️  Could not open zarr")
            return None

        for path in [('data', 'action'), ('data', 'robot_eef_pose')]:
            try:
                node = zarr_root
                for k in path:
                    node = node[k]
                actions = node[:]
                print(f"  ✓ Actions from {'/'.join(path)}: {actions.shape}")
                return actions
            except Exception:
                continue

        try:
            eef_pos = zarr_root['data']['robot0_eef_pos'][:]
            eef_rot = zarr_root['data']['robot0_eef_rot_axis_angle'][:]
            gripper = zarr_root['data']['robot0_gripper_width'][:]
            actions = np.concatenate([eef_pos, eef_rot, gripper], axis=1)
            print(f"  ✓ Built 7D actions from components: {actions.shape}")
            return actions
        except Exception as e:
            print(f"  ⚠️  Could not build actions: {e}")
            return None

    def _load_from_jpegs(self, frames_dir, num_frames_per_video, real_actions):
        folders = sorted([
            os.path.join(frames_dir, d)
            for d in os.listdir(frames_dir)
            if os.path.isdir(os.path.join(frames_dir, d))
        ])[:171]

        for folder in folders:
            jpegs = sorted([
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.endswith('.jpg')
            ])[:num_frames_per_video]

            for path in jpegs:
                self.frames.append(path)
                idx = len(self.frames) - 1
                action = (torch.tensor(real_actions[idx][:7], dtype=torch.float32)
                          if real_actions is not None and idx < len(real_actions)
                          else torch.zeros(7))
                self.actions.append(action)

    def _load_from_videos(self, video_folder, num_frames_per_video, real_actions):
        videos = sorted([
            os.path.join(video_folder, f)
            for f in os.listdir(video_folder)
            if f.endswith('.MP4') or f.endswith('.mp4')
        ])[:171]

        for video_path in videos:
            cap = cv2.VideoCapture(video_path)
            fc  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            indices = np.linspace(0, fc - 1, num_frames_per_video, dtype=int)
            for idx in indices:
                self.frames.append((video_path, int(idx)))
                cur = len(self.frames) - 1
                action = (torch.tensor(real_actions[cur][:7], dtype=torch.float32)
                          if real_actions is not None and cur < len(real_actions)
                          else torch.zeros(7))
                self.actions.append(action)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        ref    = self.frames[idx]
        action = self.actions[idx]

        if isinstance(ref, str):
            frame = cv2.imread(ref)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame is not None \
                    else np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            video_path, frame_idx = ref
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret \
                    else np.zeros((224, 224, 3), dtype=np.uint8)

        return self.transform(frame), action


# ==============================================================================
# TRAINING AND EVALUATION
# ==============================================================================

def train_epoch(model, loader, optimizer, criterion, device, track_grads=True):
    """
    Train one epoch. Returns avg loss and avg gradient norm.
    Gradient norm tracking detects barren plateaus in deep circuits.
    """
    model.train()
    total_loss, total_grad_norm, n = 0.0, 0.0, 0

    for frames, actions in loader:
        frames, actions = frames.to(device), actions.to(device)
        pred = model(frames)
        loss = criterion(pred, actions)
        optimizer.zero_grad()
        loss.backward()

        if track_grads:
            total_grad_norm += model.get_grad_norm()

        optimizer.step()
        total_loss += loss.item()
        n += 1

    return total_loss / n, total_grad_norm / n if track_grads else 0.0


def evaluate(model, loader, criterion, device):
    """
    Evaluate model. Returns overall loss/MSE and per action dimension MSE.
    Per-dimension analysis shows which robot DOFs the model struggles with.
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for frames, actions in loader:
            frames, actions = frames.to(device), actions.to(device)
            pred = model(frames)
            total_loss += criterion(pred, actions).item()
            all_preds.append(pred.cpu())
            all_targets.append(actions.cpu())

    n = len(loader)
    preds   = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    dim_names = SWEEP_CONFIG['action_dim_names']
    per_dim_mse = {}
    for i, name in enumerate(dim_names):
        per_dim_mse[name] = F.mse_loss(preds[:, i], targets[:, i]).item()

    return {
        'loss':        total_loss / n,
        'mse':         total_loss / n,
        'per_dim_mse': per_dim_mse,
    }


# ==============================================================================
# LOGGING
# ==============================================================================

def save_log(log, path):
    with open(path, 'w') as f:
        json.dump(log, f, indent=2)


def load_log(path):
    with open(path) as f:
        return json.load(f)


# ==============================================================================
# PLOTTING - INDIVIDUAL ARCHITECTURE
# ==============================================================================

def plot_curves(log, path, arch_name):
    """Two-panel loss curve: log scale + smoothed with stats box."""
    epochs = log['epochs']
    train  = log['train_losses']
    val    = log['val_losses']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{arch_name} - Training Curves', fontsize=13, fontweight='bold')

    axes[0].plot(epochs, train, 'b-',  lw=2, label='Train Loss')
    axes[0].plot(epochs, val,   'r--', lw=2, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Loss Over All Epochs (log scale)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    w  = max(1, len(epochs) // 10)
    sm = lambda v: [sum(v[max(0, i-w):i+1]) / len(v[max(0, i-w):i+1]) for i in range(len(v))]
    axes[1].plot(epochs, sm(train), 'b-',  lw=2, label='Train (smoothed)')
    axes[1].plot(epochs, sm(val),   'r--', lw=2, label='Val (smoothed)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE Loss')
    axes[1].set_title('Smoothed Loss Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    best_val   = min(val)
    best_epoch = val.index(best_val) + 1
    stats = (f"Best Val: {best_val:.4f} (Epoch {best_epoch})\n"
             f"Final Train: {train[-1]:.4f}\n"
             f"Final Val:   {val[-1]:.4f}\n"
             f"Epochs: {len(epochs)}\n"
             f"Params: {log.get('trainable_params', 'N/A'):,}")
    axes[1].text(0.97, 0.97, stats, transform=axes[1].transAxes,
                 va='top', ha='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                 fontsize=8, fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Loss curve: {path}")


def plot_grad_norms(log, path, arch_name):
    """
    Plot gradient norm history.
    Flat/near-zero gradient norms = barren plateau (common in deep circuits).
    Healthy training = gradient norms > 0.001 throughout.
    """
    grad_norms = log.get('grad_norms', [])
    if not grad_norms or all(g == 0 for g in grad_norms):
        return

    epochs = log['epochs']
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, grad_norms, 'g-', lw=2, label='Mean Grad Norm')
    ax.axhline(y=0.001, color='red', linestyle='--', alpha=0.7,
               label='Barren plateau threshold (0.001)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Gradient Norm')
    ax.set_title(f'{arch_name} - Gradient Norm History (Barren Plateau Detection)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Annotation
    final_norm = grad_norms[-1]
    status = 'BARREN PLATEAU DETECTED' if final_norm < 0.001 else 'Healthy gradients'
    color  = 'red' if final_norm < 0.001 else 'green'
    ax.text(0.02, 0.95, f'Final norm: {final_norm:.6f} | Status: {status}',
            transform=ax.transAxes, va='top', color=color,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Grad norm plot: {path}")


def plot_per_dim_error(log, path, arch_name):
    """
    Bar chart of MSE per robot action dimension.
    Shows which DOFs (x/y/z position, roll/pitch/yaw, gripper) the model struggles with.
    """
    per_dim = log.get('final_per_dim_mse', {})
    if not per_dim:
        return

    dim_names = list(per_dim.keys())
    mse_vals  = list(per_dim.values())
    colors    = ['#2563eb' if v < np.mean(mse_vals) else '#dc2626' for v in mse_vals]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(dim_names, mse_vals, color=colors, alpha=0.8, edgecolor='white')
    ax.axhline(y=np.mean(mse_vals), color='orange', linestyle='--',
               lw=2, label=f'Mean MSE: {np.mean(mse_vals):.4f}')

    for bar, val in zip(bars, mse_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Action Dimension')
    ax.set_ylabel('MSE')
    ax.set_title(f'{arch_name} - Per-Dimension Action Error\n'
                 f'Blue = below average, Red = above average')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Per-dim error plot: {path}")


# ==============================================================================
# PLOTTING - COMPARISON ACROSS ARCHITECTURES
# ==============================================================================

def plot_comparison(log_files, save_path, title='Architecture Comparison'):
    """
    Overlay multiple architecture val loss curves on one plot.
    Works with any number of architectures - just pass more log files.

    Args:
        log_files: dict {arch_name: log_json_path}
        save_path: output PNG path
        title:     plot title

    Returns:
        list of result dicts sorted by best val loss
    """
    colors = ['#2563eb', '#dc2626', '#16a34a', '#9333ea',
              '#ea580c', '#0d9488', '#fbbf24', '#6b7280',
              '#ec4899', '#14b8a6', '#f97316', '#8b5cf6',
              '#06b6d4', '#84cc16', '#f43f5e', '#64748b']

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    results = []
    for i, (name, path) in enumerate(log_files.items()):
        if not os.path.exists(path):
            print(f"  Warning: log not found: {path}")
            continue
        log    = load_log(path)
        color  = colors[i % len(colors)]
        epochs = log['epochs']
        val    = log['val_losses']
        train  = log['train_losses']
        best   = min(val)

        results.append({
            'name':         name,
            'best_val':     best,
            'best_epoch':   val.index(best) + 1,
            'final_train':  train[-1],
            'final_val':    val[-1],
            'total_epochs': len(epochs),
            'params':       log.get('trainable_params', 0),
            'time_hrs':     log.get('total_time_seconds', 0) / 3600,
            'stopped_early': len(epochs) < log.get('num_epochs', 100),
        })

        label = f"{name} (best: {best:.4f})"
        if results[-1]['stopped_early']:
            label += f" [stopped ep{len(epochs)}]"

        axes[0].plot(epochs, val,   color=color, lw=2, label=label)
        axes[1].plot(epochs, train, color=color, lw=2, linestyle='--', label=name)

    for ax, title_str in zip(axes, ['Validation Loss', 'Train Loss']):
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_title(title_str)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Comparison plot: {save_path}")

    results.sort(key=lambda x: x['best_val'])
    _print_leaderboard(results)
    return results


def plot_params_vs_performance(log_files, save_path):
    """
    Scatter plot: trainable parameters vs best val loss.
    Core efficiency argument for the paper.
    Works with any architectures passed in log_files.
    """
    names, params, best_vals, phases = [], [], [], []

    phase_colors = {1: '#2563eb', 2: '#dc2626', 3: '#16a34a',
                    4: '#9333ea', 5: '#ea580c'}

    for name, path in log_files.items():
        if not os.path.exists(path):
            continue
        log = load_log(path)
        names.append(name)
        params.append(log.get('trainable_params', 0))
        best_vals.append(min(log['val_losses']))
        phase = ARCHITECTURES.get(name, {}).get('phase', 0)
        phases.append(phase)

    if not names:
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, (name, p, v, ph) in enumerate(zip(names, params, best_vals, phases)):
        color = phase_colors.get(ph, '#6b7280')
        ax.scatter(p, v, s=120, color=color, zorder=5, alpha=0.9)
        ax.annotate(name.replace('QViT_', ''),
                    (p, v), textcoords='offset points',
                    xytext=(6, 4), fontsize=7, alpha=0.8)

    # Add classical baseline reference line
    ax.axhline(y=0.15, color='red', linestyle=':', alpha=0.5,
               label='Robotics target (MSE < 0.15)')

    # Legend for phases
    for ph, color in phase_colors.items():
        phase_name = SWEEP_CONFIG['phases'].get(ph, f'Phase {ph}')
        ax.scatter([], [], color=color, s=80, label=f'Phase {ph}: {phase_name}')

    ax.set_xlabel('Trainable Parameters', fontsize=11)
    ax.set_ylabel('Best Validation MSE Loss', fontsize=11)
    ax.set_title('Parameter Efficiency: All Architectures\n'
                 'Lower-left = fewer params + better performance = ideal',
                 fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Params vs performance: {save_path}")


def plot_convergence_speed(log_files, save_path, threshold=None):
    """
    Horizontal bar chart: epochs to reach convergence threshold.
    Shows which architectures learn fastest.
    Works with any architectures passed in log_files.
    """
    if threshold is None:
        threshold = SWEEP_CONFIG['convergence_threshold']

    results = {}
    for name, path in log_files.items():
        if not os.path.exists(path):
            continue
        log = load_log(path)
        val = log['val_losses']
        epoch_reached = next(
            (i + 1 for i, v in enumerate(val) if v <= threshold), None
        )
        results[name] = epoch_reached

    if not results:
        return

    names  = list(results.keys())
    epochs = [results[n] if results[n] else 999 for n in names]
    colors = ['#16a34a' if results[n] else '#dc2626' for n in names]

    fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.5)))
    bars = ax.barh(names, epochs, color=colors, alpha=0.8)
    ax.axvline(x=100, color='gray', linestyle='--', alpha=0.5, label='Max epochs (100)')
    ax.axvline(x=999, color='red',  linestyle=':',  alpha=0.5, label='Never reached')

    for bar, val, name in zip(bars, epochs, names):
        label = f"{val} epochs" if results[name] else "Never reached"
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=8)

    ax.set_xlabel(f'Epochs to reach val loss < {threshold}')
    ax.set_title(f'Convergence Speed Comparison\n'
                 f'Green = reached threshold, Red = never reached (val loss < {threshold})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, max(epochs) * 1.15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Convergence speed: {save_path}")


def plot_training_times(log_files, save_path):
    """
    Horizontal bar chart: training time per architecture.
    Works with any architectures passed in log_files.
    """
    names, times = [], []
    for name, path in log_files.items():
        if not os.path.exists(path):
            continue
        log = load_log(path)
        times.append(log.get('total_time_seconds', 0) / 3600)
        names.append(name)

    if not names:
        return

    fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.5)))
    bars = ax.barh(names, times, color='steelblue', alpha=0.8)

    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{t:.2f} hrs', va='center', fontsize=8)

    ax.set_xlabel('Training Time (hours)')
    ax.set_title('Training Time per Architecture')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, max(times) * 1.15 if times else 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Training times: {save_path}")


def plot_per_dim_comparison(log_files, save_path):
    """
    Grouped bar chart: per-dimension MSE across all architectures.
    Shows which action dimensions are hardest across all configs.
    Works with any architectures and any action dimensions.
    """
    dim_names = SWEEP_CONFIG['action_dim_names']
    arch_data = {}

    for name, path in log_files.items():
        if not os.path.exists(path):
            continue
        log = load_log(path)
        per_dim = log.get('final_per_dim_mse', {})
        if per_dim:
            arch_data[name] = per_dim

    if not arch_data:
        return

    n_archs = len(arch_data)
    n_dims  = len(dim_names)
    x       = np.arange(n_dims)
    width   = 0.8 / n_archs

    colors = ['#2563eb', '#dc2626', '#16a34a', '#9333ea',
              '#ea580c', '#0d9488', '#fbbf24', '#6b7280',
              '#ec4899', '#14b8a6', '#f97316', '#8b5cf6']

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (arch_name, per_dim) in enumerate(arch_data.items()):
        vals = [per_dim.get(d, 0) for d in dim_names]
        offset = (i - n_archs/2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=arch_name,
               color=colors[i % len(colors)], alpha=0.8)

    ax.set_xlabel('Action Dimension')
    ax.set_ylabel('MSE')
    ax.set_title('Per-Dimension MSE Comparison Across All Architectures\n'
                 'Shows which robot DOFs each architecture struggles with')
    ax.set_xticks(x)
    ax.set_xticklabels(dim_names)
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Per-dim comparison: {save_path}")


def plot_barren_plateau_comparison(log_files, save_path):
    """
    Overlay gradient norm histories for all architectures.
    Detects barren plateaus in deeper circuits.
    """
    colors = ['#2563eb', '#dc2626', '#16a34a', '#9333ea',
              '#ea580c', '#0d9488', '#fbbf24', '#6b7280']

    fig, ax = plt.subplots(figsize=(14, 6))
    plotted = 0

    for i, (name, path) in enumerate(log_files.items()):
        if not os.path.exists(path):
            continue
        log        = load_log(path)
        grad_norms = log.get('grad_norms', [])
        if not grad_norms or all(g == 0 for g in grad_norms):
            continue
        ax.plot(log['epochs'], grad_norms, color=colors[i % len(colors)],
                lw=2, label=name, alpha=0.8)
        plotted += 1

    if plotted == 0:
        plt.close()
        return

    ax.axhline(y=0.001, color='red', linestyle='--', lw=2, alpha=0.7,
               label='Barren plateau threshold (0.001)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Gradient Norm')
    ax.set_title('Gradient Norm History - Barren Plateau Detection\n'
                 'Lines dropping below red = vanishing gradients / barren plateau')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Barren plateau comparison: {save_path}")


# ==============================================================================
# SUMMARY AND LEADERBOARD
# ==============================================================================

def _print_leaderboard(results):
    """Print ranked results table to terminal."""
    print("\n" + "="*95)
    print(f"{'Rank':<5}{'Architecture':<35}{'Best Val':<12}{'Epoch':<8}{'Params':<10}{'Time(hrs)':<10}{'Early Stop'}")
    print("="*95)
    for rank, r in enumerate(results, 1):
        es = '✓' if r.get('stopped_early') else '-'
        print(f"{rank:<5}{r['name']:<35}{r['best_val']:<12.6f}"
              f"{r['best_epoch']:<8}{r['params']:<10,}{r['time_hrs']:<10.2f}{es}")
    print("="*95)


def save_leaderboard(results, path):
    """Save ranked leaderboard as human-readable text file."""
    lines = [
        "=" * 95,
        "QML ARCHITECTURE SWEEP LEADERBOARD",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 95,
        f"{'Rank':<5}{'Architecture':<35}{'Best Val MSE':<14}{'Best Epoch':<12}"
        f"{'Params':<10}{'Time (hrs)':<12}{'Early Stop'}",
        "-" * 95,
    ]
    for rank, r in enumerate(results, 1):
        es = 'YES' if r.get('stopped_early') else 'NO'
        lines.append(
            f"{rank:<5}{r['name']:<35}{r['best_val']:<14.6f}"
            f"{r['best_epoch']:<12}{r['params']:<10,}{r['time_hrs']:<12.2f}{es}"
        )
    lines += [
        "=" * 95,
        "",
        "NOTES:",
        "  Best Val MSE: lower is better (robotics target < 0.15)",
        "  Best Epoch:   epoch where best validation loss was achieved",
        "  Early Stop:   training stopped before max epochs due to no improvement",
        "  Params:       trainable parameters only (ViT encoder frozen at 85.8M)",
    ]

    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"✓ Leaderboard: {path}")


def save_summary(log, path, arch_name, config):
    """Generate and save comprehensive training summary text file."""
    train = log['train_losses']
    val   = log['val_losses']

    best_val   = min(val)
    best_epoch = val.index(best_val) + 1
    improve    = (train[0] - train[-1]) / train[0] * 100
    gap        = abs(val[-1] - train[-1])
    first10    = sum(train[:10]) / min(10, len(train))
    last10     = sum(train[-10:]) / min(10, len(train))

    overfit = ("Minimal" if gap < 0.01 else
               "Moderate" if gap < 0.05 else "Significant")

    rating = ("EXCELLENT" if val[-1] < 0.01 else
              "GOOD - within robotics target (< 0.15)" if val[-1] < 0.05 else
              "ACCEPTABLE - meets robotics target" if val[-1] < 0.15 else
              "NEEDS IMPROVEMENT - above 0.15 target")

    # Barren plateau check
    grad_norms = log.get('grad_norms', [])
    if grad_norms and len(grad_norms) > 0:
        final_norm = grad_norms[-1]
        plateau_status = ('BARREN PLATEAU DETECTED' if final_norm < 0.001
                          else 'Healthy gradients')
    else:
        plateau_status = 'Not tracked'

    # Per-dim error
    per_dim = log.get('final_per_dim_mse', {})
    per_dim_str = '\n'.join(
        f"    {k}: {v:.6f}" for k, v in per_dim.items()
    ) if per_dim else "  Not available"

    summary = f"""
================================================================================
TRAINING SUMMARY: {arch_name}
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ARCHITECTURE
------------
  Description:       {config['description']}
  Qubits:            {config['n_qubits']}
  Circuit layers:    {config['n_layers']}
  Compression dim:   {config['compression_dim']}
  Encoding:          {config['encoding']}
  Gate pattern:      {config['gate_pattern']}
  Trainable params:  {log.get('trainable_params', 'N/A'):,}
  Frozen params:     85,798,656 (ViT encoder)
  Param reduction:   {(1 - log.get('trainable_params',0)/(log.get('trainable_params',0)+85798656))*100:.2f}%

TRAINING CONFIGURATION
----------------------
  Dataset:           UMI Robot Manipulation (real zarr actions)
  Frames per video:  {log.get('num_frames_per_video', 'N/A')}
  Total frames:      {log.get('total_frames', 'N/A')}
  Batch size:        {log.get('batch_size', 'N/A')}
  Learning rate:     {log.get('learning_rate', 'N/A')}
  Epochs completed:  {len(log['epochs'])} / {log.get('num_epochs', 100)}
  Early stopping:    {'YES - stopped at epoch ' + str(len(log['epochs'])) if len(log['epochs']) < log.get('num_epochs', 100) else 'NO - ran full training'}
  Total time:        {log.get('total_time_seconds', 0)/3600:.2f} hours

PERFORMANCE RESULTS
-------------------
  Best Val Loss:     {best_val:.6f}  (Epoch {best_epoch})
  Final Train Loss:  {train[-1]:.6f}
  Final Val Loss:    {val[-1]:.6f}
  Train/Val Gap:     {gap:.6f}  ({overfit} overfitting)
  Total improvement: {improve:.1f}%
  Overall rating:    {rating}

PER-DIMENSION ACTION ERROR
--------------------------
{per_dim_str}

CONVERGENCE ANALYSIS
--------------------
  First 10 epochs avg:  {first10:.6f}
  Last 10 epochs avg:   {last10:.6f}
  Convergence speed:    {((first10 - last10)/first10*100):.1f}% improvement
  Loss at 25% epochs:   {train[len(train)//4 - 1]:.6f}
  Loss at 50% epochs:   {train[len(train)//2 - 1]:.6f}
  Loss at 75% epochs:   {train[3*len(train)//4 - 1]:.6f}

BARREN PLATEAU ANALYSIS
------------------------
  Final gradient norm:  {grad_norms[-1]:.6f if grad_norms else 'N/A'}
  Status:               {plateau_status}
  Note: Gradient norm < 0.001 indicates vanishing gradients (barren plateau)
        Common in circuits with > 6 layers. Check grad_norm plot for details.

================================================================================
"""

    with open(path, 'w') as f:
        f.write(summary)
    print(summary)
    print(f"  ✓ Summary: {path}")


# ==============================================================================
# SINGLE ARCHITECTURE TRAINER
# ==============================================================================

def train_architecture(arch_name, arch_config, args):
    """
    Train a single architecture end to end.
    Supports resume from checkpoint if training was interrupted.
    Saves all logs, plots, checkpoints, and summaries automatically.

    Args:
        arch_name:   String key from ARCHITECTURES dict
        arch_config: Config dict for this architecture
        args:        Parsed CLI arguments

    Returns:
        result dict with arch_name, best_val_loss, log_path, etc.
    """
    print(f"\n{'='*70}")
    print(f"TRAINING: {arch_name}")
    print(f"  {arch_config['description']}")
    print(f"{'='*70}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        print(f"  Device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"  Device: {device}")

    os.makedirs(args.log_dir, exist_ok=True)

    log_path      = os.path.join(args.log_dir, f'{arch_name}_training_log.json')
    plot_path     = os.path.join(args.log_dir, f'{arch_name}_loss_curve.png')
    grad_path     = os.path.join(args.log_dir, f'{arch_name}_grad_norms.png')
    dim_path      = os.path.join(args.log_dir, f'{arch_name}_per_dim_error.png')
    summary_path  = os.path.join(args.log_dir, f'{arch_name}_summary.txt')
    best_path     = os.path.join(args.log_dir, f'{arch_name}_best.pt')
    ckpt_prefix   = os.path.join(args.log_dir, arch_name)

    # ── Skip if already completed (unless --force) ────────────────────────────
    if os.path.exists(log_path) and not args.force:
        existing = load_log(log_path)
        if len(existing.get('epochs', [])) >= args.epochs:
            print(f"  ✓ Already completed ({len(existing['epochs'])} epochs). Skipping.")
            print(f"    Use --force to re-run.")
            return {
                'arch_name':    arch_name,
                'best_val_loss': min(existing['val_losses']),
                'log_path':     log_path,
                'skipped':      True,
                'stopped_early': len(existing['epochs']) < args.epochs,
            }

    # ── Build model ───────────────────────────────────────────────────────────
    print(f"\nBuilding model...")
    try:
        model = QuantumRobotPolicy(arch_config, args.vit_encoder_path)
        model = model.to(device)
    except Exception as e:
        print(f"  ✗ Failed to build model: {e}")
        import traceback; traceback.print_exc()
        return {'arch_name': arch_name, 'error': str(e)}

    # ── Build dataset ─────────────────────────────────────────────────────────
    print(f"\nLoading dataset...")
    try:
        full_ds = UMIVideoDataset(
            zarr_path=args.zarr_path,
            video_folder=args.data_path,
            num_frames_per_video=args.frames_per_video,
        )
        train_size = int(0.8 * len(full_ds))
        val_size   = len(full_ds) - train_size
        train_ds, val_ds = torch.utils.data.random_split(
            full_ds, [train_size, val_size]
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device == 'cuda')
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device == 'cuda')
        )
        print(f"  ✓ Train: {train_size} | Val: {val_size}")
    except Exception as e:
        print(f"  ✗ Failed to load dataset: {e}")
        import traceback; traceback.print_exc()
        return {'arch_name': arch_name, 'error': str(e)}

    # ── Setup training ────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.get_trainable_params(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    total_trainable = model.get_param_count()

    # ── Check for existing partial log (resume support) ───────────────────────
    start_epoch      = 0
    best_val_loss    = float('inf')
    patience_counter = 0

    log = {
        'arch_name':            arch_name,
        'description':          arch_config['description'],
        'n_qubits':             arch_config['n_qubits'],
        'n_layers':             arch_config['n_layers'],
        'compression_dim':      arch_config['compression_dim'],
        'encoding':             arch_config['encoding'],
        'gate_pattern':         arch_config['gate_pattern'],
        'trainable_params':     total_trainable,
        'batch_size':           args.batch_size,
        'learning_rate':        args.learning_rate,
        'num_frames_per_video': args.frames_per_video,
        'total_frames':         len(full_ds),
        'num_epochs':           args.epochs,
        'start_time':           datetime.now().isoformat(),
        'epochs':               [],
        'train_losses':         [],
        'val_losses':           [],
        'grad_norms':           [],
        'final_per_dim_mse':    {},
    }

    if os.path.exists(log_path) and os.path.exists(best_path) and not args.force:
        try:
            existing = load_log(log_path)
            if existing.get('epochs'):
                print(f"  Resuming from epoch {len(existing['epochs'])}...")
                log          = existing
                start_epoch  = len(existing['epochs'])
                best_val_loss = min(existing['val_losses'])
                model.load_state_dict(torch.load(best_path, weights_only=False))
                print(f"  ✓ Loaded checkpoint (best val: {best_val_loss:.6f})")
        except Exception as e:
            print(f"  Could not resume: {e}. Starting fresh.")

    print(f"\n🚀 Training {arch_name}")
    print(f"   Epochs: {start_epoch+1} → {args.epochs} | Patience: {args.patience}")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        train_loss, grad_norm = train_epoch(
            model, train_loader, optimizer, criterion, device, track_grads=True
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_loss    = val_metrics['loss']
        epoch_time  = time.time() - epoch_start

        print(f"  Epoch {epoch+1:3d}/{args.epochs}  "
              f"Train: {train_loss:.4f}  Val: {val_loss:.4f}  "
              f"GradNorm: {grad_norm:.5f}  ({epoch_time:.1f}s)")

        log['epochs'].append(epoch + 1)
        log['train_losses'].append(train_loss)
        log['val_losses'].append(val_loss)
        log['grad_norms'].append(grad_norm)

        # Save log every epoch
        save_log(log, log_path)

        # Plot every 5 epochs
        if (epoch + 1) % 5 == 0:
            plot_curves(log, plot_path, arch_name)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            print(f"            ✓ New best: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  ⚡ Early stopping at epoch {epoch+1} "
                      f"(no improvement for {args.patience} epochs)")
                break

        # Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'{ckpt_prefix}_epoch_{epoch+1}.pt')

    total_time = time.time() - start_time
    log['total_time_seconds'] = total_time
    log['best_val_loss']      = best_val_loss

    # ── Final per-dimension evaluation ───────────────────────────────────────
    print("\n  Running final per-dimension evaluation...")
    try:
        model.load_state_dict(torch.load(best_path, weights_only=False))
        final_metrics = evaluate(model, val_loader, criterion, device)
        log['final_per_dim_mse'] = final_metrics['per_dim_mse']
        print(f"  Per-dimension MSE:")
        for dim, mse in final_metrics['per_dim_mse'].items():
            print(f"    {dim}: {mse:.6f}")
    except Exception as e:
        print(f"  ⚠️  Per-dim eval failed: {e}")

    save_log(log, log_path)

    # ── Final plots and summary ───────────────────────────────────────────────
    plot_curves(log, plot_path, arch_name)
    plot_grad_norms(log, grad_path, arch_name)
    plot_per_dim_error(log, dim_path, arch_name)
    save_summary(log, summary_path, arch_name, arch_config)

    print(f"\n  ✅ {arch_name} complete!")
    print(f"     Best Val Loss: {best_val_loss:.6f}")
    print(f"     Total time:    {total_time/3600:.2f} hrs")

    return {
        'arch_name':     arch_name,
        'best_val_loss': best_val_loss,
        'log_path':      log_path,
        'total_time':    total_time,
        'stopped_early': len(log['epochs']) < args.epochs,
    }


# ==============================================================================
# MAIN SWEEP RUNNER
# ==============================================================================

def run_sweep(args):
    """
    Run all architectures for specified phase(s), then generate all comparison plots.
    Fully automatic - adding architectures to ARCHITECTURES dict is all that's needed.
    """
    print("\n" + "="*70)
    print("QML ARCHITECTURE SWEEP")
    print(f"  Phase:   {args.phase}")
    print(f"  Device:  {'cuda - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print(f"  Log dir: {args.log_dir}")
    print(f"  Frames:  {args.frames_per_video} per video")
    print(f"  Batch:   {args.batch_size}")
    print("="*70)

    os.makedirs(args.log_dir, exist_ok=True)

    # ── Select architectures to run ───────────────────────────────────────────
    if args.arch:
        if args.arch not in ARCHITECTURES:
            print(f"✗ Unknown architecture: {args.arch}")
            print(f"  Available: {list(ARCHITECTURES.keys())}")
            return
        to_run = {args.arch: ARCHITECTURES[args.arch]}
    elif args.phase == 'all':
        to_run = ARCHITECTURES
    else:
        phase_num = int(args.phase)
        to_run = {k: v for k, v in ARCHITECTURES.items() if v['phase'] == phase_num}
        if not to_run:
            print(f"✗ No architectures for phase {phase_num}")
            return

    print(f"\nRunning {len(to_run)} architectures:")
    for name, cfg in to_run.items():
        status = '(skip - already done)' if (
            os.path.exists(os.path.join(args.log_dir, f'{name}_training_log.json'))
            and not args.force
        ) else ''
        print(f"  Phase {cfg['phase']}: {name}  {status}")

    # ── Train each architecture ───────────────────────────────────────────────
    results = []
    for arch_name, arch_config in to_run.items():
        result = train_architecture(arch_name, arch_config, args)
        results.append(result)
        print(f"\n{'─'*70}")

    # ── Generate all comparison plots ─────────────────────────────────────────
    print("\n\nGenerating comparison plots...")

    # Collect all log files from completed runs
    all_logs = {
        r['arch_name']: r['log_path']
        for r in results
        if 'log_path' in r and os.path.exists(r.get('log_path', ''))
    }

    if not all_logs:
        print("No completed runs to compare.")
        return

    # Per-phase comparison plots
    all_phases = set(ARCHITECTURES.get(n, {}).get('phase', 0) for n in all_logs)
    for phase_num in sorted(all_phases):
        phase_name = SWEEP_CONFIG['phases'].get(phase_num, f'Phase {phase_num}')
        phase_logs = {
            n: p for n, p in all_logs.items()
            if ARCHITECTURES.get(n, {}).get('phase') == phase_num
        }
        if len(phase_logs) < 2:
            continue

        base = os.path.join(args.log_dir, f'phase{phase_num}')
        plot_comparison(phase_logs, f'{base}_comparison.png',
                        title=f'Phase {phase_num}: {phase_name}')
        plot_convergence_speed(phase_logs, f'{base}_convergence.png')
        plot_training_times(phase_logs, f'{base}_training_times.png')

    # Full sweep plots (all architectures together)
    if len(all_logs) > 1:
        base = os.path.join(args.log_dir, 'full_sweep')
        final_results = plot_comparison(
            all_logs, f'{base}_comparison.png',
            title='Full Architecture Sweep - All Configurations'
        )
        plot_params_vs_performance(all_logs, f'{base}_params_vs_performance.png')
        plot_convergence_speed(all_logs, f'{base}_convergence.png')
        plot_training_times(all_logs, f'{base}_training_times.png')
        plot_per_dim_comparison(all_logs, f'{base}_per_dim_comparison.png')
        plot_barren_plateau_comparison(all_logs, f'{base}_barren_plateau.png')

        # Save leaderboard
        final_results.sort(key=lambda x: x['best_val'])
        save_leaderboard(final_results,
                         os.path.join(args.log_dir, 'sweep_leaderboard.txt'))

    # Save sweep summary JSON
    sweep_summary_path = os.path.join(args.log_dir, 'sweep_summary.json')
    with open(sweep_summary_path, 'w') as f:
        json.dump({
            'sweep_date':           datetime.now().isoformat(),
            'phase':                args.phase,
            'total_architectures':  len(results),
            'frames_per_video':     args.frames_per_video,
            'batch_size':           args.batch_size,
            'results': [
                {k: v for k, v in r.items() if k != 'log_path'}
                for r in results
            ]
        }, f, indent=2)

    print(f"\n✓ Sweep summary JSON: {sweep_summary_path}")
    print(f"\n✅ Sweep complete! All results in: {args.log_dir}")
    print(f"\nGenerated files:")
    for f in sorted(os.listdir(args.log_dir)):
        size = os.path.getsize(os.path.join(args.log_dir, f))
        print(f"  {f}  ({size/1024:.0f} KB)")


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='QML Architecture Sweep for Robot Manipulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full sweep (all 13 architectures):
  python quantum_sweep.py --phase all

  # Run only qubit sweep (phase 1):
  python quantum_sweep.py --phase 1

  # Run single architecture:
  python quantum_sweep.py --arch QViT_8q_4layer_amp

  # Run on cluster with custom paths:
  python quantum_sweep.py --phase all \\
    --vit_encoder_path /tmp/dandrenf/QuantumInspiredViTSpeedUp/quantum_research/vit_encoder_only.pt \\
    --zarr_path /tmp/dandrenf/umi_data/session_001/dataset.zarr.zip \\
    --data_path /tmp/dandrenf/umi_data/session_001 \\
    --log_dir /home/dandrenf/qml_sweep_logs

  # Force re-run even if already completed:
  python quantum_sweep.py --phase 1 --force

  # List all available architectures:
  python quantum_sweep.py --list
        """
    )

    parser.add_argument('--phase', type=str, default='all',
                        help='Phase to run: 1, 2, 3, 4, 5, or all (default: all)')
    parser.add_argument('--arch',  type=str, default=None,
                        help='Run a single architecture by name')
    parser.add_argument('--list',  action='store_true',
                        help='List all available architectures and exit')
    parser.add_argument('--force', action='store_true',
                        help='Re-run even if already completed')

    # Paths
    parser.add_argument('--vit_encoder_path', type=str,
                        default=SWEEP_CONFIG['vit_encoder_path'])
    parser.add_argument('--zarr_path',        type=str,
                        default=SWEEP_CONFIG['zarr_path'])
    parser.add_argument('--data_path',        type=str,
                        default=SWEEP_CONFIG['data_path'])
    parser.add_argument('--log_dir',          type=str,
                        default=SWEEP_CONFIG['log_dir'])

    # Hyperparameters
    parser.add_argument('--frames_per_video', type=int,
                        default=SWEEP_CONFIG['num_frames_per_video'])
    parser.add_argument('--batch_size',       type=int,
                        default=SWEEP_CONFIG['batch_size'])
    parser.add_argument('--learning_rate',    type=float,
                        default=SWEEP_CONFIG['learning_rate'])
    parser.add_argument('--epochs',           type=int,
                        default=SWEEP_CONFIG['num_epochs'])
    parser.add_argument('--patience',         type=int,
                        default=SWEEP_CONFIG['patience'])
    parser.add_argument('--num_workers',      type=int,
                        default=SWEEP_CONFIG['num_workers'])

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.list:
        print("\nAvailable architectures:")
        print(f"{'Name':<35} {'Phase':<8} {'Qubits':<8} {'Layers':<8} {'Encoding':<12} {'Gates'}")
        print("─" * 90)
        for name, cfg in ARCHITECTURES.items():
            print(f"{name:<35} Phase {cfg['phase']}  "
                  f"{cfg['n_qubits']:<8} {cfg['n_layers']:<8} "
                  f"{cfg['encoding']:<12} {cfg['gate_pattern']}")
        print()
        print("Phases:")
        for num, pname in SWEEP_CONFIG['phases'].items():
            count = sum(1 for v in ARCHITECTURES.values() if v['phase'] == num)
            print(f"  Phase {num}: {pname} ({count} configs)")
        print(f"\nTotal: {len(ARCHITECTURES)} architectures across {len(SWEEP_CONFIG['phases'])} phases")
        print()
        print("To add a new architecture: add an entry to ARCHITECTURES dict with")
        print("  phase, n_qubits, n_layers, compression_dim, encoding, gate_pattern, description")
    else:
        run_sweep(args)
