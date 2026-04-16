#!/usr/bin/env python3
"""
Low-Rank Diagnostic for ViT-B/16 Attention Matrices
Run this locally on your machine, then upload the output folder.

Usage:
    python run_local_analysis.py

Outputs saved to: ./lowrank_results/
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - EDIT THESE PATHS
# ============================================================================

CHECKPOINT_PATH = r'C:\Users\neald\Desktop\QuantumInspiredViTSpeedUp\quantum_research\vit_encoder_only.pt'
USE_EXTRACTED = True  # Add this line
# VIDEO_PATHS = [
#        '/home/wadeab/universal_manipulation_interface/example_demo_session/demos/demo_C3441328164125_2024.01.10_11.03.35.292183/raw_video.mp4',
#        '/home/wadeab/universal_manipulation_interface/example_demo_session/demos/demo_C3441328164125_2024.01.10_11.03.03.860783/raw_video.mp4',
#        '/home/wadeab/universal_manipulation_interface/example_demo_session/demos/demo_C3441328164125_2024.01.10_11.02.36.633583/raw_video.mp4',
#        '/home/wadeab/universal_manipulation_interface/example_demo_session/demos/demo_C3441328164125_2024.01.10_10.57.34.882133/raw_video.mp4',
# ]

VIDEO_PATHS = [
    r'C:\Users\neald\Desktop\QuantumInspiredViTSpeedUp\quantum_research\data_for_quantum_research2\GX010460.MP4',
    r'C:\Users\neald\Desktop\QuantumInspiredViTSpeedUp\quantum_research\data_for_quantum_research2\GX010461.MP4',
    r'C:\Users\neald\Desktop\QuantumInspiredViTSpeedUp\quantum_research\data_for_quantum_research2\GX010462.MP4',
    r'C:\Users\neald\Desktop\QuantumInspiredViTSpeedUp\quantum_research\data_for_quantum_research2\GX010464.MP4',
    r'C:\Users\neald\Desktop\QuantumInspiredViTSpeedUp\quantum_research\data_for_quantum_research2\GX010465.MP4',
]

IMAGE_DIR = None  # Set this if you have a folder of images instead of videos
OUTPUT_DIR = './lowrank_results500frames'
NUM_FRAMES = 500
BATCH_SIZE = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# VIT CONFIG
# ============================================================================

VIT_CONFIG = {
    "num_layers": 12,
    "num_heads": 12,
    "head_dim": 64,
    "seq_len": 197,
    "hidden_dim": 768,
}

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ============================================================================
# LOAD MODEL
# ============================================================================

def load_vit_from_checkpoint(checkpoint_path: str, device: str):
    """Load ViT encoder from extracted weights or full checkpoint."""
    import timm
    
    print(f"Loading checkpoint: {checkpoint_path}")
    vit_state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # If it's the full checkpoint, extract ViT weights
    if 'state_dicts' in vit_state:
        state = vit_state['state_dicts']['model']
        vit_prefix = 'obs_encoder.key_model_map.camera0_rgb.'
        vit_state = {k.replace(vit_prefix, ''): v for k, v in state.items() if k.startswith(vit_prefix)}
    
    print(f"  Found {len(vit_state)} ViT keys")
    
    # Create ViT-B/16 model
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
    
    # Load weights
    model_state = model.state_dict()
    loaded = 0
    for name in model_state:
        if name in vit_state and vit_state[name].shape == model_state[name].shape:
            model_state[name] = vit_state[name]
            loaded += 1
    
    model.load_state_dict(model_state)
    print(f"  Loaded {loaded}/{len(model_state)} parameters")
    
    model = model.to(device)
    model.eval()
    return model

# ============================================================================
# ATTENTION HOOKS
# ============================================================================

class AttentionCaptureHook:
    def __init__(self, num_layers=12, num_heads=12):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.captured_qk = {}
        self.hooks = []
    
    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            x = input[0]
            B, N, C = x.shape
            qkv = module.qkv(x)
            qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            self.captured_qk[layer_idx] = {'Q': qkv[0].detach().cpu(), 'K': qkv[1].detach().cpu()}
        return hook
    
    def register(self, model):
        for idx, block in enumerate(model.blocks):
            # Use forward hook on norm1 (runs right before attention)
            hook = block.norm1.register_forward_hook(self._make_norm1_hook(idx, block.attn))
            self.hooks.append(hook)
        print(f"  Registered hooks on {len(self.hooks)} attention layers")
    
    def _make_norm1_hook(self, layer_idx, attn_module):
        def hook(module, input, output):
            # output is the normalized x that goes into attention
            x = output
            B, N, C = x.shape
            qkv = attn_module.qkv(x)
            qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            self.captured_qk[layer_idx] = {'Q': qkv[0].detach().cpu(), 'K': qkv[1].detach().cpu()}
        return hook
    
    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def clear(self):
        self.captured_qk = {}

# ============================================================================
# LOAD IMAGES
# ============================================================================

def load_frames_from_videos(video_paths: List[str], num_frames: int) -> torch.Tensor:
    """Extract frames from video files."""
    import cv2
    
    frames_per_video = num_frames // len(video_paths)
    images = []
    
    for video_path in video_paths:
        print(f"  Extracting from: {video_path}")
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total - 1, frames_per_video, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                images.append(IMAGE_TRANSFORM(img))
        cap.release()
    
    print(f"  Loaded {len(images)} frames")
    return torch.stack(images)

def load_frames_from_dir(image_dir: str, num_frames: int) -> torch.Tensor:
    """Load images from directory."""
    import glob
    
    extensions = ['*.png', '*.jpg', '*.jpeg']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))
    
    files = sorted(files)[:num_frames]
    print(f"  Found {len(files)} images")
    
    images = []
    for f in tqdm(files, desc="Loading"):
        img = Image.open(f).convert('RGB')
        images.append(IMAGE_TRANSFORM(img))
    
    return torch.stack(images)

# ============================================================================
# ANALYSIS
# ============================================================================

def compute_attention_matrices(model, hook, images, device, batch_size):
    """Run forward passes and collect QK^T matrices."""
    print("\nComputing attention matrices...")
    
    attention_matrices = {(l, h): [] for l in range(12) for h in range(12)}
    
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Batches"):
            batch = images[i:i+batch_size].to(device)
            hook.clear()
            _ = model(batch)
            
            for layer_idx in range(12):
                Q = hook.captured_qk[layer_idx]['Q']
                K = hook.captured_qk[layer_idx]['K']
                
                for head_idx in range(12):
                    q_h = Q[:, head_idx, :, :]
                    k_h = K[:, head_idx, :, :]
                    attn = torch.bmm(q_h, k_h.transpose(1, 2))
                    for j in range(attn.shape[0]):
                        attention_matrices[(layer_idx, head_idx)].append(attn[j])
    
    return attention_matrices

def compute_svd_analysis(attention_matrices):
    """Compute SVD and EVR curves."""
    print("\nRunning SVD analysis...")
    
    all_evr = []
    
    for layer_idx in tqdm(range(12), desc="Layers"):
        layer_evr = []
        for head_idx in range(12):
            matrices = attention_matrices[(layer_idx, head_idx)]
            head_evr = []
            for mat in matrices:
                sv = torch.linalg.svdvals(mat.float()).numpy()
                sv_sq = sv ** 2
                total = sv_sq.sum()
                evr = np.cumsum(sv_sq) / total if total > 0 else np.ones_like(sv_sq)
                head_evr.append(evr[:64])  # Limit to 64
            layer_evr.append(np.stack(head_evr))
        all_evr.append(np.stack(layer_evr))
    
    return np.stack(all_evr)  # [12, 12, num_samples, 64]

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(evr_curves, output_dir):
    """Generate plots and compute statistics."""
    os.makedirs(output_dir, exist_ok=True)
    
    mean_evr = evr_curves.mean(axis=2)  # [12, 12, 64]
    
    # Plot 1: Heatmap at rank 16
    evr_at_16 = mean_evr[:, :, 15] * 100
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(evr_at_16, cmap='RdYlGn', vmin=90, vmax=100)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('EVR at r=16 (%)', fontsize=12)
    
    for i in range(12):
        for j in range(12):
            color = 'white' if evr_at_16[i, j] < 95 else 'black'
            ax.text(j, i, f'{evr_at_16[i, j]:.1f}', ha='center', va='center', fontsize=9, color=color, fontweight='bold')
    
    ax.set_xticks(range(12))
    ax.set_yticks(range(12))
    ax.set_xticklabels([f'H{i+1}' for i in range(12)])
    ax.set_yticklabels([f'L{i+1}' for i in range(12)])
    ax.set_xlabel('Attention Head', fontsize=12)
    ax.set_ylabel('Transformer Layer', fontsize=12)
    ax.set_title('ViT-B/16 Attention Matrix Low-Rank Structure\n(EVR at Rank 16)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evr_heatmap_rank16.png'), dpi=150)
    plt.close()
    
    # Plot 2: EVR curves
    fig, ax = plt.subplots(figsize=(12, 8))
    ranks = np.arange(1, 65)
    
    colors = plt.cm.viridis(np.linspace(0, 1, 12))
    for layer_idx in range(12):
        layer_mean = mean_evr[layer_idx].mean(axis=0)
        ax.plot(ranks, layer_mean, color=colors[layer_idx], alpha=0.3, linewidth=1)
    
    overall_mean = mean_evr.mean(axis=(0, 1))
    ax.plot(ranks, overall_mean, color='black', linewidth=3, label='Mean (all layers)')
    ax.axhline(y=0.99, color='red', linestyle='--', linewidth=1.5, label='99% threshold')
    
    cross_99 = np.where(overall_mean >= 0.99)[0]
    if len(cross_99) > 0:
        r99 = cross_99[0] + 1
        ax.axvline(x=r99, color='red', linestyle=':', alpha=0.7)
        ax.annotate(f'r={r99}', xy=(r99, 0.99), xytext=(r99 + 3, 0.96), fontsize=11, color='red',
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
    
    ax.set_xlabel('Rank (r)', fontsize=12)
    ax.set_ylabel('Cumulative EVR', fontsize=12)
    ax.set_title('Low-Rank Approximation Quality of ViT-B/16 Attention Matrices', fontsize=14, fontweight='bold')
    ax.set_xlim(1, 64)
    ax.set_ylim(0.5, 1.01)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evr_curves.png'), dpi=150)
    plt.close()
    
    # Plot 3: Layer groups
    fig, ax = plt.subplots(figsize=(10, 6))
    early = mean_evr[0:3].mean(axis=(0, 1))
    middle = mean_evr[3:9].mean(axis=(0, 1))
    late = mean_evr[9:12].mean(axis=(0, 1))
    
    ax.plot(ranks, early, linewidth=2.5, label='Early (L1-3)', color='#2ecc71')
    ax.plot(ranks, middle, linewidth=2.5, label='Middle (L4-9)', color='#3498db')
    ax.plot(ranks, late, linewidth=2.5, label='Late (L10-12)', color='#e74c3c')
    ax.axhline(y=0.99, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('Rank (r)', fontsize=12)
    ax.set_ylabel('Cumulative EVR', fontsize=12)
    ax.set_title('Low-Rank Structure by Layer Group', fontsize=14, fontweight='bold')
    ax.set_xlim(1, 64)
    ax.set_ylim(0.7, 1.01)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evr_by_layer_group.png'), dpi=150)
    plt.close()
    
    # Compute stats
    stats = {
        'mean_evr_r8': mean_evr[:, :, 7].mean() * 100,
        'mean_evr_r16': mean_evr[:, :, 15].mean() * 100,
        'mean_evr_r24': mean_evr[:, :, 23].mean() * 100,
        'mean_evr_r32': mean_evr[:, :, 31].mean() * 100,
        'rank_99': int(cross_99[0] + 1) if len(cross_99) > 0 else 64,
        'worst_layer': int(np.unravel_index(evr_at_16.argmin(), evr_at_16.shape)[0] + 1),
        'worst_head': int(np.unravel_index(evr_at_16.argmin(), evr_at_16.shape)[1] + 1),
        'worst_evr': float(evr_at_16.min()),
        'best_layer': int(np.unravel_index(evr_at_16.argmax(), evr_at_16.shape)[0] + 1),
        'best_head': int(np.unravel_index(evr_at_16.argmax(), evr_at_16.shape)[1] + 1),
        'best_evr': float(evr_at_16.max()),
        'early_evr_16': float(mean_evr[0:3, :, 15].mean() * 100),
        'middle_evr_16': float(mean_evr[3:9, :, 15].mean() * 100),
        'late_evr_16': float(mean_evr[9:12, :, 15].mean() * 100),
    }
    
    # Write report
    report = f"""
================================================================================
          LOW-RANK DIAGNOSTIC REPORT: ViT-B/16 Attention Matrices
================================================================================

MEAN EXPLAINED VARIANCE RATIO (EVR) ACROSS ALL 144 MATRICES
--------------------------------------------------------------------------------
    Rank  8:   {stats['mean_evr_r8']:.2f}%
    Rank 16:   {stats['mean_evr_r16']:.2f}%
    Rank 24:   {stats['mean_evr_r24']:.2f}%
    Rank 32:   {stats['mean_evr_r32']:.2f}%

    Optimal rank (>=99% EVR):  r = {stats['rank_99']}

BEST & WORST CASES AT RANK 16
--------------------------------------------------------------------------------
    Worst:  Layer {stats['worst_layer']}, Head {stats['worst_head']}  ->  {stats['worst_evr']:.2f}% EVR
    Best:   Layer {stats['best_layer']}, Head {stats['best_head']}  ->  {stats['best_evr']:.2f}% EVR

LAYER GROUP ANALYSIS (EVR at r=16)
--------------------------------------------------------------------------------
    Early layers  (L1-3):   {stats['early_evr_16']:.2f}%
    Middle layers (L4-9):   {stats['middle_evr_16']:.2f}%
    Late layers   (L10-12): {stats['late_evr_16']:.2f}%

    Middle > Early?  {'YES' if stats['middle_evr_16'] > stats['early_evr_16'] else 'NO'}
    Middle > Late?   {'YES' if stats['middle_evr_16'] > stats['late_evr_16'] else 'NO'}

================================================================================
"""
    
    with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
        f.write(report)
    
    print(report)
    return stats

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("ViT-B/16 Low-Rank Attention Diagnostic")
    print("="*80)
    
    # Load model
    print(f"\nUsing device: {DEVICE}")
    model = load_vit_from_checkpoint(CHECKPOINT_PATH, DEVICE)
    
    # Load images
    print("\nLoading observation frames...")
    if IMAGE_DIR:
        images = load_frames_from_dir(IMAGE_DIR, NUM_FRAMES)
    else:
        images = load_frames_from_videos(VIDEO_PATHS, NUM_FRAMES)
    
    # Register hooks
    hook = AttentionCaptureHook()
    hook.register(model)
    
    try:
        # Compute attention matrices
        attention_matrices = compute_attention_matrices(model, hook, images, DEVICE, BATCH_SIZE)
        
        # Run SVD
        evr_curves = compute_svd_analysis(attention_matrices)
        
        # Plot and report
        print("\nGenerating plots...")
        stats = plot_results(evr_curves, OUTPUT_DIR)
        
        # Save raw data
        np.savez(os.path.join(OUTPUT_DIR, 'raw_results.npz'), evr_curves=evr_curves)
        
    finally:
        hook.remove()
    
    print(f"\nDone! Results saved to: {OUTPUT_DIR}/")
    print("Upload this folder to Claude for interpretation.")

if __name__ == '__main__':
    main()