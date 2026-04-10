#!/usr/bin/env python3
"""
Low-Rank Attention for ViT-B/16 - Fixed Implementation

The key insight: we want (Q @ E) @ (K @ F)^T to equal Q @ K^T when E = F = I.
So we initialize E = F = I[:, :r] (first r columns of identity), which means
at initialization the low-rank attention computes Q[:, :r] @ K[:, :r]^T.

This is a proper low-rank approximation of Q @ K^T when Q and K are already
empirically low-rank (which our SVD analysis confirmed).

Usage:
    python lowrank_attention_v2.py --mode sweep --checkpoint /tmp/vit_encoder_only.pt
"""

import os
import math
import argparse
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

try:
    import timm
except ImportError:
    raise ImportError("Please install timm: pip install timm")


@dataclass
class LowRankConfig:
    rank: int = 8
    layers_to_compress: List[int] = None
    num_heads: int = 12
    head_dim: int = 64
    
    def __post_init__(self):
        if self.layers_to_compress is None:
            self.layers_to_compress = [3, 4, 5, 6, 7, 8]  # Layers 4-9 (0-indexed)


class LowRankAttention(nn.Module):
    """
    Low-rank attention that compresses Q and K to rank r before computing attention.
    
    Instead of: attn = (Q @ K^T) / sqrt(d)           # O(N^2 * d)
    We compute: attn = (Q @ P) @ (K @ P)^T / sqrt(r) # O(N^2 * r) where r << d
    
    P is a learned projection matrix of shape [head_dim, rank].
    Using the SAME projection P for both Q and K ensures that at initialization
    with P = I[:, :r], we get a valid low-rank approximation.
    """
    
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        rank: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rank = rank
        self.scale = rank ** -0.5  # Scale by 1/sqrt(r) since we're in r-dimensional space
        
        # Standard QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # Single shared projection matrix per head: [num_heads, head_dim, rank]
        # Initialize to first r columns of identity (selects first r dimensions)
        self.P = nn.Parameter(torch.zeros(num_heads, self.head_dim, rank))
        for h in range(num_heads):
            # Initialize as truncated identity: just select first r dimensions
            self.P.data[h, :rank, :] = torch.eye(rank)
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        Q, K, V = qkv[0], qkv[1], qkv[2]  # Each: [B, num_heads, N, head_dim]
        
        # Project Q and K to low-rank space using shared projection P
        # Q_lr = Q @ P: [B, num_heads, N, head_dim] @ [num_heads, head_dim, rank] -> [B, num_heads, N, rank]
        Q_lr = torch.einsum('bhnd,hdr->bhnr', Q, self.P)
        K_lr = torch.einsum('bhnd,hdr->bhnr', K, self.P)
        
        # Compute attention in low-rank space
        attn = torch.einsum('bhnr,bhmr->bhnm', Q_lr, K_lr) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply to values (V stays full rank)
        out = torch.einsum('bhnm,bhmd->bhnd', attn, V)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out


class OriginalAttentionWrapper(nn.Module):
    """Wrapper to make timm attention interface consistent."""
    
    def __init__(self, attn_module):
        super().__init__()
        self.attn = attn_module
    
    def forward(self, x):
        return self.attn(x)


def load_vit_and_images(checkpoint_path: str, video_paths: List[str], device: str, num_frames: int = 100):
    """Load ViT model and sample images."""
    import cv2
    from torchvision import transforms
    from PIL import Image
    
    print(f"Loading ViT from: {checkpoint_path}")
    vit_state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'state_dicts' in vit_state:
        state = vit_state['state_dicts']['model']
        vit_prefix = 'obs_encoder.key_model_map.camera0_rgb.'
        vit_state = {k.replace(vit_prefix, ''): v for k, v in state.items() if k.startswith(vit_prefix)}
    
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
    model_state = model.state_dict()
    for name in model_state:
        if name in vit_state and vit_state[name].shape == model_state[name].shape:
            model_state[name] = vit_state[name]
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()
    
    # Load images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    frames_per_video = num_frames // len(video_paths)
    images = []
    
    for video_path in video_paths:
        print(f"  Loading frames from: {video_path}")
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            print(f"    Warning: Could not read video {video_path}")
            continue
        indices = np.linspace(0, total - 1, frames_per_video, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                images.append(transform(img))
        cap.release()
    
    if len(images) == 0:
        raise ValueError("No images loaded!")
    
    images = torch.stack(images)
    print(f"  Loaded {len(images)} frames")
    
    return model, images


def compute_optimal_projection(
    model: nn.Module,
    images: torch.Tensor,
    config: LowRankConfig,
    device: str = 'cuda',
    batch_size: int = 16,
) -> Dict[int, torch.Tensor]:
    """
    Compute optimal projection P for each layer using SVD of Q and K.
    
    We want P such that Q @ P captures most of the variance in Q,
    and similarly for K. Since we use the same P for both, we compute
    SVD of the concatenation [Q; K].
    """
    print("\nComputing optimal projections via SVD...")
    
    captured = {layer: {'Q': [], 'K': []} for layer in config.layers_to_compress}
    hooks = []
    
    def make_hook(layer_idx, attn_module):
        def hook(module, input, output):
            x = output
            B, N, C = x.shape
            qkv = attn_module.qkv(x)
            qkv = qkv.reshape(B, N, 3, config.num_heads, config.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            captured[layer_idx]['Q'].append(qkv[0].detach().cpu())
            captured[layer_idx]['K'].append(qkv[1].detach().cpu())
        return hook
    
    for layer_idx in config.layers_to_compress:
        h = model.blocks[layer_idx].norm1.register_forward_hook(
            make_hook(layer_idx, model.blocks[layer_idx].attn)
        )
        hooks.append(h)
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Collecting Q/K"):
            batch = images[i:i+batch_size].to(device)
            _ = model(batch)
    
    for h in hooks:
        h.remove()
    
    # Compute optimal P for each layer
    projections = {}
    
    for layer_idx in tqdm(config.layers_to_compress, desc="Computing SVD"):
        Q_all = torch.cat(captured[layer_idx]['Q'], dim=0)  # [samples, heads, N, head_dim]
        K_all = torch.cat(captured[layer_idx]['K'], dim=0)
        
        P_layer = torch.zeros(config.num_heads, config.head_dim, config.rank)
        
        for head_idx in range(config.num_heads):
            # Concatenate Q and K for this head
            Q_h = Q_all[:, head_idx, :, :].reshape(-1, config.head_dim)  # [samples*N, head_dim]
            K_h = K_all[:, head_idx, :, :].reshape(-1, config.head_dim)
            QK = torch.cat([Q_h, K_h], dim=0)  # [2*samples*N, head_dim]
            
            # Use covariance matrix for efficiency: SVD of QK^T @ QK is much faster
            # This gives us the right singular vectors we need
            cov = (QK.T @ QK) / QK.shape[0]  # [head_dim, head_dim] - much smaller!
            
            # Eigendecomposition of covariance = right singular vectors of original
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            
            # eigh returns in ascending order, we want descending (largest first)
            idx = torch.argsort(eigenvalues, descending=True)
            
            # P = top-r eigenvectors
            P_layer[head_idx] = eigenvectors[:, idx[:config.rank]]  # [head_dim, rank]
        
        projections[layer_idx] = P_layer
    
    return projections


def convert_to_lowrank(
    model: nn.Module,
    config: LowRankConfig,
    projections: Optional[Dict[int, torch.Tensor]] = None,
) -> nn.Module:
    """Replace attention in specified layers with low-rank attention."""
    print(f"\nConverting layers {[l+1 for l in config.layers_to_compress]} to low-rank (r={config.rank})...")
    
    for layer_idx in config.layers_to_compress:
        old_attn = model.blocks[layer_idx].attn
        
        new_attn = LowRankAttention(
            dim=768,
            num_heads=12,
            rank=config.rank,
            qkv_bias=old_attn.qkv.bias is not None,
        )
        
        # Copy QKV and projection weights
        new_attn.qkv.weight.data.copy_(old_attn.qkv.weight.data)
        if old_attn.qkv.bias is not None:
            new_attn.qkv.bias.data.copy_(old_attn.qkv.bias.data)
        new_attn.proj.weight.data.copy_(old_attn.proj.weight.data)
        new_attn.proj.bias.data.copy_(old_attn.proj.bias.data)
        
        # Set optimal projection if provided
        if projections and layer_idx in projections:
            new_attn.P.data.copy_(projections[layer_idx])
        
        model.blocks[layer_idx].attn = new_attn
        
        speedup = config.head_dim / config.rank
        print(f"  Layer {layer_idx + 1}: {config.head_dim}d -> {config.rank}d ({speedup:.1f}x speedup)")
    
    return model


def validate_accuracy(
    original_model: nn.Module,
    lowrank_model: nn.Module,
    images: torch.Tensor,
    device: str = 'cuda',
    batch_size: int = 16,
) -> Dict[str, float]:
    """Compare outputs of original and low-rank models."""
    print("\nValidating accuracy...")
    
    original_model.eval()
    lowrank_model.eval()
    
    orig_outputs = []
    lr_outputs = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Validating"):
            batch = images[i:i+batch_size].to(device)
            orig_outputs.append(original_model(batch).cpu())
            lr_outputs.append(lowrank_model(batch).cpu())
    
    orig = torch.cat(orig_outputs, dim=0)
    lr = torch.cat(lr_outputs, dim=0)
    
    diff = orig - lr
    frob_error = torch.norm(diff) / torch.norm(orig)
    cosine = F.cosine_similarity(orig.reshape(-1).unsqueeze(0), lr.reshape(-1).unsqueeze(0)).item()
    mse = F.mse_loss(lr, orig).item()
    
    print(f"  Frobenius error: {frob_error.item():.4f} (target: < 0.02)")
    print(f"  Cosine similarity: {cosine:.6f}")
    print(f"  MSE: {mse:.6f}")
    
    status = "✓ PASSED" if frob_error.item() < 0.02 else "✗ FAILED"
    print(f"  {status}")
    
    return {
        'frobenius_error': frob_error.item(),
        'cosine_similarity': cosine,
        'mse': mse,
    }


def rank_sweep(
    model: nn.Module,
    images: torch.Tensor,
    config: LowRankConfig,
    device: str = 'cuda',
    ranks: List[int] = [4, 6, 8, 12, 16, 24, 32, 48, 64],
) -> Dict[int, Dict[str, float]]:
    """Test multiple ranks and report accuracy vs speedup."""
    print("\n" + "="*70)
    print("RANK SWEEP ANALYSIS")
    print("="*70)
    
    # Keep a fresh copy of original
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    results = {}
    
    for rank in ranks:
        print(f"\n--- Testing rank r={rank} ---")
        
        # Restore original model
        model.load_state_dict(original_state)
        model = model.to(device)
        
        # Create fresh copy for low-rank conversion
        import copy
        original_model = copy.deepcopy(model)
        original_model.eval()
        
        # Configure
        test_config = LowRankConfig(rank=rank, layers_to_compress=config.layers_to_compress)
        
        # Compute optimal projections
        projections = compute_optimal_projection(model, images, test_config, device)
        
        # Restore again (hooks may have affected state)
        model.load_state_dict(original_state)
        model = model.to(device)
        
        # Convert to low-rank
        lowrank_model = copy.deepcopy(model)
        lowrank_model = convert_to_lowrank(lowrank_model, test_config, projections)
        lowrank_model = lowrank_model.to(device)
        
        # Validate
        metrics = validate_accuracy(original_model, lowrank_model, images, device)
        metrics['speedup'] = config.head_dim / rank
        results[rank] = metrics
    
    # Summary table
    print("\n" + "="*70)
    print("RANK SWEEP SUMMARY")
    print("="*70)
    print(f"{'Rank':<8} {'Frob Error':<14} {'Cosine Sim':<14} {'Speedup':<10} {'Pass?':<8}")
    print("-"*54)
    for rank in sorted(results.keys()):
        m = results[rank]
        passed = "✓" if m['frobenius_error'] < 0.02 else "✗"
        print(f"{rank:<8} {m['frobenius_error']:<14.4f} {m['cosine_similarity']:<14.6f} {m['speedup']:<10.1f}x {passed:<8}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/tmp/vit_encoder_only.pt')
    parser.add_argument('--videos', type=str, nargs='+', default=[
        # '/home/wadeab/universal_manipulation_interface/example_demo_session/demos/demo_C3441328164125_2024.01.10_11.03.35.292183/raw_video.mp4',
        # '/home/wadeab/universal_manipulation_interface/example_demo_session/demos/demo_C3441328164125_2024.01.10_11.03.03.860783/raw_video.mp4',
        '/home/wadeab/universal_manipulation_interface/data/session_001/GX010453.MP4',
        '/home/wadeab/universal_manipulation_interface/data/session_001/GX010454.MP4',
    ])

    parser.add_argument('--mode', type=str, choices=['sweep', 'single'], default='sweep')
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model, images = load_vit_and_images(args.checkpoint, args.videos, device)
    
    config = LowRankConfig(
        rank=args.rank,
        layers_to_compress=[3, 4, 5, 6, 7, 8],  # Layers 4-9
    )
    
    if args.mode == 'sweep':
        results = rank_sweep(model, images, config, device)
    else:
        import copy
        original_model = copy.deepcopy(model)
        projections = compute_optimal_projection(model, images, config, device)
        lowrank_model = copy.deepcopy(model)
        lowrank_model = convert_to_lowrank(lowrank_model, config, projections)
        lowrank_model = lowrank_model.to(device)
        validate_accuracy(original_model, lowrank_model, images, device)


if __name__ == '__main__':
    main()