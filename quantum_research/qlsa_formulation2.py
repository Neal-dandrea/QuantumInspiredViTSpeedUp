#!/usr/bin/env python3
"""
QLSA Formulation for ViT Attention (Updated)
=============================================
This script implements Step 1 from the quantum approach:
1. Extract attention operation matrix A_i from each layer of trained ViT
2. Construct block matrix M for the 12-layer iterative system
3. Analyze sparsity and condition number (QLSA requirements)

The iterative system:
    X_i = A_i · X_{i-1} + C  for i = 1, ..., 12

Reformulated as linear system:
    M · X = b  where X = M^{-1} · b

Block structure of M (with per-layer A_i):
    ┌ I     0     0    ...   0     0  ┐
    │-A_1   I     0    ...   0     0  │
    │ 0    -A_2   I    ...   0     0  │
    │ ⋮                  ⋱         ⋮  │
    └ 0     0     0    ... -A_12   I  ┘
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
import timm
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

# ============================================================================
# CONFIGURATION - Update these paths for your system
# ============================================================================
# Windows paths
CHECKPOINT_PATH = r'C:\Users\neald\Desktop\QuantumInspiredViTSpeedUp\quantum_research\vit_encoder_only.pt'
VIDEO_PATHS = [
    r'C:\Users\neald\Desktop\QuantumInspiredViTSpeedUp\quantum_research\data_for_quantum_research2\GX010460.MP4',
    r'C:\Users\neald\Desktop\QuantumInspiredViTSpeedUp\quantum_research\data_for_quantum_research2\GX010461.MP4',
]

# Linux paths (uncomment if running on Linux)
# CHECKPOINT_PATH = '/tmp/vit_encoder_only.pt'
# VIDEO_PATHS = [
#     '/home/wadeab/universal_manipulation_interface/data/session_001/GX010460.MP4',
#     '/home/wadeab/universal_manipulation_interface/data/session_001/GX010461.MP4',
# ]

OUTPUT_DIR = './qlsa_results'
NUM_FRAMES = 50  # Number of frames to analyze
NUM_LAYERS = 12  # ViT-B has 12 transformer layers
NUM_HEADS = 12   # ViT-B has 12 attention heads
HEAD_DIM = 64    # Each head has dimension 64
SEQ_LEN = 197    # 196 patches + 1 CLS token


# ============================================================================
# PART 1: Load ViT and Extract Attention Matrices
# ============================================================================

def load_vit_model(checkpoint_path, device='cpu'):
    """Load ViT-B/16 model with pretrained weights."""
    print(f"Loading ViT from: {checkpoint_path}")
    
    # Create ViT-B/16 model
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    
    # Load checkpoint
    vit_state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load weights
    missing, unexpected = model.load_state_dict(vit_state, strict=False)
    print(f"  Loaded {len(vit_state) - len(unexpected)}/{len(vit_state)} parameters")
    
    model.eval()
    model.to(device)
    return model


def load_frames(video_paths, num_frames=50):
    """Load frames from video files."""
    frames = []
    frames_per_video = max(1, num_frames // len(video_paths))
    
    for video_path in video_paths:
        print(f"  Loading from: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"    Warning: Could not open {video_path}")
            continue
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, frames_per_video, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB and resize
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
        
        cap.release()
    
    if not frames:
        raise ValueError("No frames loaded!")
    
    # Convert to tensor [N, C, H, W]
    frames = np.stack(frames)
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    
    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    frames = (frames - mean) / std
    
    print(f"  Loaded {len(frames)} frames")
    return frames


def extract_attention_matrices(model, images, device='cpu'):
    """
    Extract attention weight matrices from all layers.
    
    Returns:
        attention_matrices: dict[layer_idx][head_idx] -> list of [197, 197] matrices
    """
    attention_matrices = {l: {h: [] for h in range(NUM_HEADS)} for l in range(NUM_LAYERS)}
    
    # Register hooks to capture attention weights
    attention_weights = {}
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            # Get Q, K from the qkv projection
            B, N, C = input[0].shape
            qkv = module.qkv(input[0])
            qkv = qkv.reshape(B, N, 3, NUM_HEADS, HEAD_DIM).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            # Compute attention weights: softmax(Q @ K^T / sqrt(d))
            scale = HEAD_DIM ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)  # [B, num_heads, N, N]
            
            attention_weights[layer_idx] = attn.detach().cpu()
        return hook
    
    # Register hooks on attention modules
    hooks = []
    for layer_idx, block in enumerate(model.blocks):
        hook = block.attn.register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)
    
    # Process images in batches
    batch_size = 8
    print("Extracting attention matrices...")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Processing batches"):
            batch = images[i:i+batch_size].to(device)
            attention_weights.clear()
            
            _ = model(batch)
            
            # Store attention matrices
            for layer_idx in range(NUM_LAYERS):
                if layer_idx in attention_weights:
                    attn = attention_weights[layer_idx]  # [B, H, N, N]
                    for b in range(attn.shape[0]):
                        for h in range(NUM_HEADS):
                            attention_matrices[layer_idx][h].append(attn[b, h].numpy())
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return attention_matrices


def compute_average_attention_matrix(attention_matrices, layer_idx, head_idx):
    """Compute average attention matrix for a specific layer and head."""
    matrices = attention_matrices[layer_idx][head_idx]
    if not matrices:
        return None
    return np.mean(np.stack(matrices), axis=0)


def get_per_layer_attention_matrices(attention_matrices, head_idx=0):
    """
    Get average attention matrix for each layer (for a specific head).
    
    Args:
        attention_matrices: dict from extract_attention_matrices()
        head_idx: which head to use (0-11)
    
    Returns:
        A_list: [A_1, A_2, ..., A_12] - list of 12 attention matrices
    """
    A_list = []
    for layer_idx in range(NUM_LAYERS):
        matrices = attention_matrices[layer_idx][head_idx]
        if matrices:
            A_i = np.mean(np.stack(matrices), axis=0)  # Average across frames
        else:
            # Fallback: identity matrix if no data
            A_i = np.eye(SEQ_LEN)
        A_list.append(A_i)
    return A_list


# ============================================================================
# PART 2: Construct Block Matrix M
# ============================================================================

def construct_block_matrix_M(A_list, num_layers=12):
    """
    Construct the block matrix M for the iterative system with per-layer A_i.
    
    The system X_i = A_i · X_{i-1} + C is reformulated as M · X = b
    
    M has the structure:
        ┌ I     0     0    ...   0     0  ┐
        │-A_1   I     0    ...   0     0  │
        │ 0    -A_2   I    ...   0     0  │
        │ ⋮                  ⋱         ⋮  │
        └ 0     0     0    ... -A_12   I  ┘
    
    Args:
        A_list: [A_1, A_2, ..., A_12] - attention matrix per layer, each [n x n]
        num_layers: Number of transformer layers (12)
    
    Returns:
        M_sparse: Sparse block matrix [(num_layers+1)*n x (num_layers+1)*n]
    """
    n = A_list[0].shape[0]  # 197
    total_size = (num_layers + 1) * n  # 2561
    
    print(f"\nConstructing block matrix M with per-layer A_i:")
    print(f"  Number of layers: {num_layers}")
    print(f"  Block size n: {n}")
    print(f"  Total M size: {total_size} x {total_size}")
    
    rows, cols, data = [], [], []
    
    for i in range(num_layers + 1):
        # Identity block on diagonal: M[i*n:(i+1)*n, i*n:(i+1)*n] = I
        for j in range(n):
            rows.append(i * n + j)
            cols.append(i * n + j)
            data.append(1.0)
        
        # -A_i block on sub-diagonal (for i > 0)
        if i > 0:
            A_i = A_list[i - 1]  # A_1 for row block 1, A_2 for row block 2, etc.
            for row in range(n):
                for col in range(n):
                    if abs(A_i[row, col]) > 1e-10:  # Only store non-zero entries
                        rows.append(i * n + row)
                        cols.append((i - 1) * n + col)
                        data.append(-A_i[row, col])
    
    # Create sparse matrix
    M_sparse = sparse.coo_matrix((data, (rows, cols)), shape=(total_size, total_size))
    M_sparse = M_sparse.tocsr()  # Convert to CSR for efficient operations
    
    print(f"  M constructed with {len(data)} non-zero entries")
    print(f"  Sparse density: {len(data) / (total_size ** 2) * 100:.4f}%")
    
    return M_sparse


def construct_rhs_vector_b(X_0, C, num_layers=12):
    """
    Construct the right-hand side vector b for M · X = b.
    
    b = [X_0, C, C, ..., C]^T
    
    Using constant C for all layers (simplification).
    
    Args:
        X_0: Initial state [n] or [n x d] flattened
        C: Constant term (same for all layers) [n] or [n x d] flattened
        num_layers: Number of transformer layers
    
    Returns:
        b: RHS vector [(num_layers+1)*n]
    """
    X_0_flat = X_0.flatten()
    C_flat = C.flatten()
    n = len(X_0_flat)
    
    b = np.zeros((num_layers + 1) * n)
    b[:n] = X_0_flat
    for i in range(1, num_layers + 1):
        b[i*n:(i+1)*n] = C_flat  # Same C for all layers (simplification)
    
    return b


def estimate_constant_C(model, images, device='cpu'):
    """
    Estimate the constant C term from real data.
    
    C represents the "bias" or "residual" component that is added
    at each layer. We estimate it as the average difference between
    layer output and A @ layer_input.
    
    For simplification, we compute one global C (average across all layers).
    
    Args:
        model: ViT model
        images: Input images tensor
        device: Device to use
    
    Returns:
        C: Estimated constant term [SEQ_LEN]
    """
    print("\nEstimating constant C term...")
    
    layer_inputs = {}
    layer_outputs = {}
    
    # Hook to capture layer inputs and outputs
    def make_io_hook(layer_idx):
        def hook(module, input, output):
            layer_inputs[layer_idx] = input[0].detach().cpu()
            layer_outputs[layer_idx] = output.detach().cpu()
        return hook
    
    # Register hooks
    hooks = []
    for layer_idx, block in enumerate(model.blocks):
        hook = block.register_forward_hook(make_io_hook(layer_idx))
        hooks.append(hook)
    
    C_estimates = []
    
    with torch.no_grad():
        for i in range(min(10, len(images))):  # Use subset for efficiency
            img = images[i:i+1].to(device)
            layer_inputs.clear()
            layer_outputs.clear()
            
            _ = model(img)
            
            # For each layer, estimate C = output - A @ input
            # Since we don't have A here, we use a simpler approach:
            # C ≈ average of (output - input) which captures the "added" part
            for layer_idx in range(NUM_LAYERS):
                if layer_idx in layer_inputs and layer_idx in layer_outputs:
                    inp = layer_inputs[layer_idx][0]  # [197, 768]
                    out = layer_outputs[layer_idx][0]  # [197, 768]
                    
                    # Simplified: C as mean difference (projected to seq_len)
                    diff = (out - inp).mean(dim=-1).numpy()  # [197]
                    C_estimates.append(diff)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Average across all estimates
    C = np.mean(np.stack(C_estimates), axis=0)
    print(f"  Estimated C shape: {C.shape}")
    print(f"  C mean: {C.mean():.6f}, std: {C.std():.6f}")
    
    return C


# ============================================================================
# PART 3: Analyze Sparsity and Condition Number
# ============================================================================

def analyze_sparsity(M_sparse, name="M"):
    """Analyze sparsity pattern of matrix M."""
    print(f"\n{'='*60}")
    print(f"SPARSITY ANALYSIS: {name}")
    print(f"{'='*60}")
    
    total_elements = M_sparse.shape[0] * M_sparse.shape[1]
    nnz = M_sparse.nnz
    sparsity = 1.0 - (nnz / total_elements)
    
    print(f"  Matrix shape: {M_sparse.shape}")
    print(f"  Non-zero elements: {nnz:,}")
    print(f"  Total elements: {total_elements:,}")
    print(f"  Sparsity: {sparsity * 100:.4f}%")
    print(f"  Density: {(1 - sparsity) * 100:.4f}%")
    
    # Row sparsity (max non-zeros per row)
    row_nnz = np.diff(M_sparse.indptr)
    max_row_nnz = np.max(row_nnz)
    avg_row_nnz = np.mean(row_nnz)
    
    print(f"  Max non-zeros per row: {max_row_nnz}")
    print(f"  Avg non-zeros per row: {avg_row_nnz:.2f}")
    
    # For QLSA, we need sparsity parameter s = max non-zeros per row/column
    s = max_row_nnz
    print(f"\n  QLSA sparsity parameter s = {s}")
    
    return {
        'shape': M_sparse.shape,
        'nnz': nnz,
        'sparsity': sparsity,
        'max_row_nnz': max_row_nnz,
        'avg_row_nnz': avg_row_nnz,
        's_parameter': s
    }


def analyze_condition_number(M_sparse, name="M", num_singular_values=10):
    """
    Analyze condition number of matrix M.
    
    For QLSA, complexity is O(log N · κ · s) where κ is condition number.
    """
    print(f"\n{'='*60}")
    print(f"CONDITION NUMBER ANALYSIS: {name}")
    print(f"{'='*60}")
    
    n = M_sparse.shape[0]
    
    # For large matrices, use sparse SVD to estimate condition number
    # Compute largest and smallest singular values
    print(f"  Computing largest singular values...")
    
    try:
        # Largest singular values
        k_large = min(num_singular_values, n - 2)
        U_large, s_large, Vt_large = svds(M_sparse.astype(float), k=k_large, which='LM')
        sigma_max = np.max(s_large)
        
        print(f"  Computing smallest singular values...")
        # Smallest singular values
        k_small = min(num_singular_values, n - 2)
        U_small, s_small, Vt_small = svds(M_sparse.astype(float), k=k_small, which='SM')
        sigma_min = np.min(s_small[s_small > 1e-10])  # Avoid division by zero
        
        condition_number = sigma_max / sigma_min
        
        print(f"\n  Largest singular value sigma_max: {sigma_max:.6f}")
        print(f"  Smallest singular value sigma_min: {sigma_min:.6f}")
        print(f"  Condition number kappa = sigma_max/sigma_min: {condition_number:.2f}")
        
        # QLSA complexity estimate
        s = analyze_sparsity(M_sparse, name)['s_parameter']
        N = M_sparse.shape[0]
        qlsa_complexity = np.log2(N) * condition_number * s
        classical_complexity = N ** 3
        
        print(f"\n  QLSA Complexity Estimate:")
        print(f"    O(log N * kappa * s) = {qlsa_complexity:.2e}")
        print(f"    Classical O(N^3) = {classical_complexity:.2e}")
        print(f"    Potential speedup: {classical_complexity / qlsa_complexity:.2e}x")
        
        return {
            'sigma_max': sigma_max,
            'sigma_min': sigma_min,
            'condition_number': condition_number,
            'qlsa_complexity': qlsa_complexity,
            'classical_complexity': classical_complexity
        }
        
    except Exception as e:
        print(f"  Error computing condition number: {e}")
        print(f"  Matrix may be too large or ill-conditioned")
        return None


def analyze_attention_matrix_properties(A, name="A"):
    """Analyze properties of the attention matrix A."""
    print(f"\n{'='*60}")
    print(f"ATTENTION MATRIX ANALYSIS: {name}")
    print(f"{'='*60}")
    
    print(f"  Shape: {A.shape}")
    print(f"  Min value: {A.min():.6f}")
    print(f"  Max value: {A.max():.6f}")
    print(f"  Mean value: {A.mean():.6f}")
    
    # Check row sums (should be ~1 for attention matrices after softmax)
    row_sums = A.sum(axis=1)
    print(f"  Row sums - min: {row_sums.min():.6f}, max: {row_sums.max():.6f}")
    
    # Compute rank via SVD
    U, s, Vt = np.linalg.svd(A)
    
    # Effective rank at different thresholds
    total_var = np.sum(s ** 2)
    cumsum = np.cumsum(s ** 2)
    
    rank_90 = np.searchsorted(cumsum / total_var, 0.90) + 1
    rank_95 = np.searchsorted(cumsum / total_var, 0.95) + 1
    rank_99 = np.searchsorted(cumsum / total_var, 0.99) + 1
    
    print(f"\n  Singular value analysis:")
    print(f"    Top 5 singular values: {s[:5]}")
    print(f"    Rank for 90% variance: {rank_90}")
    print(f"    Rank for 95% variance: {rank_95}")
    print(f"    Rank for 99% variance: {rank_99}")
    
    # Condition number of A
    cond_A = s[0] / s[s > 1e-10][-1]
    print(f"    Condition number of A: {cond_A:.2f}")
    
    return {
        'shape': A.shape,
        'rank_90': rank_90,
        'rank_95': rank_95,
        'rank_99': rank_99,
        'condition_number': cond_A,
        'singular_values': s
    }


def analyze_per_layer_attention(A_list):
    """Analyze properties of each layer's attention matrix."""
    print(f"\n{'='*60}")
    print(f"PER-LAYER ATTENTION ANALYSIS")
    print(f"{'='*60}")
    
    print(f"\n  {'Layer':<8} {'Rank@99%':<12} {'Cond. Num.':<14} {'Mean':<10} {'Max':<10}")
    print(f"  {'-'*54}")
    
    layer_stats = []
    for i, A_i in enumerate(A_list):
        U, s, Vt = np.linalg.svd(A_i)
        total_var = np.sum(s ** 2)
        cumsum = np.cumsum(s ** 2)
        rank_99 = np.searchsorted(cumsum / total_var, 0.99) + 1
        cond = s[0] / s[s > 1e-10][-1]
        
        stats = {
            'layer': i + 1,
            'rank_99': rank_99,
            'condition_number': cond,
            'mean': A_i.mean(),
            'max': A_i.max()
        }
        layer_stats.append(stats)
        
        print(f"  {i+1:<8} {rank_99:<12} {cond:<14.2f} {A_i.mean():<10.4f} {A_i.max():<10.4f}")
    
    return layer_stats


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(A_list, M_sparse, layer_stats, output_dir, head_idx):
    """Generate visualization plots."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Attention matrices from different layers (sample 4)
    ax1 = axes[0, 0]
    sample_layers = [0, 3, 7, 11]  # Layers 1, 4, 8, 12
    combined = np.zeros((197, 197 * 4))
    for idx, layer_idx in enumerate(sample_layers):
        combined[:, idx*197:(idx+1)*197] = A_list[layer_idx]
    im1 = ax1.imshow(combined, cmap='viridis', aspect='auto')
    ax1.set_title(f'Attention Matrices A_i (Layers 1, 4, 8, 12) - Head {head_idx}', fontsize=12)
    ax1.set_xlabel('Key Position (concatenated)')
    ax1.set_ylabel('Query Position')
    ax1.set_xticks([98, 295, 492, 689])
    ax1.set_xticklabels(['L1', 'L4', 'L8', 'L12'])
    plt.colorbar(im1, ax=ax1)
    
    # 2. Rank@99% across layers
    ax2 = axes[0, 1]
    layers = [s['layer'] for s in layer_stats]
    ranks = [s['rank_99'] for s in layer_stats]
    ax2.bar(layers, ranks, color='steelblue', edgecolor='black')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Rank for 99% Variance')
    ax2.set_title('Low-Rank Structure Across Layers')
    ax2.set_xticks(layers)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Sparsity pattern of M (small portion)
    ax3 = axes[1, 0]
    M_dense_small = M_sparse[:600, :600].toarray()
    ax3.spy(M_dense_small, markersize=0.3)
    ax3.set_title('Sparsity Pattern of M (first 600x600)\nShowing block-bidiagonal structure')
    ax3.set_xlabel('Column Index')
    ax3.set_ylabel('Row Index')
    
    # 4. Condition number across layers
    ax4 = axes[1, 1]
    conds = [s['condition_number'] for s in layer_stats]
    ax4.semilogy(layers, conds, 'o-', color='crimson', linewidth=2, markersize=8)
    ax4.set_xlabel('Layer')
    ax4.set_ylabel('Condition Number (log scale)')
    ax4.set_title('Condition Number of A_i Across Layers')
    ax4.set_xticks(layers)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/qlsa_analysis_per_layer.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlots saved to {output_dir}/qlsa_analysis_per_layer.png")


def save_report(A_list, layer_stats, sparsity_analysis, condition_analysis, output_dir, head_idx):
    """Save analysis report to text file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    report_path = f'{output_dir}/qlsa_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("QLSA FORMULATION ANALYSIS REPORT (Per-Layer A_i)\n")
        f.write("ViT-B/16 Attention -> Block Matrix M for Quantum Linear System\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Configuration:\n")
        f.write(f"  Head analyzed: {head_idx}\n")
        f.write(f"  Number of layers: {NUM_LAYERS}\n")
        f.write(f"  Sequence length: {SEQ_LEN}\n\n")
        
        f.write("PER-LAYER ATTENTION MATRIX ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Layer':<8} {'Rank@99%':<12} {'Cond. Num.':<14}\n")
        for stats in layer_stats:
            f.write(f"{stats['layer']:<8} {stats['rank_99']:<12} {stats['condition_number']:<14.2f}\n")
        f.write("\n")
        
        avg_rank = np.mean([s['rank_99'] for s in layer_stats])
        max_rank = np.max([s['rank_99'] for s in layer_stats])
        min_rank = np.min([s['rank_99'] for s in layer_stats])
        f.write(f"Summary:\n")
        f.write(f"  Average rank@99%: {avg_rank:.1f}\n")
        f.write(f"  Min rank@99%: {min_rank}\n")
        f.write(f"  Max rank@99%: {max_rank}\n\n")
        
        f.write("BLOCK MATRIX M SPARSITY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Shape: {sparsity_analysis['shape']}\n")
        f.write(f"Non-zero elements: {sparsity_analysis['nnz']:,}\n")
        f.write(f"Sparsity: {sparsity_analysis['sparsity'] * 100:.4f}%\n")
        f.write(f"QLSA sparsity parameter s: {sparsity_analysis['s_parameter']}\n\n")
        
        if condition_analysis:
            f.write("CONDITION NUMBER & COMPLEXITY (Block Matrix M)\n")
            f.write("-" * 40 + "\n")
            f.write(f"sigma_max: {condition_analysis['sigma_max']:.6f}\n")
            f.write(f"sigma_min: {condition_analysis['sigma_min']:.6f}\n")
            f.write(f"Condition number kappa: {condition_analysis['condition_number']:.2f}\n")
            f.write(f"QLSA complexity O(log N * kappa * s): {condition_analysis['qlsa_complexity']:.2e}\n")
            f.write(f"Classical complexity O(N^3): {condition_analysis['classical_complexity']:.2e}\n")
            f.write(f"Potential quantum speedup: {condition_analysis['classical_complexity'] / condition_analysis['qlsa_complexity']:.2e}x\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("INTERPRETATION FOR QUANTUM APPROACH\n")
        f.write("=" * 70 + "\n")
        f.write("""
This analysis uses PER-LAYER attention matrices A_i:
    X_i = A_i * X_{i-1} + C  for i = 1, ..., 12

Block matrix M structure:
    | I     0     0    ...   0     0  |
    |-A_1   I     0    ...   0     0  |
    | 0    -A_2   I    ...   0     0  |
    | :                  .         :  |
    | 0     0     0    ... -A_12   I  |

Key findings:
1. Each layer has a DIFFERENT attention matrix A_i
   This is more accurate than using a single A for all layers.

2. The low-rank structure varies across layers
   Early/late layers may have different effective ranks.

3. The block matrix M is still highly sparse
   Sparsity comes from the block-bidiagonal structure.

4. C is treated as constant (simplification)
   In reality, C_i varies per layer due to MLP nonlinearity.

Simplifications made:
- Using average attention across frames (A_i is input-dependent)
- Using constant C for all layers (C_i varies in reality)
- Analyzing single head (12 heads in parallel in actual ViT)

Next steps:
- Implement quantum state preparation for |b>
- Design quantum circuit for block-encoding M
- Analyze measurement strategy for extracting |X>
""")
    
    print(f"Report saved to {report_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='QLSA Formulation Analysis (Per-Layer A_i)')
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH,
                        help='Path to ViT checkpoint')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR,
                        help='Output directory')
    parser.add_argument('--num-frames', type=int, default=NUM_FRAMES,
                        help='Number of frames to analyze')
    parser.add_argument('--head', type=int, default=0,
                        help='Head index to analyze (0-11)')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # ========================================
    # PART 1: Extract Attention Matrices
    # ========================================
    print("\n" + "=" * 70)
    print("PART 1: EXTRACTING ATTENTION MATRICES (ALL LAYERS)")
    print("=" * 70)
    
    model = load_vit_model(args.checkpoint, device)
    
    print("\nLoading video frames...")
    images = load_frames(VIDEO_PATHS, args.num_frames)
    
    attention_matrices = extract_attention_matrices(model, images, device)
    
    # Get per-layer attention matrices for specified head
    print(f"\nExtracting per-layer attention matrices for Head {args.head}...")
    A_list = get_per_layer_attention_matrices(attention_matrices, head_idx=args.head)
    print(f"  Extracted {len(A_list)} attention matrices (one per layer)")
    print(f"  Each A_i shape: {A_list[0].shape}")
    
    # Analyze each layer's attention matrix
    layer_stats = analyze_per_layer_attention(A_list)
    
    # ========================================
    # PART 2: Construct Block Matrix M
    # ========================================
    print("\n" + "=" * 70)
    print("PART 2: CONSTRUCTING BLOCK MATRIX M (WITH PER-LAYER A_i)")
    print("=" * 70)
    
    M_sparse = construct_block_matrix_M(A_list, num_layers=NUM_LAYERS)
    
    # ========================================
    # PART 3: Analyze Sparsity & Condition Number
    # ========================================
    print("\n" + "=" * 70)
    print("PART 3: SPARSITY & CONDITION NUMBER ANALYSIS")
    print("=" * 70)
    
    sparsity_analysis = analyze_sparsity(M_sparse, "Block Matrix M")
    
    print("\nComputing condition number (this may take a while)...")
    condition_analysis = analyze_condition_number(M_sparse, "Block Matrix M")
    
    # ========================================
    # PART 4: Estimate Constant C (Optional)
    # ========================================
    print("\n" + "=" * 70)
    print("PART 4: ESTIMATING CONSTANT C")
    print("=" * 70)
    
    C = estimate_constant_C(model, images, device)
    
    # Example: construct RHS vector b (if we had X_0)
    # X_0 = np.zeros(SEQ_LEN)  # Placeholder
    # b = construct_rhs_vector_b(X_0, C, num_layers=NUM_LAYERS)
    
    # ========================================
    # Save Results
    # ========================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    plot_results(A_list, M_sparse, layer_stats, args.output, args.head)
    save_report(A_list, layer_stats, sparsity_analysis, condition_analysis, args.output, args.head)
    
    # Save matrices for later use
    np.savez(f'{args.output}/qlsa_matrices.npz',
             A_list=np.stack(A_list),
             C=C,
             M_data=M_sparse.data,
             M_indices=M_sparse.indices,
             M_indptr=M_sparse.indptr,
             M_shape=M_sparse.shape)
    print(f"Matrices saved to {args.output}/qlsa_matrices.npz")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()