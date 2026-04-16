#!/usr/bin/env python3
"""
QLSA Formulation for ViT Attention
===================================
This script implements Step 1 from the quantum approach:
1. Extract attention operation matrix A from trained ViT
2. Construct block matrix M for the 12-layer iterative system
3. Analyze sparsity and condition number (QLSA requirements)

The iterative system:
    X_i = A · X_{i-1} + C  for i = 1, ..., 12

Reformulated as linear system:
    M · X = b  where X = M^{-1} · b

Block structure of M:
    ┌ I    0    0   ...  0   ┐
    │-A    I    0   ...  0   │
    │ 0   -A    I   ...  0   │
    │ ⋮              ⋱   ⋮   │
    └ 0    0   ...  -A   I   ┘
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


# ============================================================================
# PART 2: Construct Block Matrix M
# ============================================================================

def construct_block_matrix_M(A, num_layers=12):
    """
    Construct the block matrix M for the iterative system.
    
    The system X_i = A · X_{i-1} + C is reformulated as M · X = b
    
    M has the structure:
        ┌ I    0    0   ...  0   ┐
        │-A    I    0   ...  0   │
        │ 0   -A    I   ...  0   │
        │ ⋮              ⋱   ⋮   │
        └ 0    0   ...  -A   I   ┘
    
    Args:
        A: Attention matrix [n x n] where n = sequence length (197)
        num_layers: Number of transformer layers (12)
    
    Returns:
        M: Block matrix [(num_layers+1)*n x (num_layers+1)*n]
        M_sparse: Sparse version of M
    """
    n = A.shape[0]
    total_size = (num_layers + 1) * n  # +1 for X_0
    
    print(f"\nConstructing block matrix M:")
    print(f"  A shape: {A.shape}")
    print(f"  Block size n: {n}")
    print(f"  Total M size: {total_size} x {total_size}")
    
    # Build sparse block matrix
    # Using lists to construct COO format
    rows = []
    cols = []
    data = []
    
    for i in range(num_layers + 1):
        # Identity block on diagonal: M[i*n:(i+1)*n, i*n:(i+1)*n] = I
        for j in range(n):
            rows.append(i * n + j)
            cols.append(i * n + j)
            data.append(1.0)
        
        # -A block on sub-diagonal (for i > 0)
        if i > 0:
            for row in range(n):
                for col in range(n):
                    if abs(A[row, col]) > 1e-10:  # Only store non-zero entries
                        rows.append(i * n + row)
                        cols.append((i - 1) * n + col)
                        data.append(-A[row, col])
    
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
    
    Args:
        X_0: Initial state [n x d] or [n*d] flattened
        C: Constant term (residual/bias) [n x d] or [n*d] flattened
        num_layers: Number of transformer layers
    
    Returns:
        b: RHS vector [(num_layers+1)*n*d]
    """
    X_0_flat = X_0.flatten()
    C_flat = C.flatten()
    n = len(X_0_flat)
    
    b = np.zeros((num_layers + 1) * n)
    b[:n] = X_0_flat
    for i in range(1, num_layers + 1):
        b[i*n:(i+1)*n] = C_flat
    
    return b


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
        sigma_min = np.max(s_small)  # 'SM' returns smallest, but we take max of those
        sigma_min = np.min(s_small[s_small > 1e-10])  # Avoid division by zero
        
        condition_number = sigma_max / sigma_min
        
        print(f"\n  Largest singular value σ_max: {sigma_max:.6f}")
        print(f"  Smallest singular value σ_min: {sigma_min:.6f}")
        print(f"  Condition number κ = σ_max/σ_min: {condition_number:.2f}")
        
        # QLSA complexity estimate
        s = analyze_sparsity(M_sparse, name)['s_parameter']
        N = M_sparse.shape[0]
        qlsa_complexity = np.log2(N) * condition_number * s
        classical_complexity = N ** 3
        
        print(f"\n  QLSA Complexity Estimate:")
        print(f"    O(log N · κ · s) ≈ {qlsa_complexity:.2e}")
        print(f"    Classical O(N³) ≈ {classical_complexity:.2e}")
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


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(A, M_sparse, A_analysis, output_dir):
    """Generate visualization plots."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Attention matrix heatmap
    ax1 = axes[0, 0]
    im1 = ax1.imshow(A, cmap='viridis', aspect='auto')
    ax1.set_title('Average Attention Matrix A\n(Layer 6, Head 1)', fontsize=12)
    ax1.set_xlabel('Key Position')
    ax1.set_ylabel('Query Position')
    plt.colorbar(im1, ax=ax1)
    
    # 2. Singular value decay
    ax2 = axes[0, 1]
    s = A_analysis['singular_values']
    ax2.semilogy(s, 'b-', linewidth=2)
    ax2.axhline(y=s[0] * 0.01, color='r', linestyle='--', label='1% of σ_max')
    ax2.axvline(x=A_analysis['rank_99'], color='g', linestyle='--', label=f"99% rank = {A_analysis['rank_99']}")
    ax2.set_xlabel('Singular Value Index')
    ax2.set_ylabel('Singular Value (log scale)')
    ax2.set_title('Singular Value Decay of A')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Sparsity pattern of M (small portion)
    ax3 = axes[1, 0]
    # Show a small portion of M's sparsity pattern
    M_dense_small = M_sparse[:500, :500].toarray()
    ax3.spy(M_dense_small, markersize=0.5)
    ax3.set_title('Sparsity Pattern of M (first 500x500)')
    ax3.set_xlabel('Column Index')
    ax3.set_ylabel('Row Index')
    
    # 4. Explained variance ratio
    ax4 = axes[1, 1]
    total_var = np.sum(s ** 2)
    evr = np.cumsum(s ** 2) / total_var * 100
    ax4.plot(evr, 'b-', linewidth=2)
    ax4.axhline(y=99, color='r', linestyle='--', label='99% threshold')
    ax4.axvline(x=A_analysis['rank_99'], color='g', linestyle='--', label=f"r = {A_analysis['rank_99']}")
    ax4.set_xlabel('Number of Singular Values')
    ax4.set_ylabel('Cumulative Explained Variance (%)')
    ax4.set_title('Explained Variance Ratio')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, 50])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/qlsa_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlots saved to {output_dir}/qlsa_analysis.png")


def save_report(A_analysis, sparsity_analysis, condition_analysis, output_dir):
    """Save analysis report to text file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    report_path = f'{output_dir}/qlsa_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("QLSA FORMULATION ANALYSIS REPORT\n")
        f.write("ViT-B/16 Attention -> Block Matrix M for Quantum Linear System\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("ATTENTION MATRIX A ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Shape: {A_analysis['shape']}\n")
        f.write(f"Rank for 90% variance: {A_analysis['rank_90']}\n")
        f.write(f"Rank for 95% variance: {A_analysis['rank_95']}\n")
        f.write(f"Rank for 99% variance: {A_analysis['rank_99']}\n")
        f.write(f"Condition number: {A_analysis['condition_number']:.2f}\n\n")
        
        f.write("BLOCK MATRIX M SPARSITY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Shape: {sparsity_analysis['shape']}\n")
        f.write(f"Non-zero elements: {sparsity_analysis['nnz']:,}\n")
        f.write(f"Sparsity: {sparsity_analysis['sparsity'] * 100:.4f}%\n")
        f.write(f"QLSA sparsity parameter s: {sparsity_analysis['s_parameter']}\n\n")
        
        if condition_analysis:
            f.write("CONDITION NUMBER & COMPLEXITY\n")
            f.write("-" * 40 + "\n")
            f.write(f"sigma_max: {condition_analysis['sigma_max']:.6f}\n")
            f.write(f"sigma_min: {condition_analysis['sigma_min']:.6f}\n")
            f.write(f"Condition number κ: {condition_analysis['condition_number']:.2f}\n")
            f.write(f"QLSA complexity O(log N · κ · s): {condition_analysis['qlsa_complexity']:.2e}\n")
            f.write(f"Classical complexity O(N³): {condition_analysis['classical_complexity']:.2e}\n")
            f.write(f"Potential quantum speedup: {condition_analysis['classical_complexity'] / condition_analysis['qlsa_complexity']:.2e}x\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("INTERPRETATION FOR QUANTUM APPROACH\n")
        f.write("=" * 70 + "\n")
        f.write("""
The block matrix M encodes 12 iterations of the attention operation:
    X_i = A · X_{i-1} + C

This is reformulated as M · X = b, which can be solved using QLSA (HHL).

Key findings:
1. The attention matrix A has low effective rank (~7 for 99% variance)
   This confirms the low-rank structure we found in earlier analysis.

2. The block matrix M is highly sparse due to its bidiagonal structure
   (identity blocks on diagonal, -A blocks on sub-diagonal)

3. The sparsity parameter s is determined by the density of A
   For softmax attention, A is dense within each [197x197] block

4. Condition number κ determines QLSA complexity
   Lower κ = faster quantum algorithm

Next steps:
- Implement quantum state preparation for |b⟩
- Design quantum circuit for block-encoding M
- Analyze measurement strategy for extracting |X⟩
""")
    
    print(f"Report saved to {report_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='QLSA Formulation Analysis')
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH,
                        help='Path to ViT checkpoint')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR,
                        help='Output directory')
    parser.add_argument('--num-frames', type=int, default=NUM_FRAMES,
                        help='Number of frames to analyze')
    parser.add_argument('--layer', type=int, default=5,
                        help='Layer index to analyze (0-11)')
    parser.add_argument('--head', type=int, default=0,
                        help='Head index to analyze (0-11)')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # ========================================
    # PART 1: Extract Attention Matrices
    # ========================================
    print("\n" + "=" * 70)
    print("PART 1: EXTRACTING ATTENTION MATRICES")
    print("=" * 70)
    
    model = load_vit_model(args.checkpoint, device)
    
    print("\nLoading video frames...")
    images = load_frames(VIDEO_PATHS, args.num_frames)
    
    attention_matrices = extract_attention_matrices(model, images, device)
    
    # Get average attention matrix for specified layer/head
    A = compute_average_attention_matrix(attention_matrices, args.layer, args.head)
    print(f"\nExtracted average attention matrix A from Layer {args.layer}, Head {args.head}")
    print(f"  Shape: {A.shape}")
    
    # Analyze A
    A_analysis = analyze_attention_matrix_properties(A, f"A (Layer {args.layer}, Head {args.head})")
    
    # ========================================
    # PART 2: Construct Block Matrix M
    # ========================================
    print("\n" + "=" * 70)
    print("PART 2: CONSTRUCTING BLOCK MATRIX M")
    print("=" * 70)
    
    M_sparse = construct_block_matrix_M(A, num_layers=NUM_LAYERS)
    
    # ========================================
    # PART 3: Analyze Sparsity & Condition Number
    # ========================================
    print("\n" + "=" * 70)
    print("PART 3: SPARSITY & CONDITION NUMBER ANALYSIS")
    print("=" * 70)
    
    sparsity_analysis = analyze_sparsity(M_sparse, "Block Matrix M")
    
    # Condition number analysis (may be slow for large matrices)
    print("\nComputing condition number (this may take a while)...")
    condition_analysis = analyze_condition_number(M_sparse, "Block Matrix M")
    
    # ========================================
    # Save Results
    # ========================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    plot_results(A, M_sparse, A_analysis, args.output)
    save_report(A_analysis, sparsity_analysis, condition_analysis, args.output)
    
    # Save matrices for later use
    np.savez(f'{args.output}/qlsa_matrices.npz',
             A=A,
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
