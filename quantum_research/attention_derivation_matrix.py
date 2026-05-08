#!/usr/bin/env python3
"""
Transformer Attention: Step-by-Step Matrix Dimension Derivation
================================================================

This script traces EVERY matrix dimension through one transformer layer,
showing exactly how to derive the M matrix for QLSA.

Run on your Windows machine:
    conda activate umi
    python attention_matrix_derivation.py

With your checkpoint:
    python attention_matrix_derivation.py --checkpoint vit_encoder_only.pt --layer 0

Author: For Dr. Guan's quantum-inspired ViT research
"""

import argparse
import sys

try:
    import torch
    import torch.nn.functional as F
    import numpy as np
    from pathlib import Path
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# ViT-B/16 CONFIGURATION
# =============================================================================

CONFIG = {
    'd_model': 768,         # Hidden dimension
    'num_heads': 12,        # Number of attention heads
    'd_k': 64,              # = d_model / num_heads, dimension per head
    'd_v': 64,              # Value dimension per head (same as d_k)
    'seq_len': 197,         # 196 patches + 1 CLS token
    'batch_size': 1,        # Use 1 for clarity
    'num_layers': 12,       # Number of transformer layers
}


def print_step(num, title):
    """Print a step header."""
    print(f"\n{'─' * 80}")
    print(f"  STEP {num}: {title}")
    print(f"{'─' * 80}")


def derive_dimensions_only():
    """
    Show the matrix dimension derivation WITHOUT computation.
    This runs without PyTorch - pure educational output.
    """
    d = CONFIG['d_model']
    H = CONFIG['num_heads']
    d_k = CONFIG['d_k']
    N = CONFIG['seq_len']
    B = CONFIG['batch_size']
    L = CONFIG['num_layers']
    
    print("\n" + "=" * 80)
    print("  TRANSFORMER ATTENTION: MATRIX DIMENSION DERIVATION")
    print("  ViT-B/16 Configuration")
    print("=" * 80)
    
    print(f"""
    Parameters:
    ├── d_model (hidden dim)     = {d}
    ├── num_heads (H)            = {H}
    ├── d_k = d_v (per head)     = {d_k}
    ├── seq_len (N)              = {N}  (196 patches + 1 CLS)
    ├── batch_size (B)           = {B}
    └── num_layers (L)           = {L}
    """)
    
    # =========================================================================
    print_step(0, "INPUT X")
    print(f"""
    X = token embeddings from previous layer (or patch embedding + pos encoding)
    
    Shape: [B × N × d] = [{B} × {N} × {d}]
    
    Each of the {N} tokens is represented by a {d}-dimensional vector.
    """)
    
    # =========================================================================
    print_step(1, "WEIGHT MATRICES W^Q, W^K, W^V")
    print(f"""
    Learned projection matrices (stored in the checkpoint):
    
    W^Q : [d × d] = [{d} × {d}]   ({d*d:,} parameters)
    W^K : [d × d] = [{d} × {d}]   ({d*d:,} parameters)
    W^V : [d × d] = [{d} × {d}]   ({d*d:,} parameters)
    
    In TIMM ViT, these are stored as a single combined matrix:
    W_qkv : [3d × d] = [{3*d} × {d}]   ({3*d*d:,} parameters)
    
    Location in checkpoint:
    blocks.{{layer}}.attn.qkv.weight → shape [{3*d}, {d}]
    """)
    
    # =========================================================================
    print_step(2, "COMPUTE Q, K, V PROJECTIONS")
    print(f"""
    Q = X @ W^Q.T    (or equivalently: Q = Linear(X) where Linear.weight = W^Q)
    K = X @ W^K.T
    V = X @ W^V.T
    
    Matrix multiplication:
    
    Q = X         @    W^Q.T
      = [B × N × d]  @  [d × d]
      = [{B} × {N} × {d}]  @  [{d} × {d}]
      = [{B} × {N} × {d}]
    
    Result shapes:
    Q : [{B} × {N} × {d}]
    K : [{B} × {N} × {d}]
    V : [{B} × {N} × {d}]
    """)
    
    # =========================================================================
    print_step(3, "RESHAPE FOR MULTI-HEAD ATTENTION")
    print(f"""
    Split d={d} into {H} heads × {d_k} dimensions per head:
    
    Q : [{B} × {N} × {d}]
      → view as [{B} × {N} × {H} × {d_k}]
      → transpose to [{B} × {H} × {N} × {d_k}]
    
    Same for K and V.
    
    After reshape:
    Q_heads : [{B} × {H} × {N} × {d_k}] = [{B} × {H} × {N} × {d_k}]
    K_heads : [{B} × {H} × {N} × {d_k}]
    V_heads : [{B} × {H} × {N} × {d_k}]
    
    Now each head h has:
    Q_h : [{B} × {N} × {d_k}]  (slice along head dimension)
    K_h : [{B} × {N} × {d_k}]
    V_h : [{B} × {N} × {d_k}]
    """)
    
    # =========================================================================
    print_step(4, "ATTENTION SCORES: Q @ K^T / √d_k")
    print(f"""
    For each head, compute the attention score matrix:
    
    scores_h = Q_h @ K_h^T / √{d_k}
    
    Matrix multiplication (per head):
    
    Q_h @ K_h^T = [{N} × {d_k}] @ [{d_k} × {N}]
                = [{N} × {N}]
    
    With batch and all heads:
    
    scores = Q_heads @ K_heads.transpose(-2, -1) / √{d_k}
           = [{B} × {H} × {N} × {d_k}] @ [{B} × {H} × {d_k} × {N}]
           = [{B} × {H} × {N} × {N}]
    
    ╔══════════════════════════════════════════════════════════════════════╗
    ║  scores[b, h, i, j] = "how much should token i attend to token j"    ║
    ║                       in batch b, head h                             ║
    ║                                                                      ║
    ║  This [{N} × {N}] matrix (per head) is the RAW attention matrix.           ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # =========================================================================
    print_step(5, "SOFTMAX → ATTENTION WEIGHTS A")
    print(f"""
    Apply softmax along the key dimension (last dim):
    
    A = softmax(scores, dim=-1)
    
    A : [{B} × {H} × {N} × {N}]
    
    Properties of A:
    • Each row sums to 1.0 (probability distribution)
    • A[b, h, i, :] = attention distribution for query token i in head h
    • A[b, h, i, j] ∈ [0, 1] = weight for key token j
    
    ╔══════════════════════════════════════════════════════════════════════╗
    ║  THIS IS THE ATTENTION MATRIX A FOR YOUR QLSA FORMULATION            ║
    ║                                                                      ║
    ║  Per head, per layer:  A_{{ℓ,h}} has shape [{N} × {N}]                     ║
    ║                                                                      ║
    ║  For QLSA, you average across heads (and possibly frames):           ║
    ║    A_ℓ = mean(A_{{ℓ,h}} for h in 1..{H})   →  shape [{N} × {N}]            ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # =========================================================================
    print_step(6, "WEIGHTED SUM: head = A @ V")
    print(f"""
    Multiply attention weights by values:
    
    head_h = A_h @ V_h
           = [{N} × {N}] @ [{N} × {d_k}]
           = [{N} × {d_k}]
    
    With batch and all heads:
    
    head_outputs = A @ V_heads
                 = [{B} × {H} × {N} × {N}] @ [{B} × {H} × {N} × {d_k}]
                 = [{B} × {H} × {N} × {d_k}]
    
    Each token i now has a weighted combination of all value vectors,
    where the weights come from the attention distribution A[i, :].
    """)
    
    # =========================================================================
    print_step(7, "CONCATENATE HEADS")
    print(f"""
    Reshape back to [B × N × d]:
    
    head_outputs : [{B} × {H} × {N} × {d_k}]
      → transpose to [{B} × {N} × {H} × {d_k}]
      → view as [{B} × {N} × {H * d_k}]
      → = [{B} × {N} × {d}]
    
    concat : [{B} × {N} × {d}]
    
    The {H} heads (each with {d_k} dims) are concatenated back to {d} dims.
    """)
    
    # =========================================================================
    print_step(8, "OUTPUT PROJECTION: W^O")
    print(f"""
    Final linear projection:
    
    W^O : [d × d] = [{d} × {d}]
    
    attn_out = concat @ W^O.T
             = [{B} × {N} × {d}] @ [{d} × {d}]
             = [{B} × {N} × {d}]
    
    Location in checkpoint:
    blocks.{{layer}}.attn.proj.weight → shape [{d}, {d}]
    """)
    
    # =========================================================================
    print_step(9, "RESIDUAL CONNECTION")
    print(f"""
    X' = X + attn_out
       = [{B} × {N} × {d}] + [{B} × {N} × {d}]
       = [{B} × {N} × {d}]
    
    This is why we can write:
    
        X_ℓ = X_{{ℓ-1}} + Attention(X_{{ℓ-1}})
        
    Or in your QLSA simplified form:
    
        X_ℓ ≈ A_ℓ @ X_{{ℓ-1}} + C_ℓ
    """)
    
    # =========================================================================
    print("\n" + "=" * 80)
    print("  BLOCK MATRIX M FOR QLSA")
    print("=" * 80)
    
    print(f"""
    For L={L} layers, stack the recurrence X_ℓ = A_ℓ @ X_{{ℓ-1}} + C_ℓ:
    
    Rearrange:  X_ℓ - A_ℓ @ X_{{ℓ-1}} = C_ℓ
    
    Written as a block linear system M @ X_vec = b_vec:
    
    ┌                                        ┐   ┌     ┐     ┌              ┐
    │  I      0      0      0    ...    0    │   │ X₁  │     │ A₁@X₀ + C₁  │
    │ -A₂     I      0      0    ...    0    │   │ X₂  │     │     C₂      │
    │  0     -A₃     I      0    ...    0    │   │ X₃  │     │     C₃      │
    │  0      0     -A₄     I    ...    0    │ @ │ X₄  │  =  │     C₄      │
    │  ⋮      ⋮      ⋮      ⋮     ⋱     ⋮    │   │  ⋮  │     │      ⋮      │
    │  0      0      0      0   -A_L    I    │   │ X_L │     │     C_L     │
    └                                        ┘   └     ┘     └              ┘
    
    Where:
    • Each A_ℓ is [{N} × {N}] = [{N} × {N}]
    • Each I is [{N} × {N}] identity
    • Each X_ℓ is [{N} × {d}] flattened to [{N * d}] for the system
    
    Total M dimensions:
    • Rows: L × N = {L} × {N} = {L * N}
    • Cols: L × N = {L} × {N} = {L * N}
    • M shape: [{L * N} × {L * N}]
    
    ╔══════════════════════════════════════════════════════════════════════╗
    ║  M is BLOCK-BIDIAGONAL with:                                         ║
    ║  • Identity blocks on the diagonal                                   ║
    ║  • -A_ℓ blocks on the sub-diagonal                                   ║
    ║  • Zeros elsewhere                                                   ║
    ║                                                                      ║
    ║  This sparse structure is what makes QLSA efficient!                 ║
    ║  • Sparsity ≈ {100 * (1 - 2/L):.1f}% (most blocks are zero)                          ║
    ║  • Max nnz per row = {N} + {N} = {2*N} (one I block + one A block)            ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # =========================================================================
    print("\n" + "=" * 80)
    print("  SUMMARY: KEY MATRICES TO EXTRACT")
    print("=" * 80)
    
    print(f"""
    From your ViT checkpoint, for each layer ℓ ∈ [0, {L-1}]:
    
    1. Extract W^Q, W^K, W^V (combined as qkv.weight in TIMM):
       blocks.{{ℓ}}.attn.qkv.weight  →  [{3*d} × {d}]
       Split into W^Q, W^K, W^V each [{d} × {d}]
    
    2. For input frames X (from your video data):
       Compute Q = X @ W^Q.T   →  [{B} × {N} × {d}]
       Compute K = X @ W^K.T   →  [{B} × {N} × {d}]
    
    3. Reshape for multi-head:
       Q_heads  →  [{B} × {H} × {N} × {d_k}]
       K_heads  →  [{B} × {H} × {N} × {d_k}]
    
    4. Compute attention matrix:
       scores = Q_heads @ K_heads.T / √{d_k}  →  [{B} × {H} × {N} × {N}]
       A = softmax(scores)                    →  [{B} × {H} × {N} × {N}]
    
    5. Average across heads and frames:
       A_ℓ = A.mean(dim=[0, 1])  →  [{N} × {N}]
    
    6. Build M:
       M[ℓ*{N} : (ℓ+1)*{N},  ℓ*{N} : (ℓ+1)*{N}] = I
       M[ℓ*{N} : (ℓ+1)*{N},  (ℓ-1)*{N} : ℓ*{N}] = -A_ℓ  (for ℓ > 0)
    
    7. Analyze M:
       κ = condition_number(M)
       s = max_nnz_per_row(M)
       QLSA complexity ≈ O(log({L*N}) × κ × s)
    """)


def compute_with_torch(checkpoint_path=None, layer_idx=0):
    """
    Run the actual computation with PyTorch, extracting real weights.
    """
    d = CONFIG['d_model']
    H = CONFIG['num_heads']
    d_k = CONFIG['d_k']
    N = CONFIG['seq_len']
    B = CONFIG['batch_size']
    
    print("\n" + "=" * 80)
    print("  RUNNING WITH PYTORCH - ACTUAL MATRIX COMPUTATION")
    print("=" * 80)
    
    # Load or create weights
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"\nLoading weights from: {checkpoint_path}")
        
        if checkpoint_path.endswith('.ckpt'):
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            state_dict = ckpt['state_dicts']['model']
            prefix = f'obs_encoder.model.blocks.{layer_idx}.attn.'
        else:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            prefix = f'blocks.{layer_idx}.attn.'
        
        W_qkv = state_dict[prefix + 'qkv.weight']  # [3d, d]
        W_o = state_dict[prefix + 'proj.weight']   # [d, d]
        
        W_q, W_k, W_v = W_qkv.chunk(3, dim=0)
        print(f"  Loaded layer {layer_idx} weights successfully")
    else:
        print("\nUsing random weights (no checkpoint provided)")
        W_q = torch.randn(d, d) * 0.02
        W_k = torch.randn(d, d) * 0.02
        W_v = torch.randn(d, d) * 0.02
        W_o = torch.randn(d, d) * 0.02
    
    print(f"\n  W^Q shape: {list(W_q.shape)}")
    print(f"  W^K shape: {list(W_k.shape)}")
    print(f"  W^V shape: {list(W_v.shape)}")
    print(f"  W^O shape: {list(W_o.shape)}")
    
    # Create random input (simulating one frame from video)
    X = torch.randn(B, N, d)
    print(f"\n  Input X shape: {list(X.shape)}")
    
    # Step-by-step computation with shapes
    print("\n" + "-" * 40)
    print("  Step-by-step computation:")
    print("-" * 40)
    
    print("\n  [Step 2] Computing Q, K, V...")
    Q = X @ W_q.T  # [B, N, d]
    K = X @ W_k.T
    V = X @ W_v.T
    print(f"    Q = X @ W^Q.T : {list(X.shape)} @ {list(W_q.T.shape)} = {list(Q.shape)}")
    print(f"    K = X @ W^K.T : {list(X.shape)} @ {list(W_k.T.shape)} = {list(K.shape)}")
    print(f"    V = X @ W^V.T : {list(X.shape)} @ {list(W_v.T.shape)} = {list(V.shape)}")
    
    print("\n  [Step 3] Reshaping for multi-head attention...")
    Q_heads = Q.view(B, N, H, d_k).transpose(1, 2)  # [B, H, N, d_k]
    K_heads = K.view(B, N, H, d_k).transpose(1, 2)
    V_heads = V.view(B, N, H, d_k).transpose(1, 2)
    print(f"    Q.view({B}, {N}, {H}, {d_k}).transpose(1, 2)")
    print(f"    Q_heads: {list(Q_heads.shape)}")
    print(f"    K_heads: {list(K_heads.shape)}")
    print(f"    V_heads: {list(V_heads.shape)}")
    
    print("\n  [Step 4] Computing attention scores...")
    scores = torch.matmul(Q_heads, K_heads.transpose(-2, -1)) / (d_k ** 0.5)
    print(f"    scores = Q_heads @ K_heads.T / √{d_k}")
    print(f"           = {list(Q_heads.shape)} @ {list(K_heads.transpose(-2,-1).shape)} / {d_k**0.5:.2f}")
    print(f"           = {list(scores.shape)}")
    
    print("\n  [Step 5] Applying softmax...")
    A = F.softmax(scores, dim=-1)
    print(f"    A = softmax(scores, dim=-1)")
    print(f"    A shape: {list(A.shape)}")
    
    # Show one attention matrix
    print(f"\n    Sample attention values (head 0, first 5×5):")
    print(f"    {A[0, 0, :5, :5].numpy().round(3)}")
    
    # Average across heads for QLSA
    A_avg = A.mean(dim=1)[0]  # [N, N]
    print(f"\n  [QLSA] A averaged across heads: {list(A_avg.shape)}")
    
    # Verify row sums
    row_sums = A_avg.sum(dim=-1)
    print(f"    Row sums (should be 1.0): min={row_sums.min():.6f}, max={row_sums.max():.6f}")
    
    print("\n  [Step 6] Computing weighted values...")
    head_out = torch.matmul(A, V_heads)  # [B, H, N, d_k]
    print(f"    head_out = A @ V_heads")
    print(f"             = {list(A.shape)} @ {list(V_heads.shape)}")
    print(f"             = {list(head_out.shape)}")
    
    print("\n  [Step 7] Concatenating heads...")
    concat = head_out.transpose(1, 2).contiguous().view(B, N, d)
    print(f"    concat = head_out.transpose(1,2).view({B}, {N}, {d})")
    print(f"    concat shape: {list(concat.shape)}")
    
    print("\n  [Step 8] Output projection...")
    attn_out = concat @ W_o.T
    print(f"    attn_out = concat @ W^O.T")
    print(f"             = {list(concat.shape)} @ {list(W_o.T.shape)}")
    print(f"             = {list(attn_out.shape)}")
    
    print("\n  [Step 9] Residual connection...")
    X_out = X + attn_out
    print(f"    X_out = X + attn_out")
    print(f"          = {list(X.shape)} + {list(attn_out.shape)}")
    print(f"          = {list(X_out.shape)}")
    
    # Build M matrix demo
    print("\n" + "=" * 80)
    print("  BUILDING BLOCK MATRIX M (4-layer demo using single A)")
    print("=" * 80)
    
    L_demo = 4
    M_size = L_demo * N
    M = torch.zeros(M_size, M_size)
    
    for i in range(L_demo):
        # Identity on diagonal
        M[i*N:(i+1)*N, i*N:(i+1)*N] = torch.eye(N)
        # -A on sub-diagonal
        if i > 0:
            M[i*N:(i+1)*N, (i-1)*N:i*N] = -A_avg
    
    print(f"\n  M construction:")
    print(f"    For each layer ℓ in [0, {L_demo-1}]:")
    print(f"      M[ℓ*{N}:(ℓ+1)*{N}, ℓ*{N}:(ℓ+1)*{N}] = I_{N}")
    print(f"      M[ℓ*{N}:(ℓ+1)*{N}, (ℓ-1)*{N}:ℓ*{N}] = -A_ℓ  (if ℓ > 0)")
    print(f"\n  M shape: {list(M.shape)}")
    
    # Analyze M
    M_np = M.numpy()
    cond = np.linalg.cond(M_np)
    sparsity = (M_np == 0).sum() / M_np.size * 100
    nnz_per_row = (M_np != 0).sum(axis=1).max()
    
    print(f"\n  M Properties:")
    print(f"  ├── Condition number κ: {cond:.2f}")
    print(f"  ├── Sparsity: {sparsity:.2f}%")
    print(f"  ├── Max nnz per row (s): {nnz_per_row}")
    print(f"  │")
    print(f"  └── QLSA complexity ≈ O(log({M_size}) × {cond:.0f} × {nnz_per_row})")
    print(f"                       ≈ {np.log2(M_size) * cond * nnz_per_row:.2e}")
    
    # Save outputs
    print("\n  Saving matrices...")
    torch.save({
        'A_single_layer': A_avg,
        'A_all_heads': A[0],  # [H, N, N]
        'M_demo': M,
        'config': CONFIG,
        'W_q': W_q,
        'W_k': W_k,
        'W_v': W_v,
        'W_o': W_o,
    }, 'attention_matrices.pt')
    print("  Saved to: attention_matrices.pt")
    
    return A_avg, M


def main():
    parser = argparse.ArgumentParser(description='Transformer attention matrix dimension derivation')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to ViT checkpoint (.pt or .ckpt)')
    parser.add_argument('--layer', type=int, default=0,
                        help='Which layer to analyze (0-11)')
    args = parser.parse_args()
    
    # Always show the dimension derivation
    derive_dimensions_only()
    
    # If PyTorch is available, run actual computation
    if HAS_TORCH:
        compute_with_torch(args.checkpoint, args.layer)
    else:
        print("\n" + "=" * 80)
        print("  PyTorch not found - showing dimensions only")
        print("=" * 80)
        print("""
    To run with actual computation, copy this script to your Windows machine:
    
    conda activate umi
    python attention_matrix_derivation.py --checkpoint vit_encoder_only.pt
    
    Or on your Linux machine:
    
    conda activate umi
    python attention_matrix_derivation.py --checkpoint /path/to/latest.ckpt --layer 0
        """)


if __name__ == "__main__":
    main()