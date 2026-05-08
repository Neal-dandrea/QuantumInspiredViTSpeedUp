#!/usr/bin/env python3
"""
Toy 3-head, 4-layer Transformer Attention Visualization

This script creates a minimal transformer and visualizes:
1. The attention matrices (softmax(QK^T/√d)) at each head and layer
2. How information flows through the layers
3. The block matrix M structure for QLSA

Answers the key questions:
(a) Does update for single head depend on ALL heads from time t? 
    → Within layer: NO (parallel). Across layers: YES (via residual).
(b) Is computational structure constant across layers?
    → YES - same pattern, different weights.
(c) How to construct M?
    → Block-bidiagonal from stacked recurrence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

torch.manual_seed(42)


class ToyMultiHeadAttention(nn.Module):
    """Minimal multi-head attention that stores attention matrices for visualization."""
    
    def __init__(self, d_model=64, num_heads=3, d_k=16):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_model = d_model
        
        # Separate W^Q, W^K, W^V for each head (to show independence)
        self.W_Q = nn.ModuleList([nn.Linear(d_model, d_k, bias=False) for _ in range(num_heads)])
        self.W_K = nn.ModuleList([nn.Linear(d_model, d_k, bias=False) for _ in range(num_heads)])
        self.W_V = nn.ModuleList([nn.Linear(d_model, d_k, bias=False) for _ in range(num_heads)])
        self.W_O = nn.Linear(num_heads * d_k, d_model, bias=False)
        
        # Storage for attention matrices (for visualization)
        self.attention_matrices = []
        
    def forward(self, x, mask=None):
        """
        x: [batch, seq_len, d_model]
        Returns: [batch, seq_len, d_model], stores attention matrices
        """
        batch_size, seq_len, _ = x.shape
        self.attention_matrices = []
        
        head_outputs = []
        for h in range(self.num_heads):
            # Each head computes INDEPENDENTLY
            Q = self.W_Q[h](x)  # [batch, seq_len, d_k]
            K = self.W_K[h](x)  # [batch, seq_len, d_k]
            V = self.W_V[h](x)  # [batch, seq_len, d_k]
            
            # Attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
            
            # Causal mask (lower triangular)
            if mask is None:
                mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
                scores = scores.masked_fill(mask.to(x.device), float('-inf'))
            
            # Softmax to get attention weights
            attn = F.softmax(scores, dim=-1)  # [batch, seq_len, seq_len]
            self.attention_matrices.append(attn.detach())
            
            # Apply attention to values
            head_out = torch.matmul(attn, V)  # [batch, seq_len, d_k]
            head_outputs.append(head_out)
        
        # Concatenate heads and project
        concat = torch.cat(head_outputs, dim=-1)  # [batch, seq_len, num_heads * d_k]
        output = self.W_O(concat)  # [batch, seq_len, d_model]
        
        return output


class ToyTransformerBlock(nn.Module):
    """Single transformer block with attention + FFN."""
    
    def __init__(self, d_model=64, num_heads=3, d_ff=128):
        super().__init__()
        self.attention = ToyMultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, x):
        # Pre-norm architecture (like modern transformers)
        normed = self.norm1(x)
        attn_out = self.attention(normed)
        x = x + attn_out  # Residual connection
        
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out  # Residual connection
        
        return x


class ToyTransformer(nn.Module):
    """4-layer, 3-head toy transformer for visualization."""
    
    def __init__(self, vocab_size=100, d_model=64, num_layers=4, num_heads=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(512, d_model)
        self.layers = nn.ModuleList([
            ToyTransformerBlock(d_model, num_heads) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """x: [batch, seq_len] token ids"""
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        # Embed tokens + positions
        h = self.embedding(x) + self.pos_embedding(positions)
        
        # Store intermediate representations
        self.layer_outputs = [h.detach().clone()]
        
        # Pass through layers
        for layer in self.layers:
            h = layer(h)
            self.layer_outputs.append(h.detach().clone())
            
        return self.final_norm(h)
    
    def get_all_attention_matrices(self):
        """Get attention matrices from all layers and heads."""
        all_attn = []
        for layer in self.layers:
            all_attn.append(layer.attention.attention_matrices)
        return all_attn  # [num_layers][num_heads] each is [batch, seq, seq]


def visualize_attention_matrices(model, tokens, token_labels):
    """Create comprehensive visualization of attention at each head and layer."""
    
    # Forward pass
    with torch.no_grad():
        output = model(tokens)
    
    # Get all attention matrices
    all_attn = model.get_all_attention_matrices()
    num_layers = len(all_attn)
    num_heads = len(all_attn[0])
    
    # Create figure with subplots for each head at each layer
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(num_layers + 2, num_heads + 1, figure=fig, 
                  height_ratios=[0.5] + [1]*num_layers + [0.8],
                  width_ratios=[0.15] + [1]*num_heads)
    
    # Title
    fig.suptitle('3-Head, 4-Layer Transformer Attention Visualization\n'
                 'Each cell shows softmax(QK^T/√d_k) - the attention matrix A',
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Column headers (heads)
    for h in range(num_heads):
        ax = fig.add_subplot(gs[0, h+1])
        ax.text(0.5, 0.5, f'Head {h+1}\n(W^Q_{h+1}, W^K_{h+1}, W^V_{h+1})', 
                ha='center', va='center', fontsize=11, fontweight='bold')
        ax.axis('off')
    
    # Row labels and attention heatmaps
    for layer_idx in range(num_layers):
        # Row label
        ax_label = fig.add_subplot(gs[layer_idx + 1, 0])
        ax_label.text(0.5, 0.5, f'Layer {layer_idx + 1}\n(X_{layer_idx} → X_{layer_idx + 1})', 
                      ha='center', va='center', fontsize=10, fontweight='bold', rotation=0)
        ax_label.axis('off')
        
        for head_idx in range(num_heads):
            ax = fig.add_subplot(gs[layer_idx + 1, head_idx + 1])
            
            # Get attention matrix for this layer and head
            attn = all_attn[layer_idx][head_idx][0].numpy()  # [seq_len, seq_len]
            
            # Plot heatmap
            im = ax.imshow(attn, cmap='YlOrRd', vmin=0, vmax=1, aspect='equal')
            
            # Add text annotations
            for i in range(len(token_labels)):
                for j in range(len(token_labels)):
                    val = attn[i, j]
                    color = 'white' if val > 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                            fontsize=8, color=color)
            
            # Labels
            ax.set_xticks(range(len(token_labels)))
            ax.set_yticks(range(len(token_labels)))
            ax.set_xticklabels(token_labels, fontsize=9)
            ax.set_yticklabels(token_labels, fontsize=9)
            
            if layer_idx == num_layers - 1:
                ax.set_xlabel('Key (attending to)', fontsize=9)
            if head_idx == 0:
                ax.set_ylabel('Query (from)', fontsize=9)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.3, 0.015, 0.4])
    fig.colorbar(im, cax=cbar_ax, label='Attention Weight')
    
    # Add explanation panel at bottom
    ax_explain = fig.add_subplot(gs[-1, :])
    ax_explain.axis('off')
    explanation = """
    KEY OBSERVATIONS FOR YOUR QLSA FORMULATION:
    
    (a) Head Independence: Each head (column) computes its own Q, K, V projections independently.
        Heads do NOT see each other within a layer. But the OUTPUT of all heads is concatenated
        and projected by W^O, so the NEXT layer sees the combined result.
    
    (b) Structural Constancy: Every layer (row) follows the same pattern:
        X_ℓ = X_{ℓ-1} + Attention(LayerNorm(X_{ℓ-1})) + FFN(LayerNorm(...))
        The A_ℓ matrices differ (different learned weights), but the STRUCTURE is identical.
    
    (c) For M construction: Each A_ℓ shown here is the COMBINED attention across all heads.
        M is block-bidiagonal: M = [[I, 0, 0, 0], [-A₂, I, 0, 0], [0, -A₃, I, 0], [0, 0, -A₄, I]]
    """
    ax_explain.text(0.02, 0.95, explanation, transform=ax_explain.transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    return fig


def visualize_block_matrix_M(model, tokens):
    """Visualize the block matrix M structure for QLSA."""
    
    with torch.no_grad():
        output = model(tokens)
    
    all_attn = model.get_all_attention_matrices()
    num_layers = len(all_attn)
    num_heads = len(all_attn[0])
    seq_len = tokens.shape[1]
    
    # Compute combined attention matrix per layer (average across heads for visualization)
    # In reality, it's more complex due to W^O projection
    combined_A = []
    for layer_idx in range(num_layers):
        # Average attention across heads (simplified view)
        layer_attn = torch.stack(all_attn[layer_idx]).mean(dim=0)[0].numpy()
        combined_A.append(layer_attn)
    
    # Build block matrix M
    block_size = seq_len
    M_size = num_layers * block_size
    M = np.zeros((M_size, M_size))
    
    # Fill in the blocks
    for i in range(num_layers):
        row_start = i * block_size
        row_end = (i + 1) * block_size
        
        # Identity on diagonal
        M[row_start:row_end, row_start:row_end] = np.eye(block_size)
        
        # -A on sub-diagonal (except first row)
        if i > 0:
            col_start = (i - 1) * block_size
            col_end = i * block_size
            M[row_start:row_end, col_start:col_end] = -combined_A[i]
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Full M matrix
    ax1 = axes[0]
    im1 = ax1.imshow(M, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ax1.set_title('Block Matrix M for QLSA\nM · X = b', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Column blocks (X₁, X₂, X₃, X₄)')
    ax1.set_ylabel('Row blocks (equations for X₁, X₂, X₃, X₄)')
    
    # Add block boundaries
    for i in range(1, num_layers):
        ax1.axhline(y=i * block_size - 0.5, color='black', linewidth=2)
        ax1.axvline(x=i * block_size - 0.5, color='black', linewidth=2)
    
    # Label blocks
    for i in range(num_layers):
        for j in range(num_layers):
            center_y = i * block_size + block_size / 2
            center_x = j * block_size + block_size / 2
            if i == j:
                ax1.text(center_x, center_y, 'I', ha='center', va='center', 
                        fontsize=14, fontweight='bold', color='blue')
            elif i == j + 1:
                ax1.text(center_x, center_y, f'-A_{i+1}', ha='center', va='center', 
                        fontsize=12, fontweight='bold', color='red')
            else:
                ax1.text(center_x, center_y, '0', ha='center', va='center', 
                        fontsize=12, color='gray')
    
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # Show one A matrix in detail
    ax2 = axes[1]
    im2 = ax2.imshow(combined_A[1], cmap='YlOrRd', vmin=0, vmax=1)
    ax2.set_title('Example: A₂ (Layer 2 attention)\nAveraged across 3 heads', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Key token position')
    ax2.set_ylabel('Query token position')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    plt.tight_layout()
    return fig


def main():
    print("=" * 70)
    print("TOY TRANSFORMER VISUALIZATION")
    print("3 Attention Heads × 4 Layers")
    print("=" * 70)
    
    # Create model
    model = ToyTransformer(vocab_size=100, d_model=64, num_layers=4, num_heads=3)
    model.eval()
    
    # Create toy input: "The cat sat on" (4 tokens)
    tokens = torch.tensor([[10, 25, 42, 67]])  # Random token IDs
    token_labels = ['The', 'cat', 'sat', 'on']
    
    print(f"\nInput tokens: {token_labels}")
    print(f"Token IDs: {tokens[0].tolist()}")
    print(f"Model: d_model=64, d_k=16, num_heads=3, num_layers=4")
    
    # Generate visualizations
    print("\nGenerating attention visualization...")
    fig1 = visualize_attention_matrices(model, tokens, token_labels)
    fig1.savefig('attention_all_heads_layers.png', dpi=150, bbox_inches='tight')
    print("Saved: attention_all_heads_layers.png")
    
    print("\nGenerating block matrix M visualization...")
    fig2 = visualize_block_matrix_M(model, tokens)
    fig2.savefig('block_matrix_M.png', dpi=150, bbox_inches='tight')
    print("Saved: block_matrix_M.png")
    
    # Print answers to questions
    print("\n" + "=" * 70)
    print("ANSWERS TO YOUR QUESTIONS")
    print("=" * 70)
    
    print("""
(a) Does the update for a single attention head depend on vectors 
    from ALL attention heads at time t?
    
    ANSWER: 
    - WITHIN a layer: NO. Each head computes Q_h = X @ W^Q_h, K_h = X @ W^K_h, 
      V_h = X @ W^V_h independently. Heads don't see each other.
    - ACROSS layers: YES. All heads' outputs are concatenated → W^O → added to 
      residual → becomes input X_{ℓ+1} for next layer. So layer ℓ+1 sees the 
      combined effect of all heads from layer ℓ.

(b) Is the computational structure constant across all layers?
    
    ANSWER: YES. Every layer follows:
      t¹ = LayerNorm(X)
      t² = MultiHeadAttention(t¹)   ← Same structure, different W^Q, W^K, W^V, W^O
      t³ = t² + X                    ← Residual
      t⁴ = LayerNorm(t³)
      t⁵ = FFN(t⁴)                  ← Same structure, different W₁, W₂
      X_out = t⁵ + t³               ← Residual
    
    The PATTERN is constant → block-bidiagonal M structure works.
    The WEIGHTS differ → each A_ℓ is different.

(c) How can we construct matrix M?
    
    ANSWER: Stack the recurrence X_ℓ = A_ℓ · X_{ℓ-1} + C_ℓ:
    
    | I    0    0    0  |   | X₁ |   | A₁·X₀ + C₁ |
    | -A₂  I    0    0  | × | X₂ | = |    C₂      |
    | 0   -A₃   I    0  |   | X₃ |   |    C₃      |
    | 0    0   -A₄   I  |   | X₄ |   |    C₄      |
    
    For your ViT: M is (12 layers × 197 tokens) × (12 × 197) = 2364 × 2364
    """)
    
    plt.show()


if __name__ == "__main__":
    main()