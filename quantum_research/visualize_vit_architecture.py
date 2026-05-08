#!/usr/bin/env python3
"""
Visualize ViT-B/16 architecture using torchinfo.
Shows layer-by-layer breakdown with shapes and parameters.
"""

import torch
import torch.nn as nn
from torchinfo import summary

# Simple ViT block implementation (matches your architecture)
class Attention(nn.Module):
    def __init__(self, dim=768, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        # Combined QKV projection
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, N, C = x.shape  # [1, 197, 768]
        
        # QKV projection
        qkv = self.qkv(x)  # [1, 197, 2304]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, 1, 12, 197, 64]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class MLP(nn.Module):
    def __init__(self, dim=768, hidden_dim=3072):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim=768, num_heads=12):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_dim=3072)
        
    def forward(self, x):
        # Attention block with residual
        x = x + self.attn(self.norm1(x))
        # MLP block with residual
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
    ):
        super().__init__()
        
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2  # 196
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # CLS token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, 768, 14, 14]
        x = x.flatten(2).transpose(1, 2)  # [B, 196, 768]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 197, 768]
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = self.norm(x)
        x = x[:, 0]  # Take CLS token
        x = self.head(x)
        
        return x


def visualize_vit():
    """Visualize the ViT architecture with torchinfo."""
    
    print("="*80)
    print("ViT-B/16 ARCHITECTURE VISUALIZATION")
    print("="*80)
    print()
    
    # Create model
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
    )
    
    # Input shape
    batch_size = 1
    input_size = (batch_size, 3, 224, 224)
    
    print("Input: RGB image [1, 3, 224, 224]")
    print()
    
    # Summary with detailed information
    summary(
        model,
        input_size=input_size,
        col_names=[
            "input_size",
            "output_size", 
            "num_params",
            "kernel_size",
            "mult_adds"
        ],
        depth=4,  # Show nested layers
        verbose=1,
        row_settings=["var_names"]
    )
    
    print()
    print("="*80)
    print("KEY OBSERVATIONS")
    print("="*80)
    print()
    print("1. Patch Embedding:")
    print("   - Conv2d: [1, 3, 224, 224] → [1, 768, 14, 14]")
    print("   - Flatten + transpose: [1, 768, 14, 14] → [1, 196, 768]")
    print("   - Add CLS token: [1, 196, 768] → [1, 197, 768]")
    print()
    print("2. Each Transformer Block:")
    print("   - Input: [1, 197, 768]")
    print("   - Attention: QKV projection (768 → 2304), then split into Q, K, V")
    print("   - Multi-head: 12 heads × 64 dims = 768 dims")
    print("   - MLP: 768 → 3072 → 768 (4x expansion)")
    print("   - Output: [1, 197, 768] (same shape due to residuals)")
    print()
    print("3. Total Parameters:")
    print("   - Attention per block: ~2.36M (QKV + proj)")
    print("   - MLP per block: ~7.08M (fc1 + fc2)")
    print("   - 12 blocks × ~9.44M ≈ 113M total")
    print()
    print("4. Attention Matrix A:")
    print("   - Computed at each block: [12, 197, 197] per head")
    print("   - After averaging heads: [197, 197]")
    print("   - This is what we want to approximate as constant!")
    print()


def show_single_block():
    """Show detailed view of a single transformer block."""
    
    print()
    print("="*80)
    print("SINGLE TRANSFORMER BLOCK - DETAILED VIEW")
    print("="*80)
    print()
    
    block = TransformerBlock(dim=768, num_heads=12)
    
    summary(
        block,
        input_size=(1, 197, 768),
        col_names=[
            "input_size",
            "output_size",
            "num_params",
            "trainable"
        ],
        depth=5,
        verbose=1
    )
    
    print()
    print("Layer-by-layer breakdown:")
    print()
    print("Input:           [1, 197, 768]")
    print("  ↓")
    print("LayerNorm:       [1, 197, 768] (normalize)")
    print("  ↓")
    print("QKV Linear:      [1, 197, 768] → [1, 197, 2304]")
    print("  ↓")
    print("Reshape:         [1, 197, 2304] → [1, 12, 197, 64] (Q, K, V separately)")
    print("  ↓")
    print("Attention:       Q@K^T → [1, 12, 197, 197]")
    print("                 softmax → A [1, 12, 197, 197]  ← THIS IS A!")
    print("  ↓")
    print("Apply to V:      A @ V → [1, 12, 197, 64]")
    print("  ↓")
    print("Concat heads:    [1, 12, 197, 64] → [1, 197, 768]")
    print("  ↓")
    print("Output proj:     [1, 197, 768] → [1, 197, 768]")
    print("  ↓")
    print("Residual:        + input → [1, 197, 768]")
    print("  ↓")
    print("LayerNorm:       [1, 197, 768]")
    print("  ↓")
    print("MLP (fc1):       [1, 197, 768] → [1, 197, 3072]")
    print("  ↓")
    print("GELU:            [1, 197, 3072]")
    print("  ↓")
    print("MLP (fc2):       [1, 197, 3072] → [1, 197, 768]")
    print("  ↓")
    print("Residual:        + input → [1, 197, 768]")
    print("  ↓")
    print("Output:          [1, 197, 768]")
    print()


def show_attention_only():
    """Show just the attention mechanism."""
    
    print()
    print("="*80)
    print("ATTENTION MECHANISM - ISOLATED VIEW")
    print("="*80)
    print()
    
    attn = Attention(dim=768, num_heads=12)
    
    summary(
        attn,
        input_size=(1, 197, 768),
        col_names=[
            "input_size",
            "output_size",
            "num_params",
        ],
        depth=3,
        verbose=1
    )
    
    print()
    print("Parameter breakdown:")
    print("  - qkv.weight: [2304, 768] = 1,769,472 params")
    print("  - qkv.bias:   [2304]      = 2,304 params")
    print("  - proj.weight: [768, 768] = 589,824 params")
    print("  - proj.bias:   [768]      = 768 params")
    print("  - Total:                    2,362,368 params")
    print()
    print("This is what QViT would replace with ~48 quantum parameters!")
    print()


if __name__ == "__main__":
    # Install torchinfo if not available
    try:
        import torchinfo
    except ImportError:
        print("Installing torchinfo...")
        import subprocess
        subprocess.check_call(["pip", "install", "torchinfo", "--break-system-packages"])
        print()
    
    # Run visualizations
    visualize_vit()
    show_single_block()
    show_attention_only()
    
    print("="*80)
    print("DONE! You can now see the full architecture breakdown.")
    print("="*80)