#!/usr/bin/env python3
"""
Test QLSA approximation with REAL UMI video frames.
Shows error accumulation across all 12 transformer layers.
NOW INCLUDES FULL TRANSFORMER BLOCK: Attention + FFN!

Usage:
    python test_qlsa_approximation_umi.py --checkpoint vit_encoder_only.pt --video_path path/to/GX010460.MP4
"""

import torch
import torch.nn as nn
import argparse
import numpy as np
from pathlib import Path
import cv2
from torchvision import transforms


def load_vit_model(checkpoint_path):
    """Load ViT model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    if checkpoint_path.endswith('.ckpt'):
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = ckpt['state_dicts']['model']
        vit_state = {k.replace('obs_encoder.model.', ''): v 
                     for k, v in state_dict.items() 
                     if k.startswith('obs_encoder.model.')}
    else:
        vit_state = torch.load(checkpoint_path, map_location='cpu')
    
    return vit_state


def load_umi_video_frames(video_path, num_frames=100, start_frame=0):
    """
    Load frames from UMI video.
    
    Args:
        video_path: Path to MP4 file
        num_frames: Number of frames to extract
        start_frame: Starting frame index
    
    Returns:
        frames: List of frames as tensors [3, 224, 224]
    """
    print(f"\nLoading {num_frames} frames from video: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")
    
    # Set starting position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Image preprocessing (same as ViT training)
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
    ])
    
    frames = []
    frame_idx = start_frame
    
    while len(frames) < num_frames and frame_idx < total_frames:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        frame_tensor = preprocess(frame_rgb)  # [3, 224, 224]
        frames.append(frame_tensor)
        
        frame_idx += 1
        
        if len(frames) % 20 == 0:
            print(f"  Loaded {len(frames)}/{num_frames} frames...")
    
    cap.release()
    
    print(f"✓ Loaded {len(frames)} frames from video")
    
    return frames


def patch_embed(frames, state_dict):
    """
    Apply patch embedding to frames.
    
    Args:
        frames: List of [3, 224, 224] tensors
        state_dict: Model weights
    
    Returns:
        embeddings: [num_frames, 197, 768]
    """
    print("\nApplying patch embedding...")
    
    # Get patch embedding weights
    patch_weight = state_dict['patch_embed.proj.weight']  # [768, 3, 16, 16]
    patch_bias = state_dict.get('patch_embed.proj.bias', None)  # [768] or None
    
    # Get CLS token and positional embedding
    cls_token = state_dict['cls_token']        # [1, 1, 768]
    pos_embed = state_dict['pos_embed']        # [1, 197, 768]
    
    embeddings = []
    
    for i, frame in enumerate(frames):
        # Apply conv (patch embedding)
        x = frame.unsqueeze(0)  # [1, 3, 224, 224]
        x = nn.functional.conv2d(x, patch_weight, patch_bias, stride=16)  # [1, 768, 14, 14]
        
        # Flatten
        x = x.flatten(2).transpose(1, 2)  # [1, 196, 768]
        
        # Add CLS token
        x = torch.cat([cls_token, x], dim=1)  # [1, 197, 768]
        
        # Add positional embedding
        x = x + pos_embed  # [1, 197, 768]
        
        embeddings.append(x.squeeze(0))  # [197, 768]
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(frames)} frames...")
    
    embeddings = torch.stack(embeddings)  # [num_frames, 197, 768]
    
    print(f"✓ Patch embedding complete: {embeddings.shape}")
    
    return embeddings


def extract_attention_and_output(X, layer_state_dict, layer_idx):
    """
    Compute attention matrix A and FULL output X for a single layer.
    NOW INCLUDES FFN!
    
    Returns:
        A: [H, N, N] attention matrix (averaged across batch)
        X_out: [N, d] output after FULL transformer block (attention + FFN)
    """
    N, d = X.shape
    H = 12
    d_k = d // H
    
    X = X.unsqueeze(0)  # [1, N, d]
    
    # Extract weights for ATTENTION
    qkv_weight = layer_state_dict[f'blocks.{layer_idx}.attn.qkv.weight']
    qkv_bias = layer_state_dict[f'blocks.{layer_idx}.attn.qkv.bias']
    proj_weight = layer_state_dict[f'blocks.{layer_idx}.attn.proj.weight']
    proj_bias = layer_state_dict[f'blocks.{layer_idx}.attn.proj.bias']
    norm1_weight = layer_state_dict[f'blocks.{layer_idx}.norm1.weight']
    norm1_bias = layer_state_dict[f'blocks.{layer_idx}.norm1.bias']
    
    # Extract weights for FFN
    norm2_weight = layer_state_dict[f'blocks.{layer_idx}.norm2.weight']
    norm2_bias = layer_state_dict[f'blocks.{layer_idx}.norm2.bias']
    mlp_fc1_weight = layer_state_dict[f'blocks.{layer_idx}.mlp.fc1.weight']
    mlp_fc1_bias = layer_state_dict[f'blocks.{layer_idx}.mlp.fc1.bias']
    mlp_fc2_weight = layer_state_dict[f'blocks.{layer_idx}.mlp.fc2.weight']
    mlp_fc2_bias = layer_state_dict[f'blocks.{layer_idx}.mlp.fc2.bias']
    
    # ============ ATTENTION BLOCK ============
    # LayerNorm 1
    X_norm = nn.functional.layer_norm(X, (d,), norm1_weight, norm1_bias)
    
    # QKV projection
    qkv = nn.functional.linear(X_norm, qkv_weight, qkv_bias)
    qkv = qkv.reshape(1, N, 3, H, d_k).permute(2, 0, 3, 1, 4)
    Q, K, V = qkv[0], qkv[1], qkv[2]  # [1, H, N, d_k]
    
    # Attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
    A = torch.softmax(scores, dim=-1)  # [1, H, N, N]
    
    # Apply attention
    attn_out = torch.matmul(A, V)  # [1, H, N, d_k]
    attn_out = attn_out.transpose(1, 2).reshape(1, N, d)
    attn_out = nn.functional.linear(attn_out, proj_weight, proj_bias)
    
    # Residual connection 1
    X_attn = X + attn_out
    
    # ============ FFN BLOCK ============
    # LayerNorm 2
    X_norm2 = nn.functional.layer_norm(X_attn, (d,), norm2_weight, norm2_bias)
    
    # MLP fc1 (768 -> 3072)
    X_mlp = nn.functional.linear(X_norm2, mlp_fc1_weight, mlp_fc1_bias)
    
    # GELU activation
    X_mlp = nn.functional.gelu(X_mlp)
    
    # MLP fc2 (3072 -> 768)
    X_mlp = nn.functional.linear(X_mlp, mlp_fc2_weight, mlp_fc2_bias)
    
    # Residual connection 2
    X_out = X_attn + X_mlp
    
    return A.squeeze(0), X_out.squeeze(0)  # [H, N, N], [N, d]


def test_multi_layer_approximation(checkpoint_path, video_path=None, num_frames=100):
    """
    Test approximation error accumulation across all 12 layers.
    
    If video_path is None, uses random noise (original test).
    If video_path is provided, uses real UMI video frames.
    """
    print("\n" + "="*80)
    print("MULTI-LAYER QLSA APPROXIMATION TEST")
    print("="*80)
    
    # Load model
    state_dict = load_vit_model(checkpoint_path)
    
    # Get inputs
    if video_path is None:
        print("\n⚠️  Using RANDOM NOISE (no video provided)")
        print("   This is worst-case scenario - no structure\n")
        # Random inputs
        X_inputs = [torch.randn(197, 768) for _ in range(num_frames)]
    else:
        print(f"\n✓ Using REAL UMI VIDEO: {video_path}\n")
        # Load video frames
        frames = load_umi_video_frames(video_path, num_frames)
        # Get patch embeddings
        X_inputs = patch_embed(frames, state_dict)
        X_inputs = [X_inputs[i] for i in range(len(X_inputs))]
    
    num_layers = 12
    
    # Storage for averaged attention at each layer
    A_avg_all_layers = []
    C_avg_all_layers = []
    
    print("\n" + "="*80)
    print("PHASE 1: Computing averaged attention matrices from training data")
    print("="*80)
    
    # Compute averaged A for each layer
    for layer_idx in range(num_layers):
        print(f"\nLayer {layer_idx}:")
        
        all_A = []
        all_attn_output = []
        
        # Process all frames through this layer
        X_current = X_inputs.copy()
        
        # First, need to get X at this layer for all frames
        for frame_idx in range(num_frames):
            X_in = X_current[frame_idx]
            
            # Propagate through previous layers to get input to this layer
            X_layer_in = X_in
            for prev_layer in range(layer_idx):
                _, X_layer_in = extract_attention_and_output(X_layer_in, state_dict, prev_layer)
            
            # Now compute A at this layer
            A, X_out = extract_attention_and_output(X_layer_in, state_dict, layer_idx)
            
            # Average across heads
            A_avg_heads = A.mean(dim=0)  # [N, N]
            all_A.append(A_avg_heads)
            
            # Store attention output (for C_avg)
            attn_output = X_out - X_layer_in  # Remove residual
            all_attn_output.append(attn_output)
        
        # Average across all frames
        A_avg = torch.stack(all_A).mean(dim=0)  # [N, N]
        C_avg = torch.stack(all_attn_output).mean(dim=0)  # [N, d]
        
        A_avg_all_layers.append(A_avg)
        C_avg_all_layers.append(C_avg)
        
        print(f"  ✓ A_avg shape: {A_avg.shape}, row sums: [{A_avg.sum(dim=1).min():.4f}, {A_avg.sum(dim=1).max():.4f}]")
        print(f"  ✓ C_avg shape: {C_avg.shape}, norm: {torch.norm(C_avg):.4f}")
    
    print("\n" + "="*80)
    print("PHASE 2: Testing approximation error at each layer")
    print("="*80)
    
    # Test error at each layer
    layer_errors = []
    
    for layer_idx in range(num_layers):
        print(f"\n{'─'*80}")
        print(f"Layer {layer_idx} Error Analysis")
        print(f"{'─'*80}")
        
        errors = []
        
        for frame_idx in range(min(20, num_frames)):  # Test on first 20 frames
            # Get input to this layer
            X_in = X_inputs[frame_idx]
            
            # Propagate through previous layers (TRUE path)
            X_true = X_in
            for prev_layer in range(layer_idx):
                _, X_true = extract_attention_and_output(X_true, state_dict, prev_layer)
            
            # Compute TRUE output at this layer
            _, X_out_true = extract_attention_and_output(X_true, state_dict, layer_idx)
            
            # Compute APPROXIMATION output
            X_out_approx = X_true + C_avg_all_layers[layer_idx]
            
            # Error
            error = torch.norm(X_out_true - X_out_approx) / torch.norm(X_out_true)
            errors.append(error.item() * 100)  # Convert to percentage
        
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        layer_errors.append(mean_error)
        
        print(f"Mean Error: {mean_error:.2f}%  (std: {std_error:.2f}%)")
        
        if mean_error < 2:
            print("✓ EXCELLENT - Error < 2%")
        elif mean_error < 5:
            print("✓ GOOD - Error < 5%")
        elif mean_error < 10:
            print("~ ACCEPTABLE - Error < 10%")
        else:
            print("✗ POOR - Error > 10%")
    
    print("\n" + "="*80)
    print("SUMMARY: Error Accumulation Across All Layers")
    print("="*80)
    print()
    print(f"{'Layer':<10} {'Mean Error':<15} {'Cumulative':<15} {'Status':<10}")
    print("─"*60)
    
    cumulative_error = 0
    for layer_idx, error in enumerate(layer_errors):
        cumulative_error += error
        
        if error < 2:
            status = "✓ Excellent"
        elif error < 5:
            status = "✓ Good"
        elif error < 10:
            status = "~ OK"
        else:
            status = "✗ Poor"
        
        print(f"Layer {layer_idx:<3} {error:>6.2f}%        {cumulative_error:>6.2f}%        {status}")
    
    print("─"*60)
    print(f"{'TOTAL':<10} {np.mean(layer_errors):>6.2f}%        {cumulative_error:>6.2f}%")
    print()
    
    # Final assessment
    final_error = layer_errors[-1]
    
    print("="*80)
    print("FINAL ASSESSMENT")
    print("="*80)
    print()
    
    if final_error < 5:
        assessment = "✓ VIABLE"
        explanation = "Error stays low across all layers. QLSA approximation is promising!"
    elif final_error < 10:
        assessment = "~ BORDERLINE"
        explanation = "Error is moderate. May work depending on task tolerance."
    else:
        assessment = "✗ QUESTIONABLE"
        explanation = "Error accumulates significantly. Approximation may be too lossy."
    
    print(f"Final Layer Error: {final_error:.2f}%")
    print(f"Average Layer Error: {np.mean(layer_errors):.2f}%")
    print(f"Cumulative Error: {cumulative_error:.2f}%")
    print()
    print(f"Assessment: {assessment}")
    print(f"{explanation}")
    print()
    
    if video_path:
        print("✓ This was tested on REAL UMI robot video data")
    else:
        print("⚠️  This was tested on RANDOM NOISE - test on real video for conclusive results!")
    
    print()
    print("="*80)
    
    return layer_errors


def main():
    parser = argparse.ArgumentParser(
        description='Test QLSA approximation with real UMI video or random noise'
    )
    parser.add_argument('--checkpoint', type=str, 
                       default=r'C:\Users\neald\Desktop\QuantumInspiredViTSpeedUp\quantum_research\vit_encoder_only.pt',
                       help='Path to ViT checkpoint')
    parser.add_argument('--video_path', type=str, default=None,
                       help='Path to single UMI video file (MP4). If not provided, uses random noise.')
    parser.add_argument('--video_dir', type=str, 
                       default=r'C:\Users\neald\Desktop\QuantumInspiredViTSpeedUp\quantum_research\data_for_quantum_research2',
                       help='Directory containing multiple UMI videos to test')
    parser.add_argument('--test_all', action='store_true',
                       help='Test all videos in video_dir')
    parser.add_argument('--num_frames', type=int, default=200,
                       help='Number of frames to test per video (default: 200)')
    parser.add_argument('--start_frame', type=int, default=0,
                       help='Starting frame index (default: 0)')
    
    args = parser.parse_args()
    
    if not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        return
    
    # Determine which videos to test
    videos_to_test = []
    
    if args.test_all:
        # Test all MP4 files in the directory
        video_dir = Path(args.video_dir)
        if not video_dir.exists():
            print(f"ERROR: Video directory not found: {args.video_dir}")
            return
        
        videos_to_test = sorted(list(video_dir.glob("*.MP4")))
        if not videos_to_test:
            print(f"ERROR: No MP4 files found in {args.video_dir}")
            return
        
        print(f"\n{'='*80}")
        print(f"Found {len(videos_to_test)} videos to test:")
        for v in videos_to_test:
            print(f"  - {v.name}")
        print(f"{'='*80}\n")
    
    elif args.video_path:
        # Test single video
        if not Path(args.video_path).exists():
            print(f"ERROR: Video not found: {args.video_path}")
            return
        videos_to_test = [Path(args.video_path)]
    
    else:
        # No video provided - use random noise
        videos_to_test = [None]
    
    # Store results for all videos
    all_results = []
    
    # Test each video
    for video_idx, video_path in enumerate(videos_to_test):
        if len(videos_to_test) > 1:
            print(f"\n{'#'*80}")
            print(f"# VIDEO {video_idx + 1}/{len(videos_to_test)}: {video_path.name if video_path else 'Random Noise'}")
            print(f"{'#'*80}\n")
        
        # Run test
        layer_errors = test_multi_layer_approximation(
            args.checkpoint,
            str(video_path) if video_path else None,
            args.num_frames
        )
        
        # Store results
        video_name = video_path.name if video_path else "Random Noise"
        all_results.append({
            'name': video_name,
            'path': str(video_path) if video_path else None,
            'layer_errors': layer_errors,
            'avg_error': np.mean(layer_errors),
            'final_error': layer_errors[-1],
            'cumulative_error': np.sum(layer_errors),
            'early_error': np.mean(layer_errors[:4]),  # Layers 0-3
            'late_error': np.mean(layer_errors[8:]),   # Layers 8-11
        })
    
    # If multiple videos, print comparison
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("COMPARISON: ALL VIDEOS")
        print("="*80)
        print()
        
        # Summary table
        print(f"{'Video':<20} {'Avg Error':<12} {'Final Error':<13} {'Cumulative':<13} {'Early (0-3)':<13} {'Late (8-11)':<13}")
        print("─"*90)
        
        for result in all_results:
            name = result['name'][:18]  # Truncate if too long
            print(f"{name:<20} {result['avg_error']:>6.2f}%      "
                  f"{result['final_error']:>6.2f}%        "
                  f"{result['cumulative_error']:>6.2f}%        "
                  f"{result['early_error']:>6.2f}%        "
                  f"{result['late_error']:>6.2f}%")
        
        print("─"*90)
        
        # Overall statistics
        avg_of_avgs = np.mean([r['avg_error'] for r in all_results])
        avg_of_finals = np.mean([r['final_error'] for r in all_results])
        avg_cumulative = np.mean([r['cumulative_error'] for r in all_results])
        
        print(f"{'AVERAGE':<20} {avg_of_avgs:>6.2f}%      "
              f"{avg_of_finals:>6.2f}%        "
              f"{avg_cumulative:>6.2f}%")
        print()
        
        # Best and worst
        best = min(all_results, key=lambda x: x['avg_error'])
        worst = max(all_results, key=lambda x: x['avg_error'])
        
        print(f"Best video:  {best['name']} (avg: {best['avg_error']:.2f}%)")
        print(f"Worst video: {worst['name']} (avg: {worst['avg_error']:.2f}%)")
        print()
        
        # Layer-by-layer comparison
        print("\n" + "="*80)
        print("LAYER-BY-LAYER COMPARISON")
        print("="*80)
        print()
        
        print(f"{'Layer':<8}", end='')
        for result in all_results:
            name = result['name'][:12]
            print(f"{name:<14}", end='')
        print()
        print("─" * (8 + 14 * len(all_results)))
        
        for layer_idx in range(12):
            print(f"Layer {layer_idx:<2}", end='')
            for result in all_results:
                error = result['layer_errors'][layer_idx]
                print(f"{error:>6.2f}%       ", end='')
            print()
        
        print("─" * (8 + 14 * len(all_results)))
        print(f"{'Average':<8}", end='')
        for result in all_results:
            print(f"{result['avg_error']:>6.2f}%       ", end='')
        print()
        print()
        
        # Final assessment across all videos
        print("="*80)
        print("FINAL ASSESSMENT ACROSS ALL VIDEOS")
        print("="*80)
        print()
        
        if avg_of_finals < 5:
            assessment = "✓ VIABLE"
            recommendation = "QLSA approximation shows promise! Proceed to validate on robot control task."
        elif avg_of_finals < 10:
            assessment = "~ BORDERLINE"
            recommendation = "QLSA approximation is questionable. Consider alternative approaches or accept accuracy trade-off."
        else:
            assessment = "✗ NOT VIABLE"
            recommendation = "QLSA approximation has too much error. Recommend pivoting to alternative quantum approaches."
        
        print(f"Average Final Layer Error: {avg_of_finals:.2f}%")
        print(f"Average Error Per Layer: {avg_of_avgs:.2f}%")
        print(f"Average Cumulative Error: {avg_cumulative:.2f}%")
        print()
        print(f"Overall Assessment: {assessment}")
        print(f"Recommendation: {recommendation}")
        print()
        
        # Variance analysis
        error_variance = np.var([r['avg_error'] for r in all_results])
        print(f"Error Variance Across Videos: {error_variance:.2f}")
        if error_variance < 5:
            print("✓ Low variance - approximation is consistent across different robot scenarios")
        else:
            print("✗ High variance - approximation quality depends heavily on the specific video")
        
        print()
        print("="*80)


if __name__ == "__main__":
    main()
