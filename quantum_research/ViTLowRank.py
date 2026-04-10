import torch

ckpt = torch.load(
    '/home/wadeab/universal_manipulation_interface/data/outputs/2026.02.20/18.26.46_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt',
    map_location='cpu',
    weights_only=False
)

# Get the model state dict (not EMA)
state = ckpt['state_dicts']['model']

# Extract ViT keys - they're under obs_encoder.key_model_map.camera0_rgb
vit_prefix = 'obs_encoder.key_model_map.camera0_rgb.'
vit_keys = [k for k in state.keys() if k.startswith(vit_prefix)]

print(f"Found {len(vit_keys)} ViT keys")
print("Sample keys:", vit_keys[:5])

# Save with simplified key names (strip the prefix for easier loading)
vit_state = {k.replace(vit_prefix, ''): state[k] for k in vit_keys}
torch.save(vit_state, 'vit_encoder_only.pt')

param_count = sum(p.numel() for p in vit_state.values())
print(f"\nSaved vit_encoder_only.pt ({param_count / 1e6:.1f}M params)")