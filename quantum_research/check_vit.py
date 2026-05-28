import torch

# Load the file
checkpoint = torch.load('vit_encoder_only.pt', weights_only=False)

print("=" * 60)
print("ViT Encoder File Contents")
print("=" * 60)

# Check what type it is
print(f"\nType: {type(checkpoint)}")

# If it's a dict, show the keys
if isinstance(checkpoint, dict):
    print(f"\nKeys in checkpoint:")
    for key in checkpoint.keys():
        print(f"  - {key}")
    
    # Check if it has 'model' or 'state_dict' key
    if 'model' in checkpoint:
        print("\n✓ Found 'model' key")
        print(f"  Type: {type(checkpoint['model'])}")
    
    if 'state_dict' in checkpoint:
        print("\n✓ Found 'state_dict' key")
        print(f"  Number of parameters: {len(checkpoint['state_dict'])}")
        print(f"  First few keys:")
        for i, key in enumerate(list(checkpoint['state_dict'].keys())[:5]):
            print(f"    {i+1}. {key}")
    
    # Show all top-level content
    print("\n📋 Top-level structure:")
    for key, value in checkpoint.items():
        if isinstance(value, dict):
            print(f"  {key}: dict with {len(value)} items")
        elif isinstance(value, torch.nn.Module):
            print(f"  {key}: PyTorch model")
        else:
            print(f"  {key}: {type(value)}")
else:
    print("\n✓ This is a full model, not a dict")
    print(f"  Model type: {type(checkpoint)}")

print("\n" + "=" * 60)