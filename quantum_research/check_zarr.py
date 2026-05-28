import zarr
import numpy as np
import zipfile
import os
import shutil

ZARR_PATH = r'C:\Users\neald\Desktop\QuantumInspiredViTSpeedUp\quantum_research\data_for_quantum_research2\dataset.zarr.zip'
EXTRACT_PATH = r'C:\Users\neald\Desktop\QuantumInspiredViTSpeedUp\quantum_research\data_for_quantum_research2\dataset_extracted'

print("=" * 60)
print("Inspecting UMI Dataset")
print("=" * 60)

print(f"\nZarr version: {zarr.__version__}")
print(f"File: {ZARR_PATH}")
print(f"Size: {os.path.getsize(ZARR_PATH) / 1024 / 1024:.2f} MB")

# Try with allowZip64=True
print(f"\nTrying to read zip with ZIP64 support...")
try:
    with zipfile.ZipFile(ZARR_PATH, 'r', allowZip64=True) as zip_ref:
        names = zip_ref.namelist()
        print(f"Total files in zip: {len(names)}")
        print(f"\nFirst 30 files:")
        for name in names[:30]:
            print(f"  {name}")
        
        print(f"\nUnique top-level folders:")
        top_level = set()
        for name in names:
            parts = name.split('/')
            if parts[0]:
                top_level.add(parts[0])
        for tl in sorted(top_level):
            print(f"  {tl}/")
        
        # Look for key files
        print(f"\nLooking for important files...")
        for name in names:
            if 'action' in name.lower() or '.zarray' in name or '.zgroup' in name or '.zattrs' in name:
                if name.count('/') <= 3:  # Don't go too deep
                    print(f"  {name}")
        
        # Extract everything
        if not os.path.exists(EXTRACT_PATH):
            print(f"\nExtracting to {EXTRACT_PATH}...")
            print("This may take a few minutes for 2.2GB...")
            zip_ref.extractall(EXTRACT_PATH)
            print("Extracted!")
        else:
            print(f"\nAlready extracted at {EXTRACT_PATH}")

except Exception as e:
    print(f"Failed with zipfile: {e}")
    print("\nTrying with shutil.unpack_archive...")
    try:
        shutil.unpack_archive(ZARR_PATH, EXTRACT_PATH, 'zip')
        print("Extracted with shutil!")
    except Exception as e2:
        print(f"Also failed: {e2}")

# Now look at what got extracted
print(f"\nContents of {EXTRACT_PATH}:")
if os.path.exists(EXTRACT_PATH):
    for item in os.listdir(EXTRACT_PATH):
        full = os.path.join(EXTRACT_PATH, item)
        if os.path.isdir(full):
            print(f"  [DIR]  {item}/")
        else:
            print(f"  [FILE] {item}")

# Try to open with zarr
print(f"\nTrying to open extracted folder as zarr...")
try:
    # Try a few possible paths
    possible_paths = [
        EXTRACT_PATH,
        os.path.join(EXTRACT_PATH, 'dataset.zarr'),
    ]
    
    # Find the actual zarr folder
    for item in os.listdir(EXTRACT_PATH):
        full = os.path.join(EXTRACT_PATH, item)
        if os.path.isdir(full):
            possible_paths.append(full)
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                root = zarr.open(path, mode='r')
                print(f"Success! Opened: {path}")
                print(f"Keys: {list(root.keys())}")
                
                # Walk structure
                print(f"\nStructure:")
                def walk(group, indent=0):
                    prefix = "  " * indent
                    for key in group.keys():
                        item = group[key]
                        if hasattr(item, 'keys'):
                            print(f"{prefix}[GROUP] {key}/")
                            walk(item, indent + 1)
                        else:
                            print(f"{prefix}[ARRAY] {key}: shape={item.shape}, dtype={item.dtype}")
                walk(root)
                
                # Get actions
                if 'data' in root and 'action' in root['data']:
                    actions = root['data']['action'][:]
                    print(f"\nACTIONS FOUND!")
                    print(f"  Shape: {actions.shape}")
                    print(f"  Dtype: {actions.dtype}")
                    print(f"  First: {actions[0]}")
                    print(f"  Range: [{actions.min():.4f}, {actions.max():.4f}]")
                
                break
            except Exception as e:
                print(f"  Tried {path}: {e}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)