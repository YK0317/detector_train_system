import torch
from pathlib import Path

# Check if checkpoint exists
checkpoint_path = Path("C:/Users/mingw/Desktop/refactor/checkpoint_epoch_5.pth")
print(f"Checkpoint exists: {checkpoint_path.exists()}")

if checkpoint_path.exists():
    print(f"Checkpoint size: {checkpoint_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Load and inspect checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        print(f"Checkpoint type: {type(checkpoint)}")

        if isinstance(checkpoint, dict):
            print("Checkpoint keys:")
            for key in checkpoint.keys():
                if key.endswith("_dict"):
                    print(
                        f"  {key}: {type(checkpoint[key])} ({len(checkpoint[key])} items)"
                    )
                else:
                    print(f"  {key}: {checkpoint[key]}")

        print("✅ Checkpoint loaded successfully")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
else:
    print("❌ Checkpoint file not found")
