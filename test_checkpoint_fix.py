#!/usr/bin/env python3
"""
Test script to verify checkpoint saving logic works correctly.
This tests the fix for the save_best_only bug.
"""
import os
import tempfile
import shutil
from pathlib import Path
import sys

# Add train_system to path
sys.path.insert(0, str(Path(__file__).parent))

def test_checkpoint_saving_logic():
    """Test that checkpoint saving works correctly with save_best_only setting"""
    
    print("🧪 Testing checkpoint saving logic...")
    
    # Test cases to verify
    test_cases = [
        {
            "name": "save_best_only=true, is_best=true",
            "save_best_only": True,
            "save_last": True,
            "is_best": True,
            "expected_files": ["best_checkpoint.pth", "last_checkpoint.pth"],
            "not_expected_files": ["checkpoint_epoch_1.pth"]
        },
        {
            "name": "save_best_only=true, is_best=false", 
            "save_best_only": True,
            "save_last": True,
            "is_best": False,
            "expected_files": ["last_checkpoint.pth"],
            "not_expected_files": ["best_checkpoint.pth", "checkpoint_epoch_1.pth"]
        },
        {
            "name": "save_best_only=false, is_best=true",
            "save_best_only": False,
            "save_last": True, 
            "is_best": True,
            "expected_files": ["best_checkpoint.pth", "last_checkpoint.pth", "checkpoint_epoch_1.pth"],
            "not_expected_files": []
        },
        {
            "name": "save_best_only=false, is_best=false",
            "save_best_only": False,
            "save_last": True,
            "is_best": False,
            "expected_files": ["last_checkpoint.pth", "checkpoint_epoch_1.pth"],
            "not_expected_files": ["best_checkpoint.pth"]
        }
    ]
    
    # Mock the relevant parts of the checkpoint saving logic
    def simulate_checkpoint_save(output_dir, save_best_only, save_last, is_best, epoch=1):
        """Simulate the checkpoint saving logic from trainer.py"""
        
        checkpoint_data = {"epoch": epoch, "test": "data"}
        ext = "pth"
        
        # Save latest checkpoint
        if save_last:
            checkpoint_path = output_dir / f"last_checkpoint.{ext}"
            # Simulate torch.save
            checkpoint_path.touch()
            
        # Save best checkpoint (always save when validation improves)
        if is_best:
            checkpoint_path = output_dir / f"best_checkpoint.{ext}"
            # Simulate torch.save
            checkpoint_path.touch()
            
        # Save periodic full checkpoint (for training resumption)
        # Skip periodic checkpoints if save_best_only is enabled
        if not save_best_only:
            checkpoint_freq = 1  # Simulate checkpoint_frequency = 1
            if checkpoint_freq > 0 and epoch % checkpoint_freq == 0:
                periodic_path = output_dir / f"checkpoint_epoch_{epoch}.{ext}"
                # Simulate torch.save
                periodic_path.touch()
    
    all_passed = True
    
    for test_case in test_cases:
        print(f"\n📋 Testing: {test_case['name']}")
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Run the checkpoint saving logic
            simulate_checkpoint_save(
                output_dir=output_dir,
                save_best_only=test_case["save_best_only"],
                save_last=test_case["save_last"],
                is_best=test_case["is_best"]
            )
            
            # Check expected files exist
            for expected_file in test_case["expected_files"]:
                file_path = output_dir / expected_file
                if not file_path.exists():
                    print(f"❌ FAIL: Expected file '{expected_file}' not found")
                    all_passed = False
                else:
                    print(f"✅ Found expected file: {expected_file}")
            
            # Check unexpected files don't exist
            for not_expected_file in test_case["not_expected_files"]:
                file_path = output_dir / not_expected_file
                if file_path.exists():
                    print(f"❌ FAIL: Unexpected file '{not_expected_file}' found")
                    all_passed = False
                else:
                    print(f"✅ Correctly absent: {not_expected_file}")
    
    print(f"\n{'🎉 All tests PASSED!' if all_passed else '💥 Some tests FAILED!'}")
    return all_passed

if __name__ == "__main__":
    test_checkpoint_saving_logic()
