#!/usr/bin/env python3
"""
Kaggle Installation Launcher
Redirects to the universal installer with Kaggle-specific messaging
"""

import os
import sys
from pathlib import Path

# Add current directory to path to import universal installer
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Launch universal installer with Kaggle context"""
    print("Kaggle Installation")
    print("Launching universal installer optimized for Kaggle...")
    print("=" * 50)
    
    # Set environment variable to help with detection
    os.environ['KAGGLE_INSTALLER'] = 'true'
    
    try:
        from install_universal import UniversalInstaller
        installer = UniversalInstaller(verbose=True)
        
        # Verify we're in Kaggle environment
        if installer.environment != 'kaggle':
            print("WARNING: This doesn't appear to be a Kaggle environment.")
            print("The installer will still work but may not be optimized.")
            print()
        
        success = installer.install()
        
        if success and installer.environment == 'kaggle':
            print("\nKaggle-specific tips:")
            print("• Upload datasets using Kaggle Datasets")
            print("• Use /kaggle/input/ for dataset access")
            print("• Save outputs to /kaggle/working/")
            print("• Use num_workers=2 for optimal performance")
            print("• Enable GPU in notebook settings if available")
            print("• Keep batch_size moderate (32-64) to avoid memory issues")
        
        return 0 if success else 1
        
    except ImportError:
        print("ERROR: Could not import universal installer")
        print("Make sure install_universal.py is in the same directory")
        return 1

if __name__ == "__main__":
    sys.exit(main())
