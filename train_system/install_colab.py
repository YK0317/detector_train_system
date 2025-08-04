#!/usr/bin/env python3
"""
Google Colab Installation Launcher
Redirects to the universal installer with Colab-specific messaging
"""

import os
import sys
from pathlib import Path

# Add current directory to path to import universal installer
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Launch universal installer with Colab context"""
    print("Google Colab Installation")
    print("Launching universal installer optimized for Colab...")
    print("=" * 50)
    
    # Set environment variable to help with detection
    os.environ['COLAB_INSTALLER'] = 'true'
    
    try:
        from install_universal import UniversalInstaller
        installer = UniversalInstaller(verbose=True)
        
        # Verify we're in Colab or similar environment
        if installer.environment not in ['colab', 'jupyter']:
            print("WARNING: This doesn't appear to be a Colab environment.")
            print("The installer will still work but may not be optimized.")
            print()
        
        success = installer.install()
        
        if success and installer.environment == 'colab':
            print("\nColab-specific tips:")
            print("• Use !python commands to run CLI tools")
            print("• Mount Google Drive for persistent storage")
            print("• Enable GPU runtime for faster training")
            print("• Use smaller batch sizes if you encounter memory errors")
        
        return 0 if success else 1
        
    except ImportError:
        print("ERROR: Could not import universal installer")
        print("Make sure install_universal.py is in the same directory")
        return 1

if __name__ == "__main__":
    sys.exit(main())
