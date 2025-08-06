#!/usr/bin/env python3
"""
Optional Web API Setup Extension for Train System

This extends the base installation with web API functionality.
Install with: pip install -e ".[web]" or python setup_web.py install
"""

import sys
from pathlib import Path
from setuptools import setup

# Add web API entry points
web_entry_points = {
    "console_scripts": [
        "ts-api=train_system.api.server:run_server",
        "ts-web-api=train_system.api.web_server:run_web_server",
        "ts-enhanced-api=train_system.api.web_server:run_web_server",
    ]
}

def install_web_api():
    """Install web API components"""
    print("Installing Train System with Web API support...")
    print("=" * 60)
    
    # Import the main setup configuration
    setup_path = Path(__file__).parent / "setup.py"
    
    # Read setup.py and add web entry points
    with open(setup_path, 'r') as f:
        setup_content = f.read()
    
    # Add web entry points to existing setup
    setup_kwargs = {}
    exec(setup_content.replace('setup(', 'setup_kwargs.update('), setup_kwargs)
    
    # Add web entry points
    if 'entry_points' not in setup_kwargs:
        setup_kwargs['entry_points'] = {}
    
    if 'console_scripts' not in setup_kwargs['entry_points']:
        setup_kwargs['entry_points']['console_scripts'] = []
    
    # Add web API scripts
    setup_kwargs['entry_points']['console_scripts'].extend([
        "ts-api=train_system.api.server:run_server",
        "ts-web-api=train_system.api.web_server:run_web_server",
        "ts-enhanced-api=train_system.api.web_server:run_web_server",
    ])
    
    # Install with web dependencies
    if 'install_requires' not in setup_kwargs:
        setup_kwargs['install_requires'] = []
    
    # Add web requirements to install_requires
    web_deps = [
        "flask>=2.0.0",
        "flask-cors>=3.0.0",
        "flask-socketio>=5.0.0",
        "requests>=2.25.0",
        "werkzeug>=2.0.0",
    ]
    
    setup_kwargs['install_requires'].extend(web_deps)
    
    # Run setup
    setup(**setup_kwargs)
    
    print("\n✅ Train System with Web API installed successfully!")
    print("Available commands:")
    print("  ts-api              - Basic API server")
    print("  ts-web-api          - Enhanced web API server")
    print("  ts-enhanced-api     - Enhanced web API server (alias)")
    print("\nTest the installation:")
    print("  python -m train_system.api.web_server --help")


if __name__ == "__main__":
    install_web_api()
