#!/usr/bin/env python3
"""
Test script for optional API installation
"""

print('🧪 Testing Optional API Installation')
print('=' * 50)

# Test base installation
try:
    from train_system import UnifiedTrainer
    print('✅ Base installation: OK')
except ImportError as e:
    print(f'❌ Base installation: {e}')

# Test web API installation
try:
    from train_system.api.web_server import EnhancedTrainingAPI
    api = EnhancedTrainingAPI()
    print('✅ Web API installation: OK')
    print(f'   - Flask available: {api.app is not None}')
    print(f'   - SocketIO available: {api.socketio is not None}')
except ImportError as e:
    print(f'❌ Web API installation: {e}')

# Test CLI functionality
try:
    from train_system.cli.main import show_system_info
    print('✅ CLI functionality: OK')
except ImportError as e:
    print(f'❌ CLI functionality: {e}')

print('')
print('🎉 Optional API installation test completed!')
