"""
Minimal test for Enhanced API imports
"""

import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_enhanced_api():
    """Test enhanced API import and basic functionality"""
    print("🧪 Testing Enhanced Web API...")
    
    try:
        print("📦 Testing imports...")
        
        # Test Flask availability
        try:
            import flask
            print("✅ Flask available")
        except ImportError:
            print("❌ Flask not available")
            return False
        
        # Test flask-socketio availability
        try:
            import flask_socketio
            print("✅ Flask-SocketIO available")
        except ImportError:
            print("⚠️ Flask-SocketIO not available (real-time features disabled)")
        
        # Test Enhanced API import
        from train_system.api.web_server import EnhancedTrainingAPI
        print("✅ EnhancedTrainingAPI imported successfully")
        
        # Test creating API instance
        api = EnhancedTrainingAPI()
        print("✅ EnhancedTrainingAPI instance created")
        print(f"   Host: {api.host}")
        print(f"   Port: {api.port}")
        print(f"   SocketIO: {'Available' if api.socketio else 'Disabled'}")
        
        # Test managers
        if hasattr(api, 'job_manager'):
            print("✅ JobManager initialized")
            
            # Test job creation
            test_config = {
                "model": {"name": "test", "type": "torchvision", "model_name": "resnet18", "num_classes": 2},
                "training": {"epochs": 1, "learning_rate": 0.001}
            }
            job_id = api.job_manager.create_job(test_config)
            print(f"✅ Test job created: {job_id[:8]}...")
        
        if hasattr(api, 'file_manager'):
            print("✅ FileManager initialized")
        
        print("\n🎉 Enhanced API is working correctly!")
        print("📝 To start the server:")
        print("   python -m train_system.api.web_server")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Install missing dependencies with:")
        print("   pip install flask flask-cors flask-socketio")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_enhanced_api()
