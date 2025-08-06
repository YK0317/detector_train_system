"""
Deployment utilities for the Enhanced Train System API

Provides helper functions and scripts for deploying the web API.
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil
from typing import List, Optional

from .config import get_config, create_directories
from .templates import create_web_templates


class DeploymentManager:
    """Manages API deployment tasks"""
    
    def __init__(self, config_name: str = 'development'):
        self.config = get_config(config_name)
        self.config_name = config_name
    
    def setup_environment(self):
        """Setup the deployment environment"""
        print(f"Setting up {self.config_name} environment...")
        
        # Create necessary directories
        create_directories(self.config)
        print("✅ Created directories")
        
        # Create web templates
        template_path = create_web_templates()
        print(f"✅ Created web templates at {template_path}")
        
        # Install dependencies
        self.install_dependencies()
        
        print("✅ Environment setup complete!")
    
    def install_dependencies(self):
        """Install required dependencies"""
        print("Installing API dependencies...")
        
        requirements_file = Path(__file__).parent / "requirements-web.txt"
        if requirements_file.exists():
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
                ])
                print("✅ Dependencies installed")
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Warning: Failed to install some dependencies: {e}")
                print("You may need to install them manually:")
                with open(requirements_file, 'r') as f:
                    print(f.read())
        else:
            print("⚠️ requirements-web.txt not found")
    
    def create_docker_files(self):
        """Create Docker deployment files"""
        print("Creating Docker deployment files...")
        
        # Dockerfile
        dockerfile_content = """
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY train_system/train_system/api/requirements-web.txt .
COPY train_system/requirements.txt .
RUN pip install --no-cache-dir -r requirements-web.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install train_system package
RUN cd train_system && python setup.py develop

# Create data directories
RUN mkdir -p /app/data/uploads /app/data/jobs

# Expose port
EXPOSE 5000

# Set environment variables
ENV API_CONFIG=docker
ENV PYTHONPATH=/app

# Run the application
CMD ["python", "-m", "train_system.api.web_server"]
        """
        
        with open("Dockerfile", "w") as f:
            f.write(dockerfile_content)
        
        # Docker Compose
        compose_content = """
version: '3.8'

services:
  train-api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - API_CONFIG=docker
      - SECRET_KEY=your-secret-key-here
      - CORS_ORIGINS=http://localhost:3000,http://localhost:8080
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped

volumes:
  data:
  logs:
        """
        
        with open("docker-compose.yml", "w") as f:
            f.write(compose_content)
        
        print("✅ Created Dockerfile and docker-compose.yml")
    
    def create_systemd_service(self):
        """Create systemd service file for Linux deployment"""
        if sys.platform != 'linux':
            print("⚠️ Systemd service files are only for Linux systems")
            return
        
        service_content = f"""
[Unit]
Description=Train System Web API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory={os.getcwd()}
Environment=PATH={os.environ.get('PATH')}
Environment=API_CONFIG=production
Environment=SECRET_KEY=your-secret-key-here
ExecStart={sys.executable} -m train_system.api.web_server --host 0.0.0.0 --port 5000
Restart=always

[Install]
WantedBy=multi-user.target
        """
        
        service_file = "train-system-api.service"
        with open(service_file, "w") as f:
            f.write(service_content)
        
        print(f"✅ Created systemd service file: {service_file}")
        print("To install:")
        print(f"  sudo cp {service_file} /etc/systemd/system/")
        print("  sudo systemctl daemon-reload")
        print("  sudo systemctl enable train-system-api")
        print("  sudo systemctl start train-system-api")
    
    def create_nginx_config(self):
        """Create nginx configuration for reverse proxy"""
        nginx_content = """
server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 16G;  # Allow large file uploads

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts for long-running requests
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    # Static file serving (if needed)
    location /static/ {
        alias /path/to/your/static/files/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
        """
        
        with open("nginx-train-api.conf", "w") as f:
            f.write(nginx_content)
        
        print("✅ Created nginx configuration: nginx-train-api.conf")
    
    def check_system_requirements(self) -> bool:
        """Check if system meets requirements"""
        print("Checking system requirements...")
        
        requirements_met = True
        
        # Check Python version
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ required")
            requirements_met = False
        else:
            print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
        
        # Check available disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            free_gb = free // (1024**3)
            if free_gb < 10:
                print(f"⚠️ Low disk space: {free_gb}GB available (recommend 10GB+)")
            else:
                print(f"✅ Disk space: {free_gb}GB available")
        except:
            print("⚠️ Could not check disk space")
        
        # Check memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_gb = memory.total // (1024**3)
            if memory_gb < 4:
                print(f"⚠️ Low memory: {memory_gb}GB (recommend 8GB+)")
            else:
                print(f"✅ Memory: {memory_gb}GB")
        except ImportError:
            print("⚠️ Could not check memory (psutil not installed)")
        
        return requirements_met
    
    def deploy(self, deployment_type: str = 'development'):
        """Run full deployment process"""
        print(f"Starting {deployment_type} deployment...")
        
        if not self.check_system_requirements():
            print("❌ System requirements not met")
            return False
        
        try:
            self.setup_environment()
            
            if deployment_type == 'docker':
                self.create_docker_files()
            elif deployment_type == 'production':
                self.create_systemd_service()
                self.create_nginx_config()
            
            print(f"✅ {deployment_type} deployment ready!")
            
            if deployment_type == 'development':
                print("\nTo start the development server:")
                print("  python -m train_system.api.web_server --debug")
                print("\nThen visit: http://localhost:5000/templates/dashboard.html")
            
            return True
            
        except Exception as e:
            print(f"❌ Deployment failed: {e}")
            return False


def deploy_api(config_name: str = 'development', deployment_type: str = 'development'):
    """Main deployment function"""
    manager = DeploymentManager(config_name)
    return manager.deploy(deployment_type)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy Train System Web API")
    parser.add_argument('--config', default='development', 
                       choices=['development', 'production', 'testing', 'docker'],
                       help='Configuration to use')
    parser.add_argument('--type', default='development',
                       choices=['development', 'production', 'docker'],
                       help='Deployment type')
    
    args = parser.parse_args()
    
    success = deploy_api(args.config, args.type)
    sys.exit(0 if success else 1)
