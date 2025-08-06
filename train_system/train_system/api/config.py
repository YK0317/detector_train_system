"""
Configuration Management for Enhanced Web API

Provides configuration classes for different deployment scenarios.
"""

import os
from pathlib import Path
from typing import Optional


class APIConfig:
    """Base configuration class"""
    
    # Server settings
    HOST = os.environ.get('API_HOST', 'localhost')
    PORT = int(os.environ.get('API_PORT', 5000))
    DEBUG = os.environ.get('API_DEBUG', 'False').lower() == 'true'
    
    # File storage settings
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploaded_files')
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_FILE_SIZE', 16 * 1024 * 1024 * 1024))  # 16GB default
    
    # Security settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # CORS settings
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')
    
    # Job management
    MAX_CONCURRENT_JOBS = int(os.environ.get('MAX_CONCURRENT_JOBS', 3))
    JOB_TIMEOUT = int(os.environ.get('JOB_TIMEOUT', 3600 * 24))  # 24 hours default
    
    # File type restrictions
    ALLOWED_DATASET_EXTENSIONS = {'zip', 'tar', 'tar.gz', 'tgz'}
    ALLOWED_MODEL_EXTENSIONS = {'pth', 'pt', 'pkl', 'onnx', 'h5'}
    ALLOWED_CONFIG_EXTENSIONS = {'yaml', 'yml', 'json'}


class DevelopmentConfig(APIConfig):
    """Development configuration"""
    DEBUG = True
    HOST = 'localhost'
    PORT = 5000
    
    # More permissive settings for development
    MAX_CONCURRENT_JOBS = 1
    CORS_ORIGINS = ['*']


class ProductionConfig(APIConfig):
    """Production configuration"""
    DEBUG = False
    HOST = '0.0.0.0'
    PORT = int(os.environ.get('PORT', 5000))
    
    # Stricter settings for production
    SECRET_KEY = os.environ.get('SECRET_KEY')  # Must be set in production
    MAX_CONCURRENT_JOBS = int(os.environ.get('MAX_CONCURRENT_JOBS', 5))
    
    # Production CORS settings
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '').split(',')
    
    @classmethod
    def validate(cls):
        """Validate production configuration"""
        if not cls.SECRET_KEY:
            raise ValueError("SECRET_KEY must be set in production")
        if not cls.CORS_ORIGINS or cls.CORS_ORIGINS == ['']:
            raise ValueError("CORS_ORIGINS must be set in production")


class TestingConfig(APIConfig):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    
    # Use temporary directories for testing
    UPLOAD_FOLDER = 'test_uploads'
    MAX_CONCURRENT_JOBS = 1
    JOB_TIMEOUT = 60  # Shorter timeout for tests


class DockerConfig(ProductionConfig):
    """Docker deployment configuration"""
    HOST = '0.0.0.0'
    PORT = 5000
    
    # Docker-specific paths
    UPLOAD_FOLDER = '/app/data/uploads'
    
    @classmethod
    def setup_directories(cls):
        """Setup required directories for Docker deployment"""
        Path(cls.UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)


# Configuration mapping
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'docker': DockerConfig
}


def get_config(config_name: Optional[str] = None) -> APIConfig:
    """Get configuration class based on environment"""
    if config_name is None:
        config_name = os.environ.get('API_CONFIG', 'development')
    
    config_class = config_map.get(config_name, DevelopmentConfig)
    
    # Validate production config
    if config_name == 'production':
        config_class.validate()
    
    return config_class


def create_directories(config: APIConfig):
    """Create necessary directories"""
    Path(config.UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(config.UPLOAD_FOLDER, 'datasets').mkdir(exist_ok=True)
    Path(config.UPLOAD_FOLDER, 'models').mkdir(exist_ok=True)
    Path('training_jobs').mkdir(exist_ok=True)
    Path('templates').mkdir(exist_ok=True)
