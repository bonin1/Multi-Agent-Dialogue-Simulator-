#!/usr/bin/env python3
"""
Installation and setup script for Multi-Agent Dialogue Simulator
"""

import subprocess
import sys
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    required_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version < required_version:
        logger.error(f"Python {required_version[0]}.{required_version[1]}+ required. Current version: {current_version[0]}.{current_version[1]}")
        return False
    
    logger.info(f"Python version {current_version[0]}.{current_version[1]} is compatible")
    return True

def install_requirements():
    """Install required packages"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        logger.error("requirements.txt not found!")
        return False
    
    logger.info("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
        logger.info("Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        return False

def check_gpu_support():
    """Check for CUDA/GPU support"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            logger.info(f"CUDA GPU support detected: {gpu_count} GPU(s), Primary: {gpu_name}")
            return True
        else:
            logger.info("CUDA GPU support not available, will use CPU")
            return False
    except ImportError:
        logger.warning("PyTorch not installed, cannot check GPU support")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "memory_db",
        "logs",
        "exports",
        "models/downloaded"
    ]
    
    for directory in directories:
        dir_path = Path(__file__).parent / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def test_imports():
    """Test if all required modules can be imported"""
    logger.info("Testing imports...")
    
    required_modules = [
        "streamlit",
        "transformers",
        "torch",
        "chromadb",
        "sentence_transformers",
        "pydantic",
        "networkx",
        "numpy",
        "pandas",
        "plotly"
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            logger.info(f"✓ {module}")
        except ImportError as e:
            logger.error(f"✗ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        logger.error(f"Failed to import: {', '.join(failed_imports)}")
        return False
    
    logger.info("All required modules imported successfully")
    return True

def download_model():
    """Download the default model if not present"""
    logger.info("Checking for default model...")
    
    try:
        from huggingface_hub import snapshot_download
        model_name = "teknium/OpenHermes-2.5-Mistral-7B"
        model_path = Path(__file__).parent / "models" / "downloaded" / model_name.split('/')[-1]
        
        if not model_path.exists():
            logger.info(f"Downloading model {model_name}...")
            snapshot_download(
                repo_id=model_name,
                local_dir=str(model_path),
                local_dir_use_symlinks=False
            )
            logger.info("Model downloaded successfully")
        else:
            logger.info("Model already exists")
        
        return True
    except Exception as e:
        logger.warning(f"Failed to download model: {e}")
        logger.info("Model will be downloaded automatically when first used")
        return True  # Not critical, can download later

def main():
    """Main setup function"""
    logger.info("Starting Multi-Agent Dialogue Simulator setup...")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        logger.error("Setup failed: Could not install requirements")
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        logger.error("Setup failed: Import test failed")
        sys.exit(1)
    
    # Check GPU support
    check_gpu_support()
    
    # Download model (optional)
    download_model()
    
    logger.info("Setup completed successfully!")
    logger.info("To start the application, run: streamlit run main.py")

if __name__ == "__main__":
    main()
