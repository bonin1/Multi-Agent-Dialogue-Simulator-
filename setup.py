#!/usr/bin/env python3
"""
Bootstrap a local dev environment for Multi-Agent Dialogue Simulator.

This is not setuptools packaging metadata (there is no setup() here). For
reviewers and packaging tools, prefer declaring dependencies in pyproject.toml
and installing with ``pip install -e .`` or ``pip install -r requirements.txt``.
"""

import argparse
import logging
import os
import subprocess
import sys
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
    """Test if required modules from requirements.txt can be imported."""
    logger.info("Testing imports...")

    # Keep in sync with requirements.txt (import name may differ from PyPI name).
    required_modules = [
        "streamlit",
        "transformers",
        "torch",
        "langchain",
        "langchain_community",
        "chromadb",
        "sentence_transformers",
        "pydantic",
        "networkx",
        "numpy",
        "pandas",
        "plotly",
        "huggingface_hub",
        "accelerate",
    ]

    optional_modules = [
        # Often missing or broken on CPU-only / some Windows setups; warn only.
        "bitsandbytes",
    ]

    failed_imports = []

    for module in required_modules:
        try:
            __import__(module)
            logger.info("OK %s", module)
        except ImportError as e:
            logger.error("FAIL %s: %s", module, e)
            failed_imports.append(module)

    for module in optional_modules:
        try:
            __import__(module)
            logger.info("OK %s (optional)", module)
        except ImportError as e:
            logger.warning("SKIP %s (optional): %s", module, e)

    if failed_imports:
        logger.error("Failed to import: %s", ", ".join(failed_imports))
        return False

    logger.info("All required modules imported successfully")
    return True

def download_model():
    """Download the default model if not present"""
    logger.info("Checking for default model...")

    try:
        from huggingface_hub import snapshot_download

        model_name = "teknium/OpenHermes-2.5-Mistral-7B"
        model_path = Path(__file__).parent / "models" / "downloaded" / model_name.split("/")[-1]

        if not model_path.exists():
            logger.info("Downloading model %s...", model_name)
            kwargs = {
                "repo_id": model_name,
                "local_dir": str(model_path),
            }
            # local_dir_use_symlinks removed in newer huggingface_hub; pass only if supported.
            try:
                snapshot_download(**kwargs, local_dir_use_symlinks=False)
            except TypeError:
                snapshot_download(**kwargs)
            logger.info("Model downloaded successfully")
        else:
            logger.info("Model already exists")

        return True
    except Exception as e:
        logger.warning("Failed to download model: %s", e)
        logger.info("Model will be downloaded automatically when first used")
        return True  # Not critical, can download later


def warn_if_not_venv():
    """Encourage isolated envs so reviewers and CI reproduce the same stack."""
    in_venv = sys.prefix != sys.base_prefix or os.environ.get("VIRTUAL_ENV")
    if not in_venv:
        logger.warning(
            "Not running inside a virtualenv; consider `python -m venv .venv` "
            "and activate it before setup for reproducible installs."
        )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--skip-pip",
        action="store_true",
        help="Only create dirs and run checks; do not pip install -r requirements.txt",
    )
    p.add_argument(
        "--skip-import-check",
        action="store_true",
        help="Skip import smoke test after install",
    )
    p.add_argument(
        "--skip-model-download",
        action="store_true",
        help="Do not download the default Hugging Face model (large download)",
    )
    return p.parse_args(argv)


def main(argv=None):
    """Main setup function"""
    args = parse_args(argv)
    logger.info("Starting Multi-Agent Dialogue Simulator setup...")

    warn_if_not_venv()

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Create directories
    create_directories()

    # Install requirements
    if not args.skip_pip:
        if not install_requirements():
            logger.error("Setup failed: Could not install requirements")
            sys.exit(1)
    else:
        logger.info("Skipping pip install (--skip-pip)")

    # Test imports
    if not args.skip_import_check:
        if not test_imports():
            logger.error("Setup failed: Import test failed")
            sys.exit(1)
    else:
        logger.info("Skipping import check (--skip-import-check)")

    # Check GPU support
    check_gpu_support()

    # Download model (optional)
    if not args.skip_model_download:
        download_model()
    else:
        logger.info("Skipping model download (--skip-model-download)")

    logger.info("Setup completed successfully!")
    logger.info("To start the application, run: streamlit run main.py")

if __name__ == "__main__":
    main()
