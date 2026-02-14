import torch
import os
import logging

logger = logging.getLogger(__name__)

def set_gpu_device(gpu_id: int) -> None:
    """Set global GPU device for both PyTorch and TensorFlow."""
    if torch.cuda.is_available():
        # Set CUDA device for PyTorch
        torch.cuda.set_device(gpu_id)
        # Set environment variable for TensorFlow
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logger.info(f"Using GPU device {gpu_id}")
    else:
        logger.warning("No GPU available, using CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
