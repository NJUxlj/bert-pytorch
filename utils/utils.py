
import torch
import random
import numpy as np
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import json
import os
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logging(save_dir: str, log_config: Optional[str] = None) -> None:
    """Setup logging configuration"""
    log_config = {
        'version': 1,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(message)s'
            },
        },
        'handlers': {
            'file_handler': {
                'class': 'logging.FileHandler',
                'filename': os.path.join(save_dir, 'train.log'),
                'formatter': 'standard',
            },
            'stream_handler': {
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            }
        },
        'root': {
            'level': 'INFO',
            'handlers': ['file_handler', 'stream_handler']
        }
    }

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    logging.config.dictConfig(log_config)

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class TensorboardWriter:
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value to tensorboard"""
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag: str, values: Dict[str, float], step: int):
        """Log multiple scalar values to tensorboard"""
        self.writer.add_scalars(tag, values, step)

    def close(self):
        self.writer.close()

def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    checkpoint_dir: str,
    filename: str = 'checkpoint.pth.tar'
) -> None:
    """Save checkpoint to disk"""
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'model_best.pth.tar')
        torch.save(state, best_filepath)

def load_checkpoint(checkpoint_path: str, model, optimizer=None):
    """Load checkpoint from disk"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist")
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    return checkpoint

def get_lr(optimizer):
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']

def compute_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy"""
    with torch.no_grad():
        predictions = torch.argmax(predictions, dim=1)
        correct = (predictions == labels).float()
        accuracy = correct.sum() / len(correct)
    return accuracy.item()

