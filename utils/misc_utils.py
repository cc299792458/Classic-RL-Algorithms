import random
import numpy as np
# import torch

# TODO: Install torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def moving_average_with_padding(data, window_size):
    """Compute the moving average of the data with padding to maintain the original length."""
    smoothed_data = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    
    # Calculate padding size
    pad_size = (len(data) - len(smoothed_data)) // 2
    
    # Pad the smoothed data at the beginning and end
    smoothed_data = np.pad(smoothed_data, (pad_size, pad_size), mode='edge')
    
    # If the padding did not perfectly align, adjust the size
    if len(smoothed_data) < len(data):
        smoothed_data = np.pad(smoothed_data, (0, 1), mode='edge')
    
    return smoothed_data