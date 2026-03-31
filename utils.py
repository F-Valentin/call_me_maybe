import numpy as np

class Fsm(Enum):
    START = 0
    NAME = 1
    PARAM = 2
    END = 3

def softmax(x):
    """
    Compute softmax values for each set of scores in x.
    
    Args:
        x: Input array of shape (batch_size, num_classes) or (num_classes,)
        
    Returns:
        Softmax probabilities of same shape as input
    """
    # For numerical stability, subtract the maximum value from each input vector
    # This prevents overflow when calculating exp(x)
    shifted_x = x - np.max(x, axis=-1, keepdims=True)
    
    # Calculate exp(x) for each element
    exp_x = np.exp(shifted_x)
    
    # Calculate the sum of exp(x) for normalization
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
    
    # Normalize to get probabilities
    probabilities = exp_x / sum_exp_x
    
    return probabilities
