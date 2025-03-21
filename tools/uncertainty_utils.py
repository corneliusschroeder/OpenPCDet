import numpy as np
import scipy.stats as stats

def compute_confidence_interval(variance, confidence=0.95, distribution=stats.norm):
    """
    Computes distance from a symmetric bounding box parameter CI.

    Args:
        variance (np.ndarray): Predicted log variances, if the value is exactly 0.0, 
        assumes 0 variance (shape: [N, D]).
        confidence (float): Confidence level

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Lower and upper bounds of CI.
    """
    
    variance[variance!=0.0] = np.exp(variance[variance!=0.0])

    z_score = distribution.ppf((1 + confidence) / 2)
    std_dev = np.sqrt(variance)

    distance_from_mean = z_score * std_dev

    return distance_from_mean