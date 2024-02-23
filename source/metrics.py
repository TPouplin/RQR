import numpy as np
import torch


def coverage(y_test,y_lower,y_upper):
    """
    Compute the coverage of the prediction intervals
    """
 
    coverage = torch.mean(((y_test >= y_lower) & (y_test <= y_upper)).float(), axis=0)
    return coverage

def interval_width(y_lower,y_upper):
    """
    Compute the length of the prediction intervals
    """
    interval_length = torch.mean(y_upper - y_lower)
    return interval_length

def compute_metrics(y_pred, target):
    results = {}
    
    y_pred[:, 0], y_pred[:, 1] = torch.minimum(y_pred[:, 0], y_pred[:, 1]), torch.maximum(y_pred[:, 0], y_pred[:, 1])
    
    target = target.squeeze()
    results["coverage"] = coverage(target,y_pred[:,0],y_pred[:,1])
    results["interval_width"] = interval_width(y_pred[:,0],y_pred[:,1])
    
    return results