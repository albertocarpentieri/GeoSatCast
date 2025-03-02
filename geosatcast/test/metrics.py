# metrics.py
import numpy as np
from scipy.stats import pearsonr
from scipy.ndimage import uniform_filter
# metrics.py
import torch
import torch.nn.functional as F


def critical_success_index(y_true, y_pred, threshold, mode='above'):
    """
    Calculate the Critical Success Index (CSI) for a given threshold.
    By default, it checks how well y_pred matches y_true for values 'above' the threshold.
    You can switch to 'below' the threshold by setting mode='below'.
    """
    # Remove NaNs
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if mode == 'above':
        hits = np.sum((y_pred >= threshold) & (y_true >= threshold))
        false_alarms = np.sum((y_pred >= threshold) & (y_true < threshold))
        misses = np.sum((y_pred < threshold) & (y_true >= threshold))
    elif mode == 'below':
        hits = np.sum((y_pred <= threshold) & (y_true <= threshold))
        false_alarms = np.sum((y_pred <= threshold) & (y_true > threshold))
        misses = np.sum((y_pred > threshold) & (y_true <= threshold))
    else:
        raise ValueError("mode must be 'above' or 'below'")

    return hits / (hits + false_alarms + misses + 1e-8)

def fraction_skill_score(y_true, y_pred, scale, threshold=0.0, mode='above'):
    """
    Calculate the Fraction Skill Score (FSS) at a given neighborhood scale,
    using a threshold and mode ('above' or 'below').

    Args:
        y_true, y_pred: 2D arrays (H x W) or 3D arrays, with possible NaN values.
        scale: size of the uniform filter (integer).
        threshold: float threshold for comparing field values.
        mode: 'above' or 'below'. 'above' compares how similar the areas above
              threshold are in y_true and y_pred. 'below' does the inverse.

    Returns:
        FSS: fraction skill score between 0 and 1, where 1 is perfect.
    """

    def compute_fractions(field, threshold, scale, mode):
        # Convert to binary mask based on threshold and mode
        if mode == 'above':
            field_bin = (field >= threshold).astype(np.float32)
        else:  # mode == 'below'
            field_bin = (field <= threshold).astype(np.float32)

        # Smooth/count in the neighborhood
        return uniform_filter(field_bin, size=scale, mode='constant', cval=0)
    
    # Handle NaNs
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    # Only keep valid pixels
    y_true = y_true * valid_mask
    y_pred = y_pred * valid_mask

    # Compute fraction fields
    ft = compute_fractions(y_true, threshold, scale, mode)
    fp = compute_fractions(y_pred, threshold, scale, mode)

    numerator = np.sum((fp - ft) ** 2)
    denominator = np.sum(ft ** 2 + fp ** 2 + 1e-8)

    return 1 - (numerator / denominator)

def pearson_correlation(y_true, y_pred):
    """
    Calculate Pearson correlation for 2D fields (with shape [H, W]) or flattened arrays.
    Returns np.nan if valid (finite) samples do not exist.
    """
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[valid_mask].flatten()
    y_pred = y_pred[valid_mask].flatten()
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    return pearsonr(y_true, y_pred)[0]


def critical_success_index_torch(y_true, y_pred, threshold, mode='above'):
    """
    GPU-based CSI.
    y_true, y_pred: torch.Tensor([H, W]) or any shape. We'll flatten except the batch dimension if needed.
    threshold: float
    mode: 'above' or 'below'
    Returns: scalar tensor (CSI).
    """
    valid_mask = torch.isfinite(y_true) & torch.isfinite(y_pred)

    if mode == 'above':
        hits         = ((y_pred >= threshold) & (y_true >= threshold) & valid_mask).sum()
        false_alarms = ((y_pred >= threshold) & (y_true < threshold)  & valid_mask).sum()
        misses       = ((y_pred < threshold)  & (y_true >= threshold) & valid_mask).sum()
    elif mode == 'below':
        hits         = ((y_pred <= threshold) & (y_true <= threshold) & valid_mask).sum()
        false_alarms = ((y_pred <= threshold) & (y_true > threshold)  & valid_mask).sum()
        misses       = ((y_pred > threshold)  & (y_true <= threshold) & valid_mask).sum()
    else:
        raise ValueError("mode must be 'above' or 'below'")

    hits_f        = hits.float()
    denom         = hits + false_alarms + misses + 1e-8
    return hits_f / denom.float()

def fraction_skill_score_torch(y_true, y_pred, scale, threshold=0.0, mode='above'):
    """
    GPU-based Fraction Skill Score (FSS).
    y_true, y_pred: torch.Tensor([H, W]) or [N, H, W]. We'll treat them as [N, 1, H, W] for conv2d.
    scale: neighborhood size (int)
    threshold: float
    mode: 'above' or 'below'
    """
    # 1) Binarize
    if mode == 'above':
        true_bin = (y_true >= threshold)
        pred_bin = (y_pred >= threshold)
    else:  # mode == 'below'
        true_bin = (y_true <= threshold)
        pred_bin = (y_pred <= threshold)

    # 2) Valid mask
    valid_mask = torch.isfinite(y_true) & torch.isfinite(y_pred)
    true_bin = true_bin & valid_mask
    pred_bin = pred_bin & valid_mask

    true_bin_f = true_bin.float()
    pred_bin_f = pred_bin.float()

    # 3) Reshape for conv2d => [N, 1, H, W]
    # If shape is [H, W], treat as N=1
    if true_bin_f.dim() == 2:
        true_bin_f = true_bin_f.unsqueeze(0).unsqueeze(0)
        pred_bin_f = pred_bin_f.unsqueeze(0).unsqueeze(0)
    elif true_bin_f.dim() == 3:
        # shape [N, H, W] => [N, 1, H, W]
        true_bin_f = true_bin_f.unsqueeze(1)
        pred_bin_f = pred_bin_f.unsqueeze(1)
    else:
        raise ValueError("Input must have shape [H, W] or [N, H, W].")

    kernel = torch.ones((1, 1, scale, scale), device=y_true.device, dtype=y_true.dtype)
    pad = scale // 2
    denom = float(scale * scale)

    ft = F.conv2d(true_bin_f, kernel, padding=pad) / denom
    fp = F.conv2d(pred_bin_f, kernel, padding=pad) / denom

    # Flatten
    ftf = ft.flatten()
    fpf = fp.flatten()

    numerator = ((fpf - ftf) ** 2).sum()
    denominator = ((fpf ** 2) + (ftf ** 2) + 1e-8).sum()
    fss = 1.0 - numerator / denominator
    return fss

def pearson_correlation_torch(y_true, y_pred):
    """
    GPU-based Pearson correlation
    y_true, y_pred: torch.Tensor([H, W]) or bigger dims. We'll flatten while ignoring NaNs.
    Returns: scalar tensor (corr).
    """
    valid_mask = torch.isfinite(y_true) & torch.isfinite(y_pred)
    x = y_true[valid_mask]
    y = y_pred[valid_mask]
    if x.numel() < 2:
        return torch.tensor(float('nan'), device=y_true.device)

    xm = x - x.mean()
    ym = y - y.mean()

    numerator = (xm * ym).sum()
    denominator = torch.sqrt((xm**2).sum() * (ym**2).sum()) + 1e-8
    corr = numerator / denominator
    return corr

def rmse_torch(y_true, y_pred):
    """
    GPU-based RMSE
    """
    valid_mask = torch.isfinite(y_true) & torch.isfinite(y_pred)
    diff_sq = (y_true - y_pred)**2
    diff_sq[~valid_mask] = 0.0  # zero out invalid
    n_valid = valid_mask.sum()
    if n_valid == 0:
        return torch.tensor(float('nan'), device=y_true.device)
    return torch.sqrt(diff_sq.sum() / n_valid.float())

def mae_torch(y_true, y_pred):
    """
    GPU-based MAE
    """
    valid_mask = torch.isfinite(y_true) & torch.isfinite(y_pred)
    diff_abs = torch.abs(y_true - y_pred)
    diff_abs[~valid_mask] = 0.0
    n_valid = valid_mask.sum()
    if n_valid == 0:
        return torch.tensor(float('nan'), device=y_true.device)
    return diff_abs.sum() / n_valid.float()

def mean_error_torch(y_true, y_pred):
    """
    GPU-based Mean Error (Bias)
    """
    valid_mask = torch.isfinite(y_true) & torch.isfinite(y_pred)
    diff = y_pred - y_true
    diff[~valid_mask] = 0.0
    n_valid = valid_mask.sum()
    if n_valid == 0:
        return torch.tensor(float('nan'), device=y_true.device)
    return diff.sum() / n_valid.float()