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
    if valid_mask.float().mean().item() < 0.5:
        return torch.tensor(float('nan'), device=y_true.device)
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
    if valid_mask.float().mean().item() < 0.5:
        return torch.tensor(float('nan'), device=y_true.device)
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
    if valid_mask.float().mean().item() < 0.5:
        return torch.tensor(float('nan'), device=y_true.device)
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
    if valid_mask.float().mean().item() < 0.5:
        return torch.tensor(float('nan'), device=y_true.device)

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
    if valid_mask.float().mean().item() < 0.5:
        return torch.tensor(float('nan'), device=y_true.device)
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
    if valid_mask.float().mean().item() < 0.5:
        return torch.tensor(float('nan'), device=y_true.device)
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
    if valid_mask.float().mean().item() < 0.5:
        return torch.tensor(float('nan'), device=y_true.device)
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
    if valid_mask.float().mean().item() < 0.5:
        return torch.tensor(float('nan'), device=y_true.device)
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
    if valid_mask.float().mean().item() < 0.5:
        return torch.tensor(float('nan'), device=y_true.device)
    diff = y_pred - y_true
    diff[~valid_mask] = 0.0
    n_valid = valid_mask.sum()
    return diff.sum() / n_valid.float()

##############################################
# Metric Functions (running on GPU with torch)
##############################################


def compute_temporal_autocorrelation(sample: torch.Tensor, lags: list = [1, 2, 4, 8]) -> dict:
    C, T, H, W = sample.shape
    result = {}
    for c in range(C):
        result[c] = compute_temporal_autocorrelation_per_c(sample[c], lags)
    return result

def compute_temporal_autocorrelation_per_c(sample: torch.Tensor, lags: list = [1, 2, 4, 8]) -> dict:
    """
    Compute temporal autocorrelation for each lag in a vectorized manner.
    
    The input sample is a tensor of shape (T, H, W). For each lag L, the function:
    
      1. Determines which time frames are “globally valid”: a frame is valid if at least 50% 
         of its pixels are non-NaN.
      2. For each lag, considers only pairs (t, t+L) where both time frames are valid.
      3. For each pixel location, uses only those pairs where the pixel value is available 
         at both time points. If a pixel does not have any valid pair for that lag, its contribution 
         is treated as NaN.
      4. Computes the Pearson correlation between the vectors formed by the valid pairs 
         (using the standard formula computed on a per-pixel basis).
      5. Averages the per-pixel lag correlations (ignoring NaNs) to return the autocorrelation for that lag.
    
    If no valid pairs exist for a given lag, the autocorrelation for that lag is set to NaN.
    
    Returns:
        A dictionary mapping each lag to the average autocorrelation over pixels.
    """
    T, H, W = sample.shape
    device = sample.device
    result = {}
    
    # Determine if a given time slice (frame) is globally valid: at least 50% non-NaN pixels.
    # This gives a boolean vector of length T.
    frame_valid = ( (~torch.isnan(sample)).float().mean(dim=(1, 2)) >= 0.5 )
    
    for lag in lags:
        if T <= lag:
            result[lag] = float('nan')
            continue

        # For time pairs (t, t+lag), we require that both frames are globally valid.
        valid_pair_mask = frame_valid[:T - lag] & frame_valid[lag:]
        if valid_pair_mask.sum() == 0:
            result[lag] = float('nan')
            continue
        
        # Select the time indices where the pair of frames is globally valid.
        valid_times = torch.nonzero(valid_pair_mask, as_tuple=False).squeeze(1)
        if valid_times.dim() == 0:
            # When only one valid index exists, wrap it so that indexing works consistently.
            valid_times = valid_times.unsqueeze(0)
        
        # For these valid time indices, extract pairs:
        # x: values at time t, y: values at time t+lag; both are of shape (n_valid, H, W)
        x = sample[valid_times, :, :]
        y = sample[valid_times + lag, :, :]
        
        # For each time pair and pixel, we require that both x and y are non-NaN.
        pair_valid = (~torch.isnan(x)) & (~torch.isnan(y))
        
        # Count valid pairs per pixel (shape H x W).
        count = pair_valid.float().sum(dim=0)  # shape (H, W)
        
        # To compute means per pixel over the valid pairs:
        # Replace NaNs with zero (they are excluded via the mask) and sum up.
        sum_x = torch.where(pair_valid, x, torch.zeros_like(x)).sum(dim=0)
        sum_y = torch.where(pair_valid, y, torch.zeros_like(y)).sum(dim=0)
        # Avoid division by zero; temporary zeros will later be masked.
        mean_x = sum_x / torch.clamp(count, min=1)
        mean_y = sum_y / torch.clamp(count, min=1)
        
        # Compute covariance and variances per pixel.
        # We subtract the per-pixel means (unsqueeze the time dimension to broadcast).
        x_centered = x - mean_x.unsqueeze(0)
        y_centered = y - mean_y.unsqueeze(0)
        
        # Compute sums only where the pair is valid.
        cov = torch.where(pair_valid, x_centered * y_centered, torch.zeros_like(x_centered)).sum(dim=0)
        var_x = torch.where(pair_valid, x_centered ** 2, torch.zeros_like(x_centered)).sum(dim=0)
        var_y = torch.where(pair_valid, y_centered ** 2, torch.zeros_like(y_centered)).sum(dim=0)
        
        # Average the sums by the count of valid pairs.
        cov = cov / torch.clamp(count, min=1)
        var_x = var_x / torch.clamp(count, min=1)
        var_y = var_y / torch.clamp(count, min=1)
        
        # Compute Pearson correlation per pixel.
        std_prod = torch.sqrt(var_x * var_y) + 1e-8
        corr = cov / std_prod
        
        # For pixels with no valid pair, set correlation to NaN.
        corr[count == 0] = float('nan')
        
        # Average the per-pixel correlations, ignoring NaNs.
        # If all pixels are NaN, torch.nanmean returns NaN.
        avg_corr = torch.nanmean(corr).item()
        result[lag] = avg_corr

    return result

def compute_spatial_autocorrelation(sample: torch.Tensor, shifts: list = [1, 2, 4, 8]) -> dict:
    C, T, H, W = sample.shape
    result = {}
    for c in range(C):
        result[c] = compute_spatial_autocorrelation_per_c(sample[c], shifts)
    return result


def compute_spatial_autocorrelation_per_c(sample: torch.Tensor, shifts: list = [1, 2, 3]) -> dict:
    """
    Compute spatial autocorrelation for each timestamp in a vectorized manner.
    
    The input sample is a tensor of shape (T, H, W), where each time slice is treated as an image.
    For each timestamp, if the valid fraction (non-NaN pixels) is below 50%, the computed autocorrelation
    for that timestamp is set to NaN.
    
    For each provided shift, the function computes:
      - Horizontal autocorrelation: shifting along the width dimension.
      - Vertical autocorrelation: shifting along the height dimension.
    
    The Pearson correlation is computed over the valid pixels (i.e. pixels that are non-NaN in both
    the original and shifted images). The spatial dimensions are flattened to perform the correlation
    computation in a vectorized way.
    
    Returns:
        A dictionary with keys "horizontal_shift_{shift}" and "vertical_shift_{shift}" mapping to a
        tensor of shape (T,) containing the spatial autocorrelation for each timestamp.
    """
    T, H, W = sample.shape
    N = H * W
    device = sample.device

    # Compute the valid fraction for each timestamp (image)
    valid_fraction = (~torch.isnan(sample)).float().view(T, -1).mean(dim=1)  # shape: (T,)

    results = {}

    for shift in shifts:
        # --- Horizontal shift (shift along width; for an image of shape (H, W) this is dim=1) ---
        # For a tensor of shape (T, H, W), horizontal shift corresponds to shifting along dim=2.
        shifted_h = torch.roll(sample, shifts=-shift, dims=2)
        valid_mask_h = (~torch.isnan(sample)) & (~torch.isnan(shifted_h))
        # Flatten the spatial dims for vectorized computation.
        x = sample.view(T, -1)
        y = shifted_h.view(T, -1)
        valid_mask_h_flat = valid_mask_h.view(T, -1).float()

        # Count valid pixels for each timestamp.
        count = valid_mask_h_flat.sum(dim=1)  # shape: (T,)

        # Compute per-timestamp means over valid pixels.
        sum_x = (x * valid_mask_h_flat).sum(dim=1)
        sum_y = (y * valid_mask_h_flat).sum(dim=1)
        mean_x = sum_x / torch.clamp(count, min=1)
        mean_y = sum_y / torch.clamp(count, min=1)

        # Compute centered values.
        diff_x = x - mean_x.unsqueeze(1)
        diff_y = y - mean_y.unsqueeze(1)

        # Compute covariance and variances over valid pixels.
        cov = ((diff_x * diff_y) * valid_mask_h_flat).sum(dim=1) / torch.clamp(count, min=1)
        var_x = ((diff_x ** 2) * valid_mask_h_flat).sum(dim=1) / torch.clamp(count, min=1)
        var_y = ((diff_y ** 2) * valid_mask_h_flat).sum(dim=1) / torch.clamp(count, min=1)
        corr_h = cov / (torch.sqrt(var_x * var_y) + 1e-8)

        # For timestamps where the overall image valid fraction is below 50% or where no valid pixels exist,
        # set the correlation to NaN.
        invalid = (valid_fraction < 0.5) | (count == 0)
        corr_h[invalid] = float('nan')
        results[f"horizontal_shift_{shift}"] = corr_h

        # --- Vertical shift (shift along height; for an image (H, W) this is dim=0) ---
        # For a tensor of shape (T, H, W), vertical shift corresponds to shifting along dim=1.
        shifted_v = torch.roll(sample, shifts=-shift, dims=1)
        valid_mask_v = (~torch.isnan(sample)) & (~torch.isnan(shifted_v))
        x = sample.view(T, -1)
        y = shifted_v.view(T, -1)
        valid_mask_v_flat = valid_mask_v.view(T, -1).float()

        count = valid_mask_v_flat.sum(dim=1)
        sum_x = (x * valid_mask_v_flat).sum(dim=1)
        sum_y = (y * valid_mask_v_flat).sum(dim=1)
        mean_x = sum_x / torch.clamp(count, min=1)
        mean_y = sum_y / torch.clamp(count, min=1)

        diff_x = x - mean_x.unsqueeze(1)
        diff_y = y - mean_y.unsqueeze(1)
        cov = ((diff_x * diff_y) * valid_mask_v_flat).sum(dim=1) / torch.clamp(count, min=1)
        var_x = ((diff_x ** 2) * valid_mask_v_flat).sum(dim=1) / torch.clamp(count, min=1)
        var_y = ((diff_y ** 2) * valid_mask_v_flat).sum(dim=1) / torch.clamp(count, min=1)
        corr_v = cov / (torch.sqrt(var_x * var_y) + 1e-8)

        invalid = (valid_fraction < 0.5) | (count == 0)
        corr_v[invalid] = float('nan')
        results[f"vertical_shift_{shift}"] = corr_v

    return results

def compute_intra_channel_corr(sample: torch.Tensor) -> torch.Tensor:
    """
    Compute the correlation matrix of channels over time.
    
    For an input tensor of shape (C, T, H, W), the function first computes, for each timestamp t,
    a channel correlation matrix as follows:
    
      1. Rearrange the data to (T, C, H, W) and flatten the spatial dimensions to (C, N), where N = H*W.
      2. For each channel, determine the valid fraction of pixels. If a channel has less than 50% valid 
         (non-NaN) pixels at time t, it is considered invalid and any correlation with it is set to NaN.
      3. For each channel pair (i, j), compute a mask of pixels where both channels have valid data. 
         If the fraction of common valid pixels is less than 50%, the correlation for that pair is NaN.
      4. Otherwise, compute the Pearson correlation (with a small epsilon added for stability).
      5. Diagonals are set to 1 for channels that are valid, and NaN otherwise.
    
    Finally, the per-timestamp correlation matrices are averaged (ignoring NaNs) over time.
    
    Returns:
        A tensor of shape (C, C) with the averaged correlation values among channels.
    """
    # sample has shape (C, T, H, W)
    C, T, H, W = sample.shape
    N = H * W
    # Rearrange to (T, C, H, W) to iterate over time
    sample_t = sample.transpose(0, 1)  # shape: (T, C, H, W)
    
    corr_matrices = []  # To store per-timestamp correlation matrices
    
    for t in range(T):
        X = sample_t[t]  # shape: (C, H, W)
        X_flat = X.reshape(C, -1)  # shape: (C, N)
        
        # Compute valid fraction for each channel (per timestamp)
        valid_frac = (~torch.isnan(X_flat)).float().mean(dim=1)  # shape: (C,)
        
        # Initialize a correlation matrix for timestamp t.
        corr_matrix = torch.empty((C, C), device=sample.device)
        corr_matrix.fill_(float('nan'))
        
        # Compute the correlation for each pair of channels.
        for i in range(C):
            for j in range(i, C):
                # If either channel has less than 50% valid pixels, mark correlation as NaN.
                if valid_frac[i] < 0.5 or valid_frac[j] < 0.5:
                    corr_val = float('nan')
                else:
                    xi = X_flat[i]  # shape: (N,)
                    xj = X_flat[j]  # shape: (N,)
                    # Only use pixels where both channels are valid.
                    valid_mask = (~torch.isnan(xi)) & (~torch.isnan(xj))
                    if valid_mask.float().mean() < 0.5 or valid_mask.sum() == 0:
                        corr_val = float('nan')
                    else:
                        xi_valid = xi[valid_mask]
                        xj_valid = xj[valid_mask]
                        # Compute Pearson correlation
                        mean_i = xi_valid.mean()
                        mean_j = xj_valid.mean()
                        std_i = xi_valid.std() + 1e-8
                        std_j = xj_valid.std() + 1e-8
                        cov = ((xi_valid - mean_i) * (xj_valid - mean_j)).mean()
                        corr_val = cov / (std_i * std_j)
                corr_matrix[i, j] = corr_val
                corr_matrix[j, i] = corr_val
        
        # Ensure diagonal entries are 1 if the channel is valid, otherwise NaN.
        for i in range(C):
            if valid_frac[i] >= 0.5:
                corr_matrix[i, i] = 1.0
            else:
                corr_matrix[i, i] = float('nan')
                
        corr_matrices.append(corr_matrix)
    # Stack all per-timestamp matrices: shape (T, C, C)
    corr_stack = torch.stack(corr_matrices, dim=0)
    return corr_stack


def compute_power_metric(sample: torch.Tensor) -> torch.Tensor:
    """
    Compute a power-spectrum metric on each channel and each time step without averaging over channels or time.
    For each channel and each time step, compute the 2D FFT (after subtracting the spatial mean) and then return
    the mean power (i.e. the mean of the magnitude squared of the FFT) for that image.
    sample: Tensor of shape (C, T, H, W)
    Returns a tensor of shape (C, T) with the computed power values.
    """
    valid_fraction = (~torch.isnan(sample)).float().mean().item()
    if valid_fraction < 0.5:
        return torch.tensor(float('nan'), device=sample.device)

    C, T, H, W = sample.shape
    power_values = torch.empty((C, T), device=sample.device)
    for c in range(C):
        for t in range(T):
            img = sample[c, t]
            if torch.isnan(img).all():
                power_values[c, t] = float('nan')
            else:
                # Subtract spatial mean ignoring NaN values
                img_mean = torch.nanmean(img)
                img_zero = img - img_mean
                fft_result = torch.fft.fft2(img_zero)
                power = fft_result.abs()**2
                power_values[c, t] = torch.nanmean(power)
    return power_values


def compute_crps(forecasts: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    """
    Compute the Continuous Ranked Probability Score (CRPS) for an ensemble forecast.

    Args:
        forecasts (torch.Tensor): Ensemble forecasts of shape [M, ...],
                                  where M is the number of ensemble members and
                                  ... represents additional dimensions.
        ground_truth (torch.Tensor): Ground truth values with shape matching a single forecast (i.e. [...]).
    
    Returns:
        torch.Tensor: The averaged CRPS over all elements.
    
    The CRPS is computed as:
        CRPS = (1/M) * sum(|x_m - y|) - (1/(2M^2)) * sum_{m,n}(|x_m - x_n|)
    where x_m are the ensemble forecast samples and y is the ground truth.
    """
    M = forecasts.shape[0]
    
    # Term 1: average absolute error between each ensemble member and the ground truth.
    term1 = torch.mean(torch.abs(forecasts - ground_truth), dim=0)
    
    # Term 2: average pairwise absolute differences between ensemble members.
    diff = torch.abs(forecasts.unsqueeze(0) - forecasts.unsqueeze(1))  # shape [M, M, ...]
    term2 = torch.mean(diff, dim=(0, 1))
    
    # Compute CRPS per element and then average over all elements.
    crps = term1 - 0.5 * term2
    return torch.mean(crps)