import cv2
import numpy as np
import torch
from functools import partial
from multiprocessing import Pool

# https://www.mdpi.com/2072-4292/14/14/3372



import cv2
import numpy as np
from functools import partial
from multiprocessing import Pool

def compute_optical_flow(model, frame1, frame2):
    """
    Computes the optical flow between two frames using the TV-L1 algorithm (CPU-only).
    """
    flow = model.calc(frame1, frame2, None)
    return flow

def warp_image(frame, flow):
    """
    Warps an image using the computed optical flow.
    """
    h, w = frame.shape[:2]
    
    # Generate coordinate grid
    flow_map = np.meshgrid(np.arange(w), np.arange(h))
    flow_map = np.array(flow_map).transpose(1, 2, 0)
    
    # Add flow vectors
    flow_map = flow_map.astype(np.float32) + flow
    
    # Warp the image
    warped_frame = cv2.remap(
        frame, 
        flow_map, 
        None, 
        cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=255)
    #                             borderMode=cv2.BORDER_TRANSPARENT, borderValue=255)
    
    
    return warped_frame

def forecast_future_frame(model, frames, num_future_frames):
    """
    Forecasts future frames using optical flow.
    """
    forecasted_frames = []
    
    # Compute optical flow between last two frames
    flow = compute_optical_flow(model, frames[-1], frames[-2])
    
    # Predict future frames
    last_frame = frames[-1]
    for _ in range(num_future_frames):
        future_frame = warp_image(last_frame, flow)
        forecasted_frames.append(future_frame)
        last_frame = future_frame
    
    return np.array(forecasted_frames)


def _forecast_single_channel(x, conf, n_steps, j):
    """
    Forecasts a single channel using TV-L1 optical flow.
    """
    # Initialize TV-L1 Optical Flow **inside each process** (fixes pickling issue)
    model = cv2.optflow.createOptFlow_DualTVL1()

    # Set TV-L1 parameters
    model.setTau(conf['tau'])
    model.setLambda(conf['lambda'])
    model.setTheta(conf['theta'])
    model.setOuterIterations(conf['outer_iterations'])
    model.setScaleStep(conf['scale_step'])
    model.setGamma(conf['gamma'])
    model.setWarpingsNumber(conf['warps'])  
    model.setEpsilon(conf['epsilon'])
    model.setUseInitialFlow(False)

    # Extract single channel
    prev_frames = x[j, :, :, :, None]  # Keep as NumPy array
    min_ = prev_frames.min()
    max_ = prev_frames.max()
    # Normalize frames to [0, 254] and convert to uint8
    prev_frames = ((prev_frames - min_) / (max_ - min_)) * 254
    prev_frames = np.clip(prev_frames, 0, 254).astype(np.uint8)  

    # Predict future frames
    future_frames_ = forecast_future_frame(model, prev_frames, n_steps)
    
    # Convert back to original scale
    future_frames_ = future_frames_.astype(np.float32)
    
    future_frames_[future_frames_ == 255] = np.nan

    future_frames_ = (future_frames_ / 254) * (max_ - min_) + min_

    return future_frames_


# Example Usage
def tvl1_forecast(x, model_conf, n_steps):
    """
    Main function to forecast future frames using multiprocessing.
    """
    forecast_single_channel = partial(_forecast_single_channel, x, model_conf, n_steps)

    with Pool() as p:
        future_frames = p.map(forecast_single_channel, np.arange(11))
    
    future_frames = np.stack(future_frames, axis=0)
    return future_frames


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn")  # ✅ Fix multiprocessing issues on Linux/macOS

    model_conf = {
        'tau': 0.15,
        'epsilon': 0.005,
        'gamma': 0,
        'warps': 10, 
        'lambda': 0.05,
        'outer_iterations': 20,
        'inner_iterations': 20,
        'theta': 0.3,
        'nscales': 5,
        'median_filtering': 1,
        'scale_step': 0.5,
    }

    # Random test data
    x = np.random.randn(11, 2, 256, 256)
    future_frames = tvl1_forecast(x, model_conf, n_step=12)

    print("Forecasted future frames shape:", future_frames.shape)
