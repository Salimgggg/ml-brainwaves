from scipy.signal import butter, lfilter
import numpy as np

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def reshape_signal_into_windows(x, sample_rate, window_duration_in_seconds):

    # Calculate the number of samples in one window
    window_size = int(window_duration_in_seconds * sample_rate)
    
    # Ensure the total length of x is a multiple of window_size
    total_samples = x.shape[-1]
    if total_samples % window_size != 0:
        # Truncate or pad x to make it divisible by window_size
        x = x[..., :total_samples - (total_samples % window_size)]
    # Reshape x into (C, T, window)
    reshaped_x = x.reshape(x.shape[0], -1, window_size)

    return reshaped_x


def reshape_targets_for_windows(x, sample_rate, window_duration_in_seconds): 
    """
    Reshape the targets to match the reshaped signal.
    For example, if we have a 4-second window, and we want a target 
    value for every 2 seconds, we need to ensure the length of `x` is divisible by 2.
    """
    factor = window_duration_in_seconds // 2
    total_targets = x.shape[1]
    
    # Ensure divisibility by truncating if necessary
    remainder = total_targets % factor
    if remainder != 0:
        x = x[:, :total_targets - remainder]
    
    reshaped_x = x.reshape(x.shape[0], -1, factor)

    return reshaped_x


    

