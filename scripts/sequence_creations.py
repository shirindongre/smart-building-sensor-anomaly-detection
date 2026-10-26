import numpy as np

def create_sequences(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Convert scaled 2D array into overlapping 3D windows for LSTM input.

    Parameters
    ----------
    data        : Scaled 2D array (n_samples, n_features)
    window_size : Number of timesteps per sequence

    Returns
    -------
    sequences : 3D array (n_windows, window_size, n_features)
        n_windows = len(data) - window_size + 1. Window ``i`` covers rows ``i .. i + window_size - 1``
        (so it *ends* at row ``i + window_size - 1``). An empty (0, window_size, n_features) array is
        returned when there are fewer than ``window_size`` rows.
    """
    data = np.asarray(data)
    n_windows = len(data) - window_size + 1
    if n_windows <= 0:
        return np.empty((0, window_size, data.shape[1]), dtype=data.dtype)
    return np.stack([data[i : i + window_size] for i in range(n_windows)])