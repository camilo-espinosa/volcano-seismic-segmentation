import numpy as np


def generate_overlapping_batches(
    data, window_size=8192, stride=8192, batch_size=16, noise_norm=1
):
    num_channels, total_length = data.shape
    start_indices = np.arange(0, total_length - window_size + 1, stride)

    for i in range(0, len(start_indices), batch_size):
        batch = []
        for start in start_indices[i : i + batch_size]:
            window = data[:, start : start + window_size].copy()
            normalizing_value = np.max([window[1:9].max(), noise_norm])
            window[1:9] = window[1:9] / normalizing_value
            batch.append(window)

        # Shape: (batch_size, 14, window_size)
        yield np.stack(batch)
