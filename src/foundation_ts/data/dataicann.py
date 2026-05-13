from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


SIGNAL_COLUMNS = ["ac", "ax", "ay", "ir", "is"]


def load_dataicann(mat_file_path: str | Path = "data/raw/dataicann/dataicann.mat") -> pd.DataFrame:
    """Load all Dataicann experiments from a single .mat file.

    Returns a DataFrame with columns: ``experiment``, ``ac``, ``ax``, ``ay``,
    ``ir`` and ``is``.
    """
    data = loadmat(mat_file_path)

    samples = []
    experiment_ids = []

    for experiment_id, package in enumerate(data["z"][0]):
        samples.append(package)
        experiment_ids.append(np.repeat(experiment_id, package.shape[0]))

    samples = np.vstack(samples)
    experiment_ids = np.concatenate(experiment_ids)

    df = pd.DataFrame(samples, columns=SIGNAL_COLUMNS)
    df.insert(0, "experiment", experiment_ids)

    return df


def make_dataicann_windows(
    df: pd.DataFrame,
    window_size: int = 400,
    stride: int = 1,
    signal_columns: list[str] | None = None,
) -> np.ndarray:
    """Create multivariate windows from a Dataicann DataFrame.

    Returns an array with shape ``(n_windows, window_size, n_variables)``.
    """
    signal_columns = signal_columns or SIGNAL_COLUMNS

    if window_size <= 0:
        raise ValueError("window_size must be greater than 0.")

    if stride <= 0:
        raise ValueError("stride must be greater than 0.")

    missing_columns = [column for column in signal_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Signal columns not found in the DataFrame: {missing_columns}")

    values = df[signal_columns].to_numpy(dtype=float)
    if len(values) < window_size:
        raise ValueError(
            f"Not enough rows to build one window of size {window_size}. "
            f"Available rows: {len(values)}"
        )

    starts = range(0, len(values) - window_size + 1, stride)
    return np.stack([values[start : start + window_size] for start in starts])
