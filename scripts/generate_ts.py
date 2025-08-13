from pathlib import Path
from typing import List, Union

import pandas as pd
import numpy as np
from gluonts.dataset.arrow import ArrowWriter


def read_transform_data(csv_path: Union[str, Path], test_mode: bool = False) -> List[np.ndarray]:
    """
    Read CSV data and transform it into time series format.
    Each column represents an independent time series.
    
    Args:
        csv_path: Path to the CSV file
        test_mode: If True, only process the first column
        
    Returns:
        List of numpy arrays, each representing a time series
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Convert to time series format
    # Each column becomes a time series
    time_series = []
    
    if test_mode:
        # In test mode, only process the first column
        col = df.columns[0]
        series = df[col].dropna().values
        if len(series) > 0:
            time_series.append(series.astype(np.float32))
        print(f"🧪 TEST MODE: Processing only first column '{col}' with {len(series)} values")
    else:
        # Process all columns
        for col in df.columns:
            # Remove any NaN values and convert to numpy array
            series = df[col].dropna().values
            if len(series) > 0:
                time_series.append(series.astype(np.float32))
        print(f"📊 Processing all {len(df.columns)} columns")
    
    return time_series


def convert_to_arrow(
    path: Union[str, Path],
    time_series: Union[List[np.ndarray], np.ndarray],
    compression: str = "lz4",
):
    """
    Store a given set of series into Arrow format at the specified path.

    Input data can be either a list of 1D numpy arrays, or a single 2D
    numpy array of shape (num_series, time_length).
    """
    assert isinstance(time_series, list) or (
        isinstance(time_series, np.ndarray) and
        time_series.ndim == 2
    )

    # Set an arbitrary start time
    start = np.datetime64("2000-01-01 00:00", "s")

    dataset = [
        {"start": start, "target": ts} for ts in time_series
    ]

    ArrowWriter(compression=compression).write_to_file(
        dataset,
        path=path,
    )


if __name__ == "__main__":
    # Generate 20 random time series of length 1024
    time_series = [np.random.randn(1024) for i in range(1)]
    
    # Save the time series to a file csv
    df = pd.DataFrame(time_series).T
    df.to_csv("./scripts/noise-data.csv", index=False)
    print(df.shape)
    print(df.head().T)
    # Convert to GluonTS arrow format
    convert_to_arrow("./scripts/noise-data.arrow", time_series=time_series)


    # run the training script
    # python scripts/training/train.py --config scripts/training/configs/chronos-t5-tiny.yaml   