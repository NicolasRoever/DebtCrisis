"""Utilities used in various parts of the project."""

import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def _make_missing_values_heatmap(data, data_name, index=None):
    """Create a heatmap to visualize missing values in a DataFrame."""
    if index is not None:
        data = data.set_index(index)

    plt.figure(figsize=(10, 6))
    sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values in Dataset " + data_name)
    plt.show()


def _check_dataframe_rows(df: pd.DataFrame, num_rows: int):
    """Raises an error if DataFrame does not have a certain number of rows."""
    if len(df) != num_rows:
        msg = f"DataFrame should have {num_rows} rows, but it has {len(df)} rows."
        raise ValueError(
            msg,
        )
