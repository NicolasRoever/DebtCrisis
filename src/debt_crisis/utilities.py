"""Utilities used in various parts of the project."""

import yaml
import matplotlib.pyplot as plt
import seaborn as sns


def _make_missing_values_heatmap(data, data_name):
    """Create a heatmap to visualize missing values in a DataFrame."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values in Dataset " + data_name)
    plt.show()
