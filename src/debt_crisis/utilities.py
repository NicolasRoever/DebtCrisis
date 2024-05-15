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


def check_if_dataframe_column_is_datetime_type(data_column):
    if not pd.api.types.is_datetime64_any_dtype(data_column):
        raise ValueError("The 'Date' column must be of datetime type.")


def _check_for_missing_values_in_dataframe_column(data, column_name):
    if data[column_name].isnull().any():
        raise ValueError(
            f"The DataFrame has missing values in the {column_name} column."
        )


def _name_sentiment_index_output_file(name, configuration_settings, suffix):
    """Create a name for the sentiment index output file."""

    file_name = f"{name}_{configuration_settings['sentiment_index_calculation_method']}_{configuration_settings['words_in_environment']}_{suffix}"

    return file_name


def _create_dictionary_of_coefficients_with_stars(
    coefficient_names, fitted_model, significance_levels=[0.01, 0.05, 0.1], rounding=3
):
    """Create a dictionary of coefficients with stars based on significance levels.

    Args:
        coefficient_names (list): List of names of the coefficients.
        fitted_model (statsmodels.regression.linear_model.RegressionResultsWrapper): Fitted model.
        significance_levels (list): List of significance levels.
        rounding (int): Number of decimal places to round the coefficients to.

    Returns:
        dict: Dictionary of coefficients with stars.
        Keys are the parameter names, values are the coefficients with stars as strings.

    """

    # Define the stars for each significance level
    stars = ["***", "**", "*"]

    # Initialize an empty dictionary to store the coefficients with stars
    coefficients_with_stars = {}

    # Loop over each parameter
    for param in coefficient_names:
        # Get the coefficient value
        coefficient = fitted_model.params.get(param, pd.NA)

        # Get the p-value of the coefficient
        p_value = fitted_model.pvalues.get(param, 1)

        # Format the coefficient as a float
        coefficient_str = f"{coefficient:.{rounding}f}"

        # Add stars to the coefficient based on its p-value
        for level, star in zip(significance_levels, stars):
            if p_value < level:
                coefficient_str += star
                break

        # Add the coefficient with stars to the dictionary
        coefficients_with_stars[param] = coefficient_str

        return coefficients_with_stars
