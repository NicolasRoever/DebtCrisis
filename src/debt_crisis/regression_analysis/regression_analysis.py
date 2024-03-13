from debt_crisis.config import BLD

import pandas as pd


def create_dataset_step_one_regression(
    bond_yield_data, debt_to_gdp_data, sentiment_index_data
):
    # Merge the bond yield data and the debt to GDP data
    merged_data = pd.merge(
        bond_yield_data, debt_to_gdp_data, on=["Date", "Country"], how="inner"
    )

    # Merge the sentiment index data
    merged_data = pd.merge(
        merged_data, sentiment_index_data, on=["Date", "Country"], how="inner"
    )

    return merged_data
