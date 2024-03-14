import pandas as pd
import statsmodels.formula.api as smf


from debt_crisis.config import BLD
from debt_crisis.utilities import _make_missing_values_heatmap


def create_dataset_step_one_regression(
    bond_yield_data,
    debt_to_gdp_data,
    sentiment_index_data,
    ratings_data,
    gdp_data,
    current_account_data,
    stoxx_data,
):
    """Merge the bond yield data, the debt to GDP data, and the sentiment index data to
    prepare for the first step regression."""
    # Merge the bond yield data and the debt to GDP data
    merged_data = pd.merge(
        bond_yield_data, debt_to_gdp_data, on=["Date", "Country"], how="left"
    )

    _make_missing_values_heatmap(merged_data, "Bond Yields Merged to Debt to GDP")

    # Merge the sentiment index data
    merged_data = pd.merge(
        merged_data, sentiment_index_data, on=["Date", "Country"], how="left"
    )

    # Merge the ratings data

    merged_data = pd.merge(
        merged_data, ratings_data, on=["Date", "Country"], how="left"
    )

    # Merge the GDP data

    merged_data = pd.merge(merged_data, gdp_data, on=["Date", "Country"], how="left")

    # Merge current account data

    merged_data = pd.merge(
        merged_data, current_account_data, on=["Date", "Country"], how="left"
    )

    # Merge stoxx data

    merged_data = pd.merge(merged_data, stoxx_data, on=["Date"], how="left")

    _make_missing_values_heatmap(merged_data, "First Step Regression Data")

    return merged_data


def run_first_step_regression(data):
    """Run the first step regression."""
    # Define the regression formula
    formula = "McDonald_Sentiment_Index ~ Debt_to_GDP_Ratio + Rating_Numeric_FIS + GDP_Growth + GDP_Growth_Lead + Current_Account_Balance + stoxx50 + vstoxx"

    # Run the regression
    model = smf.ols(formula, data=data).fit()

    return model


def run_second_step_regression(data):
    """This function takes as input the data produced from the first step regression
    tasks and runs the second step regression."""

    formula = "Bond_Yield ~ Debt_to_GDP_Ratio + Rating_Numeric_FIS + GDP_Growth + GDP_Growth_Lead + Current_Account_Balance + stoxx50 + vstoxx"

    model = smf.ols(formula, data=data).fit()

    return model


def run_third_step_regression(data):
    """This function takes as input the data produced from the second step regression
    tasks and runs the third step regression."""

    formula = "Residuals_Step_Two_Regression ~ Residuals_Step_One_Regression"

    model = smf.ols(formula, data=data).fit()

    return model
