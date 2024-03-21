import pandas as pd
import statsmodels.formula.api as smf


from debt_crisis.config import (
    BLD,
    EUROPEAN_COUNTRIES_IN_QUARTERLY_MACROECONOMIC_VARIABLES,
)
from debt_crisis.utilities import _make_missing_values_heatmap, _check_dataframe_rows


def create_dataset_step_one_regression_quarterly_data(
    quarterly_macroeconomic_variables, sentiment_index
):
    """THis function creates the dataset for the step 1 regression by merging the
    quarterly macroeconomic variables data with the sentiment index."""

    # Convert the 'Date' column to datetime
    quarterly_macroeconomic_variables["Date"] = pd.to_datetime(
        quarterly_macroeconomic_variables["Date"]
    )

    # Now perform the filtering
    quarterly_macroeconomic_variables_filter = quarterly_macroeconomic_variables[
        (
            quarterly_macroeconomic_variables["Country"].isin(
                EUROPEAN_COUNTRIES_IN_QUARTERLY_MACROECONOMIC_VARIABLES
            )
        )
        & (quarterly_macroeconomic_variables["Date"] >= pd.to_datetime("2002-01-01"))
    ]

    # Merge the quarterly macroeconomic variables data with the sentiment index
    merged_data = pd.merge(
        quarterly_macroeconomic_variables_filter,
        sentiment_index,
        on=["Date", "Country"],
        how="left",
        validate="one_to_one",
    )

    _check_dataframe_rows(
        merged_data, quarterly_macroeconomic_variables_filter.shape[0]
    )

    merged_data = merged_data.sort_values(by=["Country", "Date"])

    merged_data["GDP_in_Current_Prices_Growth"] = merged_data.groupby("Country")[
        "GDP_in_USD_Current_Prices"
    ].pct_change()

    merged_data["GDP_in_Current_Prices_Growth_Lead"] = merged_data.groupby("Country")[
        "GDP_in_Current_Prices_Growth"
    ].shift(-1)

    _make_missing_values_heatmap(
        merged_data, "First Step Regression Data", index="Country"
    )

    return merged_data


def run_step_one_regression_quarterly_data(data):
    """This function runs the first step regression using the data produced from the
    first step regression tasks."""

    # Define the regression formula
    formula = "McDonald_Sentiment_Index ~ Q('Public_Debt_as_%_of_GDP')+ GDP_in_Current_Prices_Growth + GDP_in_Current_Prices_Growth_Lead + Current_Account_in_USD + VIX_Daily_Close_Quarterly_Mean"

    # Run the regression
    model = smf.ols(formula, data=data).fit()

    return model


def run_step_two_regression_quarterly_data(data):
    """This function runs the second step regression."""

    formula = "Q('10y_Maturity_Bond_Yield') ~ Q('Public_Debt_as_%_of_GDP')+ GDP_in_Current_Prices_Growth + GDP_in_Current_Prices_Growth_Lead + Current_Account_in_USD + VIX_Daily_Close_Quarterly_Mean + Q('Eurostat_CPI_Annualised Growth_Rate') + NASDAQ_Daily_Close_Quarterly_Mean + Q('3_Month_US_Treasury_Yield_Quarterly_Mean')"

    model = smf.ols(formula, data=data).fit()

    return model


def run_step_three_regression_quarterly_data(data):
    """This function runs the third step regression."""

    formula = "Residuals_Step_Two_Regression ~ Residuals_Step_One_Regression"

    model = smf.ols(formula, data=data).fit()

    return model


# ------------------------------------------------------------------------------------------
# Eurostat Data Implementation


def create_dataset_step_one_regression_eurostat_data(
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


def run_first_step_regression_eurostat(data):
    """Run the first step regression."""
    # Define the regression formula
    formula = "McDonald_Sentiment_Index ~ Debt_to_GDP_Ratio + GDP_Growth + GDP_Growth_Lead + Current_Account_Balance + stoxx50 + vstoxx"

    # Run the regression
    model = smf.ols(formula, data=data).fit()

    return model


def run_second_step_regression_eurostat(data):
    """This function takes as input the data produced from the first step regression
    tasks and runs the second step regression."""

    formula = "Bond_Yield ~ Debt_to_GDP_Ratio + Rating_Numeric_FIS + GDP_Growth + GDP_Growth_Lead + Current_Account_Balance + stoxx50 + vstoxx"

    model = smf.ols(formula, data=data).fit()

    return model


def run_third_step_regression_eurostat(data):
    """This function takes as input the data produced from the second step regression
    tasks and runs the third step regression."""

    formula = "Residuals_Step_Two_Regression ~ Residuals_Step_One_Regression"

    model = smf.ols(formula, data=data).fit()

    return model
