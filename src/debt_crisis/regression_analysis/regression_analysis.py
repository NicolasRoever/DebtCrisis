import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import re
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.iolib.summary2 import summary_col


from debt_crisis.config import (
    BLD,
    EUROPEAN_COUNTRIES_IN_QUARTERLY_MACROECONOMIC_VARIABLES,
    EVENT_STUDY_COUNTRIES,
)
from debt_crisis.utilities import (
    _make_missing_values_heatmap,
    _check_dataframe_rows,
    check_if_dataframe_column_is_datetime_type,
    _check_for_missing_values_in_dataframe_column,
)

from debt_crisis.regression_analysis.event_study import (
    extract_event_study_coefficients_from_event_study_regression_data,
)


def create_dataset_for_factor_model_regression(
    bond_yield_data,
    debt_to_gdp_data,
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


# ------------------------------------------------------------------------------------------
# Deprecated Functions


def merge_event_study_coefficients_and_exuberance_index_data(
    event_study_output_data, exuberance_index_data
):
    event_study_coefficient_data = (
        extract_event_study_coefficients_from_event_study_regression_data(
            event_study_output_data
        )
    )

    merged_data = pd.merge(
        event_study_coefficient_data,
        exuberance_index_data,
        on=["Date", "Country"],
        how="left",
        validate="one_to_one",
    )

    _check_dataframe_rows(merged_data, event_study_coefficient_data.shape[0])

    return merged_data


def run_regression_exuberance_indicator_vs_event_study_coefficients(data, country):
    """This function runs the regression of the exuberance indicator vs the event study
    coefficients for a given country.

    Args: data (pd.DataFrame): The dataset for the event study regression.

    Returns: tuple: The first element is the regression model and the second element is a DataFrame with the fitted values.

    """

    data_filter = data.loc[data["Country"] == country, :]

    if data.empty:
        return None

    formula = "Coefficient ~ Residuals_Exuberance_Regression:Country"

    model = smf.ols(formula, data=data_filter).fit()

    fitted_values = pd.DataFrame()
    fitted_values["Date"] = data_filter["Date"]
    fitted_values["Fitted_Values"] = model.fittedvalues
    fitted_values["Country"] = country

    return model, fitted_values


def plot_unfounded_spreads_vs_unfounded_sentiment(
    event_study_coefficients_data,
    unfounded_sentiment_data,
    country,
    color_scheme=["#3c5488", "#e64b35", "#4dbbd5", "#00a087", "#f39b7f"],
):
    merged_data = merge_event_study_coefficients_and_exuberance_index_data(
        event_study_coefficients_data, unfounded_sentiment_data
    )

    country_data = merged_data[merged_data["Country"] == country]

    # Set the style of the plot
    sns.set_style("white")

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(8, 5))

    (line1,) = ax1.plot(
        country_data["Date"],
        country_data["Coefficient"],
        linestyle="-",
        color=color_scheme[0],
        label="Unfounded Spread (Event Study Coefficient)",
    )

    ax2 = ax1.twinx()

    (line2,) = ax2.plot(
        country_data["Date"],
        country_data["Residuals_Exuberance_Regression"],
        linestyle="-",
        color=color_scheme[1],
        label="Unfounded Sentiment",
    )

    # Set the labels
    ax1.set_xlabel("Date", fontsize=14)
    ax1.set_ylabel("Unfounded Spread (Event Study Coefficient)", fontsize=14)
    ax2.set_ylabel("Unfounded Sentiment", fontsize=14)

    ax2.invert_yaxis()  # Invert the right y-axis

    # Create a legend for both lines
    plt.legend(
        [line1, line2],
        ["Unfounded Spread (Event Study Coefficient)", "Unfounded Sentiment"],
    )

    # Add a vertical line at 19. October 2010
    ax1.axvline(pd.to_datetime("2010-10-19"), color="grey", linestyle="--")
    ax1.text(pd.to_datetime("2010-10-19"), ax1.get_ylim()[1], "Deauville", ha="right")

    # Add a vertivcal line on 6th september 2012
    ax1.axvline(pd.to_datetime("2012-09-06"), color="grey", linestyle="--")
    ax1.text(
        pd.to_datetime("2012-09-06"), ax1.get_ylim()[1], "OMT Programme", ha="right"
    )

    # Keep only the y-axis and x-axis
    sns.despine(left=False, bottom=False, right=False, top=True)

    # Use LaTeX style for the font
    plt.rc("text", usetex=True)

    plt.close(fig)

    return fig


def plot_unfounded_spreads_vs_daily_sentiment_index(
    event_study_coefficients_data,
    daily_sentiment_data,
    country,
    color_scheme=["#3c5488", "#e64b35", "#4dbbd5", "#00a087", "#f39b7f"],
):
    clean_event_study_coefficients_data = (
        extract_event_study_coefficients_from_event_study_regression_data(
            event_study_coefficients_data
        )
    )
    event_study_coefficients_data_country = clean_event_study_coefficients_data[
        clean_event_study_coefficients_data["Country"] == country
    ]

    daily_sentiment_data_country = daily_sentiment_data[
        daily_sentiment_data["Country"] == country
    ]

    # Set the style of the plot
    sns.set_style("white")

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(8, 5))

    (line1,) = ax1.plot(
        event_study_coefficients_data_country["Date"],
        event_study_coefficients_data_country["Coefficient"],
        linestyle="-",
        color=color_scheme[0],
        label="Unfounded Spread (Event Study Coefficient)",
    )

    ax2 = ax1.twinx()

    (line2,) = ax2.plot(
        daily_sentiment_data_country["Date"],
        daily_sentiment_data_country["McDonald_Sentiment_Index"],
        linestyle="-",
        color=color_scheme[1],
        label="Daily Sentiment",
    )

    # Set the labels
    ax1.set_xlabel("Date", fontsize=14)
    ax1.set_ylabel("Unfounded Spread (Event Study Coefficient)", fontsize=14)
    ax2.set_ylabel("Unfounded Sentiment", fontsize=14)

    ax2.invert_yaxis()  # Invert the right y-axis

    # Create a legend for both lines
    plt.legend(
        [line1, line2],
        ["Unfounded Spread (Event Study Coefficient)", "Daily Sentiment"],
    )

    # Add a vertical line at 19. October 2010
    ax1.axvline(pd.to_datetime("2010-10-19"), color="grey", linestyle="--")
    ax1.text(pd.to_datetime("2010-10-19"), ax1.get_ylim()[1], "Deauville", ha="right")

    # Add a vertivcal line on 6th september 2012
    ax1.axvline(pd.to_datetime("2012-09-06"), color="grey", linestyle="--")
    ax1.text(
        pd.to_datetime("2012-09-06"), ax1.get_ylim()[1], "OMT Programme", ha="right"
    )

    # Keep only the y-axis and x-axis
    sns.despine(left=False, bottom=False, right=False, top=True)

    # Use LaTeX style for the font
    plt.rc("text", usetex=True)

    plt.close(fig)

    return fig


def run_regression_exuberance_indicator_vs_event_study_coefficients_for_all_countries(
    data,
):
    """This function runs the regression of the exuberance indicator vs the event study
    coefficients for all countries."""

    # Initialize an empty DataFrame to store the estimates
    estimates = pd.DataFrame(
        columns=["Country", "Coefficient", "Std Err", "t", "P>|t|", "R_Squared"]
    )

    # Initialize empty dataframe to store fitted values
    fitted_values = pd.DataFrame()

    for country in data.Country.unique():
        if country == "slovenia":
            continue  # THis is a bad quick fix!!!

        # Run the regression for the current country
        (
            model,
            fitted_values_for_country,
        ) = run_regression_exuberance_indicator_vs_event_study_coefficients(
            data, country
        )

        estimates = append_regression_estimates_to_dataframe(estimates, model, country)

        fitted_values = pd.concat(
            [fitted_values, fitted_values_for_country], ignore_index=True
        )

    return estimates, fitted_values


def append_regression_estimates_to_dataframe(estimates, model, country):
    """This function extracts the estimated regression parameters and appends them to a
    DataFrame."""

    # Extract the coefficient, standard error, t statistic, and p-value
    coef = model.params[1]
    std_err = model.bse[1]
    t = model.tvalues[1]
    p_value = model.pvalues[1]
    r_squared = model.rsquared

    # Append the results to the DataFrame
    estimates = pd.concat(
        [
            estimates,
            pd.DataFrame(
                {
                    "Country": [country],
                    "Coefficient": [coef],
                    "Std Err": [std_err],
                    "t": [t],
                    "P>|t|": [p_value],
                    "R_Squared": [r_squared],
                },
                index=[0],
            ),
        ],
        ignore_index=True,
    )

    return estimates


def plot_fitted_values_from_exuberance_unfounded_bond_yield_regression(
    data,
    countries,
    color_scheme=["#3c5488", "#e64b35", "#4dbbd5", "#00a087", "#f39b7f"],
):
    # Set the style of the plot
    sns.set_style("white")

    # Create the plot
    fig = plt.figure(figsize=(10, 6))

    # Loop over the list of countries
    for i, country in enumerate(countries):
        # Filter the data for the given country
        country_data = data[data["Country"] == country]
        country_data = country_data.sort_values("Date")

        # Plot the data for the country
        plt.plot(
            country_data["Date"],
            country_data["Fitted_Values"],
            marker="o",
            label=country,
            color=color_scheme[i % len(color_scheme)],
        )

    # Set the title and labels
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Fitted Values from Regression", fontsize=14)

    # Keep only the y-axis and x-axis
    sns.despine(left=False, bottom=False, right=True, top=True)

    # Legend
    plt.legend()

    # Use LaTeX style for the font
    plt.rc("text", usetex=True)

    return fig


# def create_data_with_coefficients_from_event_study(data):
#     """This function takes the output of the event study regression as an input and then creates a dataframne with the event study coeffiucients."""

#     pattern = r"Dummy_(\w+)_(\d+Q\d+)\s(-?\d+\.\d+)\s(\d+\.\d+)"
#     matches = re.findall(pattern, data)

#     data = []
#     for match in matches:
#         country, date, coefficient, std_error = match
#         data.append({
#             "date": date,
#             "country": country,
#             "coefficient": float(coefficient),
#             "std_error": float(std_error)
#         })

#     df = pd.DataFrame(data)
#     return df


# ------------------------------------------------------------------------------------------
# Quarterly Data Implementation of Standard Approach


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


def run_exuberance_index_regression_quarterly_data(data):
    """This function runs the regression of macro fundamentals on the sentiment index
    for quarterly data."""

    # Define the regression formula
    formula = "McDonald_Sentiment_Index ~ Q('Public_Debt_as_%_of_GDP')+ GDP_in_Current_Prices_Growth + GDP_in_Current_Prices_Growth_Lead + VIX_Daily_Close_Quarterly_Mean"

    # Run the regression
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
