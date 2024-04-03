import pandas as pd
from scipy.stats import t, norm
import numpy as np
from datetime import datetime, timedelta
import statsmodels.api as sm
import statsmodels.formula.api as smf
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import re
from statsmodels.iolib.summary2 import summary_col
import re


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


def create_data_set_event_study(
    quarterly_macro_variables,
    sentiment_index,
    event_study_countries,
    event_study_time_period,
):
    """This function creates the dataset for the event study regression by merging the
    quarterly macroeconomic variables data with the sentiment index.

    Args:  quarterly_macro_variables (pd.DataFrame): The quarterly macroeconomic variables data as created by the clean_quarterly_macro_variables function.
              sentiment_index (pd.DataFrame): The sentiment index data as created by the clean_sentiment_index function.

    Returns: pd.DataFrame: The merged dataset for the event study regression.

    """

    check_if_dataframe_column_is_datetime_type(quarterly_macro_variables["Date"])

    # Filter for the right time period
    quarterly_macro_variables_filter = quarterly_macro_variables[
        quarterly_macro_variables["Date"] >= pd.to_datetime("2002-01-01")
    ]

    # Add the American Bond Yield to Calculate the Spread

    american_bond_yield_data = quarterly_macro_variables[
        quarterly_macro_variables["Country"] == "usa"
    ][["Date", "10y_Maturity_Bond_Yield"]]

    quarterly_macro_variables_filter_with_us = pd.merge(
        quarterly_macro_variables_filter,
        american_bond_yield_data,
        on="Date",
        how="left",
        validate="m:1",
        suffixes=("", "_US"),
    )

    _check_dataframe_rows(
        quarterly_macro_variables_filter_with_us,
        quarterly_macro_variables_filter.shape[0],
    )
    _check_for_missing_values_in_dataframe_column(
        quarterly_macro_variables_filter_with_us, "10y_Maturity_Bond_Yield_US"
    )

    quarterly_macro_variables_filter_with_us["Bond_Yield_Spread"] = (
        quarterly_macro_variables_filter_with_us["10y_Maturity_Bond_Yield"]
        - quarterly_macro_variables_filter_with_us["10y_Maturity_Bond_Yield_US"]
    )

    # Merge the quarterly macroeconomic variables data with the sentiment index
    merged_data = pd.merge(
        quarterly_macro_variables_filter_with_us,
        sentiment_index,
        on=["Date", "Country"],
        how="left",
        validate="one_to_one",
    )

    _check_dataframe_rows(merged_data, quarterly_macro_variables_filter.shape[0])

    _make_missing_values_heatmap(
        merged_data, "Event Study Regression Data", index="Country"
    )

    # Add GDP
    merged_data["GDP_in_Current_Prices_Growth"] = merged_data.groupby("Country")[
        "GDP_in_USD_Current_Prices"
    ].pct_change()

    merged_data["GDP_in_Current_Prices_Growth_Lead"] = merged_data.groupby("Country")[
        "GDP_in_Current_Prices_Growth"
    ].shift(-1)

    # Add the dummy variables
    merged_data_with_dummies = add_dummy_columns_for_multiple_countries(
        merged_data,
        event_study_time_period[0],
        event_study_time_period[1],
        event_study_countries,
    )
    merged_data_with_dummies = create_country_fixed_effect_dummies(
        merged_data_with_dummies
    )
    merged_data_with_dummies = create_time_fixed_effect_dummies(
        merged_data_with_dummies
    )

    return merged_data_with_dummies


def run_event_study_regression(data, event_study_countries, event_study_time_period):
    """This data runs the event study specification for the data created by the
    create_data_set_event_study function."""

    formula = (
        "Q('10y_Maturity_Bond_Yield') ~ Q('Public_Debt_as_%_of_GDP')+ GDP_in_Current_Prices_Growth + Moody_Rating_PD + "
        "VIX_Daily_Close_Quarterly_Mean + Q('10y_Maturity_Bond_Yield_US') + C(Country) + C(Date) +"
        + " + ".join(
            [
                f"Dummy_{country}_{quarter}"
                for country in event_study_countries
                for quarter in pd.period_range(
                    start=event_study_time_period[0],
                    end=event_study_time_period[1],
                    freq="Q",
                )
            ]
        )
    )

    # Drop all NA values and the US
    columns_in_the_model = [
        "10y_Maturity_Bond_Yield",
        "Public_Debt_as_%_of_GDP",
        "GDP_in_Current_Prices_Growth",
        "VIX_Daily_Close_Quarterly_Mean",
        "10y_Maturity_Bond_Yield_US",
        "Country",
        "Date",
        "Moody_Rating_PD",
    ]
    data_without_us = data.loc[data["Country"] != "usa", :].dropna(
        subset=columns_in_the_model
    )

    # Sort the data

    data_without_us = data_without_us.sort_values(by=["Country", "Date"])

    # Run the regression
    model = smf.ols(formula, data=data_without_us).fit(
        cov_type="hac-panel",
        cov_kwds={"groups": data_without_us["Country"], "maxlags": 2},
    )

    return model, data_without_us


def plot_event_study_coefficients(
    coefficient_data,
    country,
    color_scheme=["#3c5488", "#e64b35", "#4dbbd5", "#00a087", "#f39b7f"],
):
    """This function plots the coefficients from the event study regression for the
    given country."""

    # Delete all unnecessary variables
    pattern = r"^Dummy_\w+_\w+$"
    coefficient_data = coefficient_data.loc[
        coefficient_data["Variable"].str.contains(pattern, regex=True), :
    ]

    coefficient_data["Date"] = pd.to_datetime(
        coefficient_data["Variable"].str.split("_").str[-1]
    )
    coefficient_data["Country"] = coefficient_data["Variable"].str.split("_").str[-2]
    coefficient_data["CI_95_lower"] = (
        coefficient_data["Coefficient"] - coefficient_data["Standard Errors"] * 1.96
    )
    coefficient_data["CI_95_upper"] = (
        coefficient_data["Coefficient"] + coefficient_data["Standard Errors"] * 1.96
    )

    # Filter the data for the given country
    country_data = coefficient_data[coefficient_data["Country"] == country]
    country_data = country_data.sort_values("Date")

    # Set the style of the plot
    sns.set_style("white")

    # Create the plot
    fig = plt.figure(figsize=(8, 5))

    plt.plot(
        country_data["Date"],
        country_data["Coefficient"],
        marker="o",
        color=color_scheme[0],
    )

    # Add a horizontal line at y=0
    plt.axhline(0, color="grey", linestyle=":")

    # Plot the confidence interval
    plt.fill_between(
        country_data["Date"],
        country_data["CI_95_lower"],
        country_data["CI_95_upper"],
        color="b",
        alpha=0.1,
    )

    # Set the title and labels
    plt.title(
        f"Coefficients for {country.capitalize()} Over Time with Confidence Interval",
        fontsize=16,
    )
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Coefficient", fontsize=14)

    # Keep only the y-axis and x-axis
    sns.despine(left=False, bottom=False, right=True, top=True)

    # Use LaTeX style for the font
    plt.rc("text", usetex=True)

    return fig


def plot_event_study_coefficients_for_multiple_countries_in_one_plot(
    raw_regression_output_data,
    countries,
    color_scheme=["#3c5488", "#e64b35", "#4dbbd5", "#00a087", "#f39b7f", "#000000"],
):
    """This function plots the coefficients from the event study regression for the
    given countries."""

    coefficient_data = (
        extract_event_study_coefficients_from_event_study_regression_data(
            raw_regression_output_data
        )
    )

    # Set the style of the plot
    sns.set_style("white")

    # Create the plot
    fig = plt.figure(figsize=(8, 6))

    # Loop over the list of countries
    for i, country in enumerate(countries):
        # Filter the data for the given country
        country_data = coefficient_data[coefficient_data["Country"] == country]
        country_data = country_data.sort_values("Date")

        # Plot the data for the country
        plt.plot(
            country_data["Date"],
            country_data["Coefficient"],
            marker="o",
            color=color_scheme[i % len(color_scheme)],
            label=country,
        )

        # Plot the confidence interval
        plt.fill_between(
            country_data["Date"],
            country_data["CI_95_lower"],
            country_data["CI_95_upper"],
            color=color_scheme[i % len(color_scheme)],
            alpha=0.1,
        )

    # Add a horizontal line at y=0
    plt.axhline(0, color="grey", linestyle=":")

    # Set the title and labels
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Event Study Coefficient", fontsize=14)

    # Keep only the y-axis and x-axis
    sns.despine(left=False, bottom=False, right=True, top=True)

    # Legend
    plt.legend()

    # Use LaTeX style for the font
    plt.rc("text", usetex=True)

    return fig


def create_country_fixed_effect_dummies(data):
    """This function adds dummy columns for the countries in the dataset."""
    countries = data["Country"].unique()
    for country in countries:
        data[f"Dummy_{country}"] = np.where(data["Country"] == country, 1, 0)
    return data


def create_time_fixed_effect_dummies(data):
    """This function adds dummy columns for the time periods in the dataset."""
    start_date = data["Date"].min()
    end_date = data["Date"].max()

    quarters = pd.period_range(start=start_date, end=end_date, freq="Q")

    for quarter in quarters:
        start_date = quarter.start_time
        end_date = quarter.end_time
        data[f"Dummy_{quarter}"] = np.where(
            (data["Date"] >= start_date) & (data["Date"] <= end_date), 1, 0
        )
    return data


def add_dummy_columns_for_quarter_year_combination(
    data, start_quarter, end_quarter, country
):
    """This function adds dummy columns to use in an event study design.

    Args: data (pd.DataFrame): The dataset for the event study regression.
                start_quarter (str): The start quarter for the dummy columns.
                end_quarter (str): The end quarter for the dummy columns.
                country (str): The country for which the dummy columns are added.
    Returns: pd.DataFrame: The dataset with the dummy columns added.

    """

    quarters = pd.period_range(start=start_quarter, end=end_quarter, freq="Q")
    for quarter in quarters:
        start_date = quarter.start_time
        end_date = quarter.end_time

        data[f"Dummy_{country}_{quarter}"] = np.where(
            (data["Date"] >= start_date)
            & (data["Date"] <= end_date)
            & (data["Country"] == country),
            1,
            0,
        )
    return data


def add_dummy_columns_for_multiple_countries(
    data, start_quarter, end_quarter, countries
):
    """This function adds dummy columns for multiple countries and quarter-year
    combinations."""
    for country in countries:
        data = add_dummy_columns_for_quarter_year_combination(
            data, start_quarter, end_quarter, country
        )
    return data


def extract_event_study_coefficients_from_event_study_regression_data(regression_data):
    """This functions takes the output from the event study regression function (named
    event_study_coefficients_data.pkl) and extracts the event study coefficients, the
    date, the country and the confidence intervals.

    Args: regression_data (pd.DataFrame): The output from the event study regression function.

    Returns: pd.DataFrame
        columns: Date (pd.Datetime) The date of the event study coefficient in year-quarter format.
                 Country (str) The country for which the event study coefficient is calculated.
                Coefficient (float) The event study coefficient.
                CI_95_lower (float) The lower bound of the 95% confidence interval.
                CI_95_upper (float) The upper bound of the 95% confidence interval.

    """

    # Delete all variables not event study coefficients
    pattern = r"^Dummy_\w+_\w+$"
    coefficient_data = regression_data.loc[
        regression_data["Variable"].str.contains(pattern, regex=True), :
    ]

    # Extract the necessary infomration
    coefficient_data["Date"] = pd.to_datetime(
        coefficient_data["Variable"].str.split("_").str[-1]
    )
    coefficient_data["Country"] = coefficient_data["Variable"].str.split("_").str[-2]
    coefficient_data["CI_95_lower"] = (
        coefficient_data["Coefficient"] - coefficient_data["Standard Errors"] * 1.96
    )
    coefficient_data["CI_95_upper"] = (
        coefficient_data["Coefficient"] + coefficient_data["Standard Errors"] * 1.96
    )

    return coefficient_data


def extract_column_names_from_regression_formula(formula):
    """This function takes in a string giving a regression formula and returns a list
    with the variable names."""

    # Split the formula into left and right parts
    left, right = formula.split("~")

    # Extract the terms from the right part of the formula
    terms = re.split(r"\+", right.strip())

    # Remove the 'Q('')' and 'C('')' wrappers and strip whitespace
    column_names = [
        re.sub(r"Q\('([^']*)'\)|C\(([^)]*)\)", r"\1\2", term).strip() for term in terms
    ]

    # Remove duplicates and sort the column names
    column_names = sorted(set(name for name in column_names if name))

    return column_names


def run_event_study_for_given_configuration(
    event_study_data,
    configuraton,
    event_study_countries,
    event_study_time_period,
    standard_errors="hac-panel",
):
    """This function runs a regression with the dataset as an input and the given
    configuration.

    THe function returns the statsmodel.model object

    """

    # Make the event study configuration:

    formula = configuraton + " + ".join(
        f"Dummy_{country}_{quarter}"
        for country in event_study_countries
        for quarter in pd.period_range(
            start=event_study_time_period[0],
            end=event_study_time_period[1],
            freq="Q",
        )
    )

    # Drop all rows where a variable in the formula is NA and the US
    columns_for_dropping = extract_column_names_from_regression_formula(configuraton)
    columns_for_dropping.append("10y_Maturity_Bond_Yield")
    data = event_study_data.dropna(subset=columns_for_dropping)
    data = data.loc[data["Country"] != "usa", :]

    # Sort the data
    # (This is required for the HAC standard errors to work correctly)

    data = data.sort_values(by=["Country", "Date"])

    # Run the regression
    model = smf.ols(formula=formula, data=data).fit(
        cov_type=standard_errors, cov_kwds={"groups": data["Country"], "maxlags": 1}
    )

    return model


def extract_parameters_for_regression_table_from_model(model, configuration):
    # Define the significance levels
    significance_levels = [0.01, 0.05, 0.1]

    # Define the stars for each significance level
    stars = ["***", "**", "*"]

    # Define the parameters to extract
    parameters = [
        "Q('Public_Debt_as_%_of_GDP')",
        "GDP_in_Current_Prices_Growth",
        "Moody_Rating_PD",
        "VIX_Daily_Close_Quarterly_Mean",
        "Q('10y_Maturity_Bond_Yield_US')",
        "Q('3_Month_US_Treasury_Yield_Quarterly_Mean')",
        "Q('NASDAQ_Daily_Close_Quarterly_Mean')",
        "Q('Current_Account_in_USD')",
        "Q('Eurostat_CPI_Annualised Growth_Rate')",
    ]

    # Initialize an empty dictionary to store the coefficients with stars
    coefficients_with_stars = {}

    # Loop over each parameter
    for param in parameters:
        # Get the coefficient value
        coefficient = model.params.get(param, "")

        # Get the p-value of the coefficient
        p_value = model.pvalues.get(param, 1)

        # Add stars to the coefficient based on its p-value
        for level, star in zip(significance_levels, stars):
            if p_value < level:
                coefficient = f"{coefficient:.2f}{star}"
                break
            else:
                coefficient = f"{coefficient:.2f}"

        # Add the coefficient with stars to the dictionary
        coefficients_with_stars[param] = coefficient

    # Check for fixed effects
    country_fe = "Yes" if "C(Country)" in configuration else "No"
    time_fe = "Yes" if "C(Date)" in configuration else "No"

    # Add other model statistics to the dictionary
    coefficients_with_stars.update(
        {
            "Country Fixed Effects": country_fe,
            "Time Fixed Effects": time_fe,
            "Number of Observations": round(model.nobs, 0),
            "R-Squared": round(model.rsquared, 2),
        }
    )

    # Convert the dictionary to a pandas Series and return it
    return pd.Series(coefficients_with_stars)


def get_parameters_for_regression_table_for_configuration(
    event_study_data,
    configuraton,
    event_study_countries,
    event_study_time_period,
    standard_errors="hac-panel",
):
    model = run_event_study_for_given_configuration(
        event_study_data,
        configuraton,
        event_study_countries,
        event_study_time_period,
        standard_errors,
    )

    parameters = extract_parameters_for_regression_table_from_model(model, configuraton)

    return parameters


def generate_regresssion_table_for_list_of_configurations(
    event_study_data, EVENT_STUDY_MODELS, EVENT_STUDY_COUNTRIES, EVENT_STUDY_TIME_PERIOD
):
    # Initialize empty dataframe to store results
    results = pd.DataFrame()

    # Loop over each configuration
    for index, configuration in enumerate(EVENT_STUDY_MODELS):
        parameters = get_parameters_for_regression_table_for_configuration(
            event_study_data,
            configuration,
            EVENT_STUDY_COUNTRIES,
            EVENT_STUDY_TIME_PERIOD,
        )
        results[str(index)] = parameters

    return results
