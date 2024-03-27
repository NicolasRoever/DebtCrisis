import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import re
import matplotlib.pyplot as plt
import seaborn as sns


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

    return model


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


def run_exuberance_index_regression_event_study_data(data):
    """This function runs the regression of macro fundamentals on the sentiment
    index."""

    # Define the regression formula
    formula = "McDonald_Sentiment_Index ~ Q('Public_Debt_as_%_of_GDP')+ GDP_in_Current_Prices_Growth + Current_Account_in_USD + VIX_Daily_Close_Quarterly_Mean + C(Country) + Moody_Rating_PD"

    # Run the regression
    model = smf.ols(formula, data=data).fit()

    return model


def run_exuberance_index_regression_quarterly_data(data):
    """This function runs the regression of macro fundamentals on the sentiment index
    for quarterly data."""

    # Define the regression formula
    formula = "McDonald_Sentiment_Index ~ Q('Public_Debt_as_%_of_GDP')+ GDP_in_Current_Prices_Growth + GDP_in_Current_Prices_Growth_Lead + Current_Account_in_USD + VIX_Daily_Close_Quarterly_Mean + C(Country)"

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
