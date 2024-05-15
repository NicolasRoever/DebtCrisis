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
    _create_dictionary_of_coefficients_with_stars,
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


def run_al_amine_regression(data, moody_rating_mapping):
    """This function takes in the event study data set and runs a regression
    specification analogous to the one in Al-Amine and Willems (2022)."""

    # Get Public Debt Change as Percent
    data["Public_Debt_as_%_of_GDP_Change"] = data.groupby("Country")[
        "Public_Debt_as_%_of_GDP"
    ].pct_change()

    # Get Log Bond Spread
    data["Log_Bond_Spread"] = np.log(data["10y_Maturity_Bond_Yield"]) - np.log(
        data["10y_Maturity_Bond_Yield_US"]
    )

    # Make mapping
    data["Moody_Rating_Numerical"] = data["Rating_Moody_Last_Quarter_Day"].map(
        moody_rating_mapping
    )

    model_specification = "Log_Bond_Spread ~ Moody_Rating_Numerical + Q('Public_Debt_as_%_of_GDP') + Q('Public_Debt_as_%_of_GDP_Change') + Q('Eurostat_CPI_Annualised Growth_Rate') + Current_Account_in_USD + GDP_in_Current_Prices_Growth + VIX_Daily_Close_Quarterly_Mean + NASDAQ_Daily_Close_Quarterly_Mean"

    # Drop all NA's
    regression_data = data.dropna(
        subset=[
            "Log_Bond_Spread",
            "Moody_Rating_Numerical",
            "Public_Debt_as_%_of_GDP",
            "Public_Debt_as_%_of_GDP_Change",
            "Eurostat_CPI_Annualised Growth_Rate",
            "Current_Account_in_USD",
            "GDP_in_Current_Prices_Growth",
            "VIX_Daily_Close_Quarterly_Mean",
            "NASDAQ_Daily_Close_Quarterly_Mean",
        ]
    )

    model = smf.ols(model_specification, data=regression_data).fit(
        cov_type="cluster", cov_kwds={"groups": regression_data["Country"]}
    )

    return model, regression_data


def create_comparison_table_our_results_vs_al_amine(fitted_model, data):
    """THis function creates a Latex table comparing the results from our model to the
    results from Al-Amine and Willems (2022)."""

    (
        dictionary_with_coefficients,
        t_values,
    ) = create_dictionary_with_coefficients_with_stars_and_t_values(fitted_model)

    r_squared = fitted_model.rsquared

    number_countries = len(data["Country"].unique())
    observations = len(data)

    table = f"""
    \\begin{{center}}
    \\begin{{table}}[h!] \\caption{{Descriptive Statistics of Data Used in Event Study}}
    \\label{{table:descriptives_event_study}}
    \\begin{{tabular}}{{p{{4cm}}p{{4cm}}p{{4cm}}p{{4cm}}}}
    \\toprule
     Variable in Our Estimation & Value & Value in Al-Amine and Willems (2022) & Value  \\\\
     Moody Rating & {dictionary_with_coefficients['Moody_Rating_Numerical']} & Average credit rating  & −0.245***\\\\
      & ({t_values["Moody_Rating_Numerical"]:.2f}) & & (−29.99) \\\\
        Public Debt as \\% of GDP & {dictionary_with_coefficients["Q('Public_Debt_as_%_of_GDP')"]} & Public Debt as \\% of GDP & −0.002** *\\\\
         & ({t_values["Q('Public_Debt_as_%_of_GDP')"]:.2f}) & & (−2.17) \\\\
                Public Debt as \\% of GDP Change & {dictionary_with_coefficients["Q('Public_Debt_as_%_of_GDP_Change')"]} & Public Debt as \\% of GDP Change & −0.002**\\\\
        & ({t_values["Q('Public_Debt_as_%_of_GDP_Change')"]:.2f}) & & (−2.63) \\\\
        Inflation (\\%) & {dictionary_with_coefficients["Q('Eurostat_CPI_Annualised Growth_Rate')"]} & Inflation (\\%) & -0.003**\\\\
         & ({t_values["Q('Eurostat_CPI_Annualised Growth_Rate')"]:.2f}) & & (-2.63) \\\\
        Current Account Balance & {dictionary_with_coefficients['Current_Account_in_USD']} & Current Account Balance & 0.014**\\\\
         & ({t_values["Current_Account_in_USD"]:.2f}) & & (2.54) \\\\
        Real GDP Growth & {dictionary_with_coefficients['GDP_in_Current_Prices_Growth']} &  Real GDP Growth & 0.032***\\\\
         & ({t_values["GDP_in_Current_Prices_Growth"]:.2f}) & & (3.45) \\\\
        VIX & {dictionary_with_coefficients['VIX_Daily_Close_Quarterly_Mean']} & VIX & 0.045***\\\\
        & ({t_values["VIX_Daily_Close_Quarterly_Mean"]:.2f}) & & (10.08) \\\\
        NASDAQ Returns & {dictionary_with_coefficients['NASDAQ_Daily_Close_Quarterly_Mean']} & S\\&P 500 Returns  & 2.018***\\\\
         & ({t_values["NASDAQ_Daily_Close_Quarterly_Mean"]:.2f}) & & (5.46) \\\\
        Constant & {dictionary_with_coefficients['Intercept']} & Constant & 7.476***\\\\
         & ({t_values["Intercept"]:.2f}) & & (52.10) \\\\
        R-Squared & {r_squared:.2f} & R-Squared & 0.753 & \\\\
        Countries & {number_countries} & Countries & 87 & \\\\
        Observations & {observations} & Observations & 4364 & \\\\
    \\midrule
    \\begin{{minipage}}{{15cm}}
    \\footnotesize{{\\textbf{{Notes:}} The dependent variable in this regression is the natural log of the 10 year government bond yield of the country minus the 10 year government bond yield of the US. The table compares the results from an estimtation with our dataset to the results from Al-Amine and Willems (2022). The t-values are in parentheses. The stars indicate the significance level of the coefficient: *** p<0.01, ** p<0.05, * p<0.1.}}
    \\end{{minipage}}
       \\end{{tabular}}
    \\end{{table}}
    \\end{{center}}
    """

    return table


def create_dictionary_with_coefficients_with_stars_and_t_values(fitted_model):
    # Define the parameters to extract
    parameters = [
        "Moody_Rating_Numerical",
        "Q('Public_Debt_as_%_of_GDP')",
        "Q('Public_Debt_as_%_of_GDP_Change')",
        "Q('Eurostat_CPI_Annualised Growth_Rate')",
        "Current_Account_in_USD",
        "GDP_in_Current_Prices_Growth",
        "VIX_Daily_Close_Quarterly_Mean",
        "NASDAQ_Daily_Close_Quarterly_Mean",
        "Intercept",
    ]

    coefficients_with_stars = _create_dictionary_of_coefficients_with_stars(
        parameters, fitted_model
    )

    t_values = fitted_model.tvalues

    # Convert the dictionary to a pandas Series and return it
    return pd.Series(coefficients_with_stars), t_values


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
