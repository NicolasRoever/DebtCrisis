import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

from debt_crisis.gpt_sentiment_index.gpt_index_analysis import (
    calculate_gpt_sentiment_index,
)


def calculate_p_value_for_from_univariate_regression(
    df, formula="Bond_Yield_Spread ~ McDonald_Sentiment_Index"
):
    try:
        model = smf.ols(formula=formula, data=df).fit()
        # Exclude the intercept term to get the p-value of the x variable
        pvalues = model.pvalues.drop("Intercept", errors="ignore")
        return pvalues.iloc[0]  # Return the p-value of the first independent variable
    except ValueError:
        return np.nan


def generate_summary_statistics_table_event_study(data):
    table_body = convert_dataframe_content_to_latex_table_body(data)

    number_of_observations = data["Number_of_Observations"].sum()

    table = f"""
    \\begin{{center}}
    \\begin{{table}}[H] \\caption{{Descriptive Statistics of Data Used in Event Study}}
    \\label{{table:descriptives_event_study}}
    \\scalebox{{0.7}}{{
    \\begin{{tabular}}{{p{{3cm}}p{{0.2cm}}p{{3cm}}p{{3cm}}p{{1.5cm}}p{{2cm}}p{{1.5cm}}}}
    \\toprule
    & \multicolumn{{4}}{{c}}{{Average}} & & \\\\
    & \\cline{{2-4}} & & \\\\
   Country & & Public Debt as \\% of GDP & 10 Year Bond Yield & Real GDP Growth &   Mode Moody Rating & No. Observations \\\\
   \\midrule
    {table_body}
    \\midrule
    \\begin{{minipage}}{{15cm}}
    \\footnotesize{{\\textbf{{Notes:}} The table presents summary statistics for the dataset we compiled for our event study.  We have a total of {number_of_observations} observations as there are some missing data (i.e. in the bond yields we obtained from the OECD).}}
    \\end{{minipage}}
       \\end{{tabular}}
    }}
    \\end{{table}}
    \\end{{center}}
    """

    return table


def generate_descriptive_statistics_from_full_event_study_dataset(data):
    # Create Dexcriptive Statistics

    # Group the data by 'Country' and calculate the mean of the specified columns
    average_data = data.groupby("Country")[
        [
            "Public_Debt_as_%_of_GDP",
            "10y_Maturity_Bond_Yield",
            "GDP_in_Current_Prices_Growth",
        ]
    ].mean()

    # Calculate the most frequent 'Rating_Moody_Last_Quarter_Day' for each country
    average_data["Most Frequent Rating_Moody_Last_Quarter_Day"] = data.groupby(
        "Country"
    )["Rating_Moody_Last_Quarter_Day"].agg(pd.Series.mode)

    # Calculate the number of observations for each country
    average_data["Number_of_Observations"] = data.groupby("Country").size()

    average_data = average_data.round(2)

    # Reset the index
    average_data = average_data.reset_index()

    average_data["Country"] = average_data["Country"].str.title()

    # Insert empty columns for breaks
    average_data.insert(1, "Break1", "")

    return average_data


def generate_sentiment_bond_spread_correlation_table(correlation_data):
    table_body = convert_dataframe_content_to_latex_table_body(correlation_data)

    table = f"""

     \\begin{{center}}
    \\begin{{table}}[H] \\caption{{Results for Raw Sentiment Index}}
    \\label{{table:correlation_sentiment_index}}
    \\scalebox{{1}}{{
    \\begin{{tabular}}{{p{{3cm}}p{{5 cm}}p{{5 cm}}}}
    \\toprule
    \\textbf{{Country}} & \\textbf{{Correlation GPT 4o-mini Index}} & \\textbf{{Correlation Loughran and McDonald Index}} \\\\
    \\midrule
    {table_body}
    \\bottomrule
    \\begin{{minipage}}{{15cm}}
    \\footnotesize{{\\textbf{{Notes:}} The table shows correlations between the bond yield spread of the country (measured as the difference between the 10 year government bond yield of that country with the United States' bond yield) and our raw sentiment index.  The stars indicate the significance level of the correlation coefficient obtained by running a t-test on the regresssion coefficient on a univariate regression of the bond spread on the sentiment index: *** p<0.01, ** p<0.05, * p<0.1.}})
    \\end{{minipage}}
    \\end{{tabular}}
    }}
    \\end{{table}}
    \\end{{center}}
    """

    return table


def generate_sentiment_bond_spread_correlation_from_event_study_data(event_study_data):
    # Group the data by 'Country' and calculate the correlation of 'Bond_Yield_Spread' and 'McDonald_Sentiment_Index'
    correlations = (
        event_study_data.groupby("Country")
        .apply(
            lambda x: x[["Bond_Yield_Spread", "McDonald_Sentiment_Index"]]
            .corr()
            .iloc[0, 1]
        )
        .reset_index()
        .rename(columns={0: "Correlation"})
    )

    correlations["Correlation"] = correlations["Correlation"].round(2)

    p_values = event_study_data.groupby("Country").apply(
        calculate_p_value_for_from_univariate_regression
    )

    # Add stars to the correlation coefficients based on their p-values
    correlations["Correlation"] = np.where(
        p_values < 0.01,
        correlations["Correlation"].astype(str) + "***",
        correlations["Correlation"],
    )
    correlations["Correlation"] = np.where(
        (p_values >= 0.01) & (p_values < 0.05),
        correlations["Correlation"].astype(str) + "**",
        correlations["Correlation"],
    )
    correlations["Correlation"] = np.where(
        (p_values >= 0.05) & (p_values < 0.1),
        correlations["Correlation"].astype(str) + "*",
        correlations["Correlation"],
    )

    # Capitalise the country names and sort by 'Country'
    correlations["Country"] = correlations["Country"].str.title()
    correlations = correlations.dropna().sort_values("Country").dropna()

    return correlations


def generate_event_study_regression_output_table(event_study_output_data):
    row_names = [
        "Public Debt as Percent of GDP",
        "Real GDP Growth",
        "Moody Sovereign Rating",
        "CBOE VIX Index",
        "Bond Yield US Bond 10 Year Maturity",
        "Three Month US Treasury Yield",
        "NASDAQ Index",
        "Current Account",
        "Consumer Price Index",
        "Country Fixed Effects",
        "Time Fixed Effects",
        "Number of Observations",
        "R-Squared",
    ]

    event_study_output_data.insert(0, "Variable", row_names)

    table_body = convert_dataframe_content_to_latex_table_body(event_study_output_data)

    table = f"""

    {{
    \\begin{{tabular}}{{l*{{7}}{{c}}}}
    \\hline\\hline
    & & (1) & (2) & (3) & (4) & (5) & (6) & (7) \\\\
    \\cmidrule(l{{0.5em}} r{{0.5em}}) {{2-3}}

    {table_body}

    \\hline\\hline
    \\multicolumn{{5}}{{l}}{{\\footnotesize * \\(p<0.05\\),**\\(p<0.01\\), *** \\(p<0.001\\)}}\\\\
    \\end{{tabular}}
    }}
    """

    return table


def convert_dataframe_content_to_latex_table_body(data):
    # Convert each row to a string with ' & ' as the separator
    data_string = data.apply(lambda row: " & ".join(row.astype(str)), axis=1)

    # Join all rows into a single string with ' \\\\\n' as the separator
    data_string = " \\\\".join(data_string)

    # Add ' \\\\' at the end of the string
    data_string += " \\\\"

    return data_string


def create_df_with_correlation_values_between_bond_yield_and_sentiment(
    bond_yield_spread: pd.DataFrame,
    llm_output_data_clean: pd.DataFrame,
    mcdonald_sentiment_data: pd.DataFrame,
    countries: list[str],
):
    """THis function takes in the values for the bond yield spread, the GPT Index and
    the McDonald Index and calculates the correlation between the bond yield spread and
    the sentiment index for each country in the list of countries under study.

    It then generates a table with the correlation values for each country.

    """
    result_df = pd.DataFrame()

    countries = sorted(countries)
    for country in countries:
        # Import Data

        bond_yield_spread_filter = bond_yield_spread[
            bond_yield_spread["Country"] == country
        ]
        bond_yield_spread_filter["Date"] = pd.to_datetime(
            bond_yield_spread_filter["Date"]
        )
        quarterly_dates = bond_yield_spread_filter["Date"]

        llm_data_country = llm_output_data_clean[
            llm_output_data_clean["Country"] == country
        ]

        llm_sentiment_index_data = calculate_gpt_sentiment_index(
            preprocessed_data=llm_data_country, country_under_study=country
        )
        llm_quarter_data = llm_sentiment_index_data[
            llm_sentiment_index_data["Date"].isin(quarterly_dates)
        ]

        mcdonald_quarter_data = mcdonald_sentiment_data[
            mcdonald_sentiment_data["Date"].isin(quarterly_dates)
        ]

        # Merge Data
        merged_data = pd.merge(
            llm_quarter_data,
            mcdonald_quarter_data,
            on="Date",
            suffixes=("_GPT", "_McDonald"),
            validate="one_to_one",
        )
        merged_data = pd.merge(
            merged_data, bond_yield_spread_filter, on="Date", validate="one_to_one"
        )

        # Calculate Correlations
        correlation_gpt = merged_data[f"Sentiment_GPT_{country}"].corr(
            merged_data["10y_Maturity_Bond_Yield"]
        )
        correlation_mcdonald = merged_data[f"Sentiment_Index_McDonald_{country}"].corr(
            merged_data["10y_Maturity_Bond_Yield"]
        )

        # Get p_values
        p_value_gpt = calculate_p_value_for_from_univariate_regression(
            merged_data, f"Q('10y_Maturity_Bond_Yield') ~ Sentiment_GPT_{country}"
        )
        p_value_mcdonald = calculate_p_value_for_from_univariate_regression(
            merged_data,
            f"Q('10y_Maturity_Bond_Yield') ~ Sentiment_Index_McDonald_{country}",
        )

        # Append Results with Stars
        result_df = pd.concat(
            [
                result_df,
                pd.DataFrame(
                    [
                        {
                            "Country": country.capitalize(),
                            "Correlation GPT Index": f"{correlation_gpt:.2f}{add_stars(p_value_gpt)}",
                            "Correlation McDonald Index": f"{correlation_mcdonald:.2f}{add_stars(p_value_mcdonald)}",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    return result_df


def add_stars(p_value):
    if p_value < 0.01:
        return "***"
    elif p_value < 0.05:
        return "**"
    elif p_value < 0.1:
        return "*"
    else:
        return ""


def generate_sentiment_bond_spread_correlation_table(correlation_data):
    table_body = convert_dataframe_content_to_latex_table_body(correlation_data)

    table = f"""

     \\begin{{center}}
    \\begin{{table}}[H] \\caption{{Results for Sentiment Indices}}
    \\label{{table:correlation_sentiment_index}}
    \\scalebox{{1}}{{
    \\begin{{tabular}}{{p{{3cm}}p{{5cm}}p{{5cm}}}}
    \\toprule
    \\textbf{{Country}} & \\textbf{{Index Loughran and McDonald}} & \\textbf{{Index GPT 4o}} \\\\
    \\midrule
    {table_body}
    \\bottomrule
    \\begin{{minipage}}{{15cm}}
    \\footnotesize{{\\textbf{{Notes:}} The table shows corrleations between the bond yield of the country and the two sentiment indices.  The stars indicate the significance level of the correlation coefficient obtained by running a t-test on the regresssion coefficient on a univariate regression of the bond spread on the sentiment index: *** p<0.01, ** p<0.05, * p<0.1.}})
    \\end{{minipage}}
    \\end{{tabular}}
    }}
    \\end{{table}}
    \\end{{center}}
    """

    return table
