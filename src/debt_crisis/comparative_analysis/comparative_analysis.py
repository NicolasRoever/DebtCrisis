import pandas as pd

from debt_crisis.gpt_sentiment_index.gpt_index_analysis import (
    calculate_gpt_sentiment_index,
)


def create_dataframe_with_all_results_for_analysis(
    gpt_data: pd.DataFrame,
    mcdonald_data: pd.DataFrame,
    economic_factors: pd.DataFrame,
    quarterly_macro_data: pd.DataFrame,
) -> pd.DataFrame:
    """Create a dataframe with all the results for analysis.

    Args:
    gpt_data: The GPT data.
    mcdonald_data: The McDonald data.
    economic_factors: The economic factors data.
    quarterly_macro_data: The quarterly macro data.

    Returns:
    A dataframe with all the results for analysis.

    """

    # Create full GPT data

    all_countries = gpt_data["Country"].unique()
    full_gpt_index = pd.DataFrame()

    for country in all_countries:
        gpt_data_country = gpt_data[gpt_data["Country"] == country]
        gpt_index_country = calculate_gpt_sentiment_index(gpt_data_country, country)
        gpt_index_country["Country"] = country  # Add the country name as a column
        full_gpt_index = pd.concat(
            [full_gpt_index, gpt_index_country], ignore_index=True
        )

    # Merge Everything

    both_indices = pd.merge(
        full_gpt_index,
        mcdonald_data,
        on=["Date", "Country"],
        how="left",
        validate="one_to_one",
    )

    indices_and_economic_factors = pd.merge(
        both_indices,
        economic_factors,
        on=["Date", "Country"],
        how="left",
        validate="one_to_one",
    )

    indices_and_economic_factors_and_macro = pd.merge(
        indices_and_economic_factors,
        quarterly_macro_data,
        on=["Date", "Country"],
        how="left",
        validate="one_to_one",
    )

    # Add US bond Yields
    us_bond_yields = quarterly_macro_data[quarterly_macro_data["Country"] == "usa"][
        ["Date", "10y_Maturity_Bond_Yield"]
    ]

    indices_and_economic_factors_and_macro = pd.merge(
        indices_and_economic_factors_and_macro,
        us_bond_yields,
        on="Date",
        how="left",
        validate="many_to_one",
        suffixes=("", "_US"),
    )

    indices_and_economic_factors_and_macro["Bond_Yield_Spread"] = (
        indices_and_economic_factors_and_macro["10y_Maturity_Bond_Yield"]
        - indices_and_economic_factors_and_macro["10y_Maturity_Bond_Yield_US"]
    )

    return indices_and_economic_factors_and_macro


def create_correlation_table(full_data: pd.DataFrame) -> pd.DataFrame:
    """Create a correlation table.

    Args:
    full_data: The full data.

    Returns:
    A dataframe with the country, the correlation between the 10y maturity bond yield and the GPT sentiment index, and the correlation between the 10y maturity bond yield and the McDonald sentiment index and the correlation between the bond yield spread and both factors.

    """

    result_df = full_data.groupby("Country").apply(calculate_correlations).reset_index()

    return result_df


def calculate_correlations(group):
    """THis functionsd calculates a predifined set of correlations for a given group."""
    return pd.Series(
        {
            "Cor10_year_yield_gpt": group["10y_Maturity_Bond_Yield"].corr(
                group["Sentiment_GPT"]
            ),
            "Cor10_year_yield_mcdonald": group["10y_Maturity_Bond_Yield"].corr(
                group["McDonald_Sentiment_Index"]
            ),
            "Cor_10_year_spread_gpt": group["Bond_Yield_Spread"].corr(
                group["Sentiment_GPT"]
            ),
            "Cor10_year_spread_mcdonald": group["Bond_Yield_Spread"].corr(
                group["McDonald_Sentiment_Index"]
            ),
        }
    )
