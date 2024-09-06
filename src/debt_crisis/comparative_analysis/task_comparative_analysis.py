from debt_crisis.config import SRC, BLD

from debt_crisis.comparative_analysis.comparative_analysis import (
    create_dataframe_with_all_results_for_analysis,
    create_correlation_table,
)

import pandas as pd


def task_create_dataframe_with_all_results(
    depends_on={
        "llm_output": BLD
        / "data"
        / "GPT_Output_Data"
        / f"sentiment_data_clean_full.pkl",
        "loughran_mcdonald_output": BLD
        / "data"
        / "mcdonald_sentiment_index_cleaned_negative_and_positive_20_.pkl",
        "economic_factors": BLD / "factor_models" / "factor_model_dataset.pkl",
        "quarterly_macro_data": BLD
        / "data"
        / "financial_data"
        / "Quarterly Macroeconomic Variables_cleaned.pkl",
    },
    produces=BLD / "data" / "sentiment_data_full_with_economic_factors.pkl",
):
    gpt_data = pd.read_pickle(depends_on["llm_output"])
    mcdonald_data = pd.read_pickle(depends_on["loughran_mcdonald_output"])
    economic_factors = pd.read_pickle(depends_on["economic_factors"])
    quarterly_macro_data = pd.read_pickle(depends_on["quarterly_macro_data"])

    full_data = create_dataframe_with_all_results_for_analysis(
        gpt_data=gpt_data,
        mcdonald_data=mcdonald_data,
        economic_factors=economic_factors,
        quarterly_macro_data=quarterly_macro_data,
    )

    full_data.to_pickle(produces)


def task_make_correlational_analysis(
    depends_on=BLD / "data" / "sentiment_data_full_with_economic_factors.pkl",
    produces=BLD / "results" / "correlation_analysis.xlsx",
):
    full_data = pd.read_pickle(depends_on)

    correlations = create_correlation_table(full_data)

    correlations.to_excel(produces)
