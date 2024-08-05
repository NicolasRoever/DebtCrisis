from debt_crisis.config import BLD, CONFIGURATION_SETTINGS, TOP_LEVEL_DIR

from debt_crisis.paper.generate_tables import (
    generate_summary_statistics_table_event_study,
    generate_descriptive_statistics_from_full_event_study_dataset,
    generate_sentiment_bond_spread_correlation_table,
    generate_event_study_regression_output_table,
    create_df_with_correlation_values_between_bond_yield_and_sentiment,
    convert_dataframe_content_to_latex_table_body,
)

from debt_crisis.gpt_sentiment_index.gpt_index_analysis import (
    calculate_gpt_sentiment_index,
)

from debt_crisis.utilities import _name_sentiment_index_output_file

import pandas as pd


# def task_create_model_run_configuration_table(
#     depends_on=BLD
#     / "data"
#     / "event_study_approach"
#     / _name_sentiment_index_output_file(
#         "event_study_full_model_data", CONFIGURATION_SETTINGS, ".pkl"
#     ),
#     produces=TOP_LEVEL_DIR
#     / "Input_for_Paper"
#     / "tables"
#     / "summary_statistics_event_study.tex",
# ):
#     data = pd.read_pickle(depends_on)

#     descriptive_statistics = (
#         generate_descriptive_statistics_from_full_event_study_dataset(data)
#     )

#     latex_table = generate_summary_statistics_table_event_study(descriptive_statistics)

#     with open(produces, "w") as f:
#         f.write(latex_table)


# def task_generate_correlation_sentiment_index_bond_yield_spread_table(
#     depends_on=BLD
#     / "data"
#     / "event_study_approach"
#     / _name_sentiment_index_output_file(
#         "event_study_full_model_data", CONFIGURATION_SETTINGS, ".pkl"
#     ),
#     produces=TOP_LEVEL_DIR
#     / "Input_for_Paper"
#     / "tables"
#     / "correlation_sentiment_index_bond_yield_spread.tex",
# ):
#     data = pd.read_pickle(depends_on)

#     correlation_data = generate_sentiment_bond_spread_correlation_table(data)

#     with open(produces, "w") as f:
#         f.write(correlation_data)

country_list = [
    "greece",
    "portugal",
    "germany",
    "france",
    "italy",
    "ireland",
    "netherlands",
    "austria",
    "hungary",
    "poland",
    "denmark",
    "sweden",
]


def task_generate_table_with_bond_yield_sentiment_correlations(
    depends_on={
        "llm_output_data_clean": BLD
        / "data"
        / "GPT_Output_Data"
        / f"sentiment_data_clean_full.pkl",
        "mcdonald_sentiment_data": BLD
        / "data"
        / "mcdonald_sentiment_index_negative_and_positive_20_.pkl",
        "bond_yield_spread": BLD
        / "data"
        / "financial_data"
        / "Quarterly Macroeconomic Variables_cleaned.pkl",
    },
    countries=country_list,
    produces=TOP_LEVEL_DIR
    / "Input_for_Paper"
    / "tables"
    / "correlation_sentiment_index_bond_yield_spread.tex",
):
    sentiment_data_full = pd.read_pickle(depends_on["llm_output_data_clean"])
    mcdonald_sentiment_data = pd.read_pickle(depends_on["mcdonald_sentiment_data"])
    bond_yield_spread = pd.read_pickle(depends_on["bond_yield_spread"])

    correlations = create_df_with_correlation_values_between_bond_yield_and_sentiment(
        bond_yield_spread=bond_yield_spread,
        llm_output_data_clean=sentiment_data_full,
        mcdonald_sentiment_data=mcdonald_sentiment_data,
        countries=countries,
    )

    latex_table = generate_sentiment_bond_spread_correlation_table(correlations)

    with open(produces, "w") as f:
        f.write(latex_table)


# def task_generate_event_study_regression_output_table(
#     depends_on=BLD
#     / "data"
#     / "event_study_approach"
#     / "event_study_regression_table_data.pkl",
#     produces=TOP_LEVEL_DIR
#     / "Input_for_Paper"
#     / "tables"
#     / "event_study_regression_output.tex",
# ):
#     data = pd.read_pickle(depends_on)

#     table = generate_event_study_regression_output_table(data)

#     with open(produces, "w") as f:
#         f.write(table)
