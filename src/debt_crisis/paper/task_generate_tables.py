from debt_crisis.config import BLD, CONFIGURATION_SETTINGS, TOP_LEVEL_DIR

from debt_crisis.paper.generate_tables import (
    generate_summary_statistics_table_event_study,
    generate_descriptive_statistics_from_full_event_study_dataset,
    generate_sentiment_bond_spread_correlation_table,
)

from debt_crisis.utilities import _name_sentiment_index_output_file

import pandas as pd


def task_create_model_run_configuration_table(
    depends_on=BLD
    / "data"
    / "event_study_approach"
    / _name_sentiment_index_output_file(
        "event_study_full_model_data", CONFIGURATION_SETTINGS, ".pkl"
    ),
    produces=TOP_LEVEL_DIR
    / "Input_for_Paper"
    / "tables"
    / "summary_statistics_event_study.tex",
):
    data = pd.read_pickle(depends_on)

    descriptive_statistics = (
        generate_descriptive_statistics_from_full_event_study_dataset(data)
    )

    latex_table = generate_summary_statistics_table_event_study(descriptive_statistics)

    with open(produces, "w") as f:
        f.write(latex_table)


def task_generate_correlation_sentiment_index_bond_yield_spread_table(
    depends_on=BLD
    / "data"
    / "event_study_approach"
    / _name_sentiment_index_output_file(
        "event_study_full_model_data", CONFIGURATION_SETTINGS, ".pkl"
    ),
    produces=TOP_LEVEL_DIR
    / "Input_for_Paper"
    / "tables"
    / "correlation_sentiment_index_bond_yield_spread.tex",
):
    data = pd.read_pickle(depends_on)

    correlation_data = generate_sentiment_bond_spread_correlation_table(data)

    with open(produces, "w") as f:
        f.write(correlation_data)
