from debt_crisis.sentiment_index.validate_sentiment_index import (
    clean_sentiment_count_data,
    plot_actual_word_frequency,
    plot_sentiment_index_and_bond_yield_spread_for_country,
    plot_sentiment_index_and_exuberance_index_for_country,
)

from debt_crisis.config import BLD, TOP_LEVEL_DIR, CONFIGURATION_SETTINGS

from debt_crisis.utilities import _name_sentiment_index_output_file

import pandas as pd
import numpy as np
from pytask import task


task_clean_sentiment_word_count_data_dependencies = {
    "raw_count_data": BLD / "data" / "filled_word_count_dict.pkl",
    "sentiment_dictionary": BLD / "data" / "sentiment_dictionary_clean.pkl",
}


def task_clean_sentiment_word_count_data(
    produces=BLD / "data" / "sentiment_data" / "sentiment_word_count_clean.csv",
):
    # Not best practice, but using the depends_on argument leads to a bug in pytask

    raw_count_data = pd.read_pickle(
        task_clean_sentiment_word_count_data_dependencies["raw_count_data"]
    )
    sentiment_dictionary = pd.read_pickle(
        task_clean_sentiment_word_count_data_dependencies["sentiment_dictionary"]
    )

    cleaned_data = clean_sentiment_count_data(
        raw_count_data, sentiment_dictionary
    ).reset_index(drop=True)

    cleaned_data.to_csv(produces, index=False)


def task_plot_empirical_word_frequency(
    depends_on=BLD / "data" / "sentiment_data" / "sentiment_word_count_clean.csv",
    produces=[
        BLD / "figures" / "sentiment_index" / "positive_word_frequency.png",
        BLD / "figures" / "sentiment_index" / "negative_word_frequency.png",
        TOP_LEVEL_DIR / "Input_for_Paper" / "figures" / "positive_word_frequency.png",
        TOP_LEVEL_DIR / "Input_for_Paper" / "figures" / "negative_word_frequency.png",
    ],
):
    data = pd.read_csv(depends_on)

    positive_plot = plot_actual_word_frequency(data, "Positive_Indicator")
    negative_plot = plot_actual_word_frequency(data, "Negative_Indicator")

    positive_plot.savefig(produces[0])
    negative_plot.savefig(produces[1])

    positive_plot.savefig(produces[2])
    negative_plot.savefig(produces[3])


for country in ["ireland", "greece"]:

    @task(id=country)
    def task_plot_sentiment_and_exuberance_index_for_country(
        depends_on=BLD
        / "data"
        / "sentiment_exuberance"
        / "exuberance_index_regression_quarterly.pkl",
        country=country,
        produces=[
            BLD
            / "figures"
            / "sentiment_index"
            / f"sentiment_exuberance_index_{country}.png",
            TOP_LEVEL_DIR
            / "Input_for_Paper"
            / "figures"
            / f"sentiment_exuberance_index_{country}.png",
        ],
    ):
        data = pd.read_pickle(depends_on)

        plot = plot_sentiment_index_and_exuberance_index_for_country(data, country)

        plot.savefig(produces[0])
        plot.savefig(produces[1])


for country in ["ireland", "greece"]:

    @task(id=country)
    def task_plot_sentiment_index_and_bond_yield_spread_for_country(
        depends_on=BLD
        / "data"
        / "event_study_approach"
        / _name_sentiment_index_output_file(
            "event_study_dataset", CONFIGURATION_SETTINGS, ".pkl"
        ),
        country=country,
        produces=[
            BLD
            / "figures"
            / "sentiment_index"
            / _name_sentiment_index_output_file(
                f"sentiment_index_and_bond_yield{country})",
                CONFIGURATION_SETTINGS,
                ".png",
            ),
            TOP_LEVEL_DIR
            / "Input_for_Paper"
            / "figures"
            / _name_sentiment_index_output_file(
                f"sentiment_index_and_bond_yield{country})",
                CONFIGURATION_SETTINGS,
                ".png",
            ),
        ],
    ):
        data = pd.read_pickle(depends_on)

        plot = plot_sentiment_index_and_bond_yield_spread_for_country(data, country)

        plot.savefig(produces[0])
        plot.savefig(produces[1])


# def task_create_empty_dataframe(
#     produces=BLD / "data" / "sentiment_data" / "sentiment_word_count_clean.csv"
# ):
#     df = pd.DataFrame(columns=["Word", "Count", "Negative", "Positive", "Uncertainty"])
#     df.to_csv(produces)
