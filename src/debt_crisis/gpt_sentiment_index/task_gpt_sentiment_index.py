from debt_crisis.gpt_sentiment_index.gpt_sentiment_index import (
    create_set_with_all_country_words,
    get_a_text_snippet_if_there_is_country_mentioned,
    extract_gpt_training_dataset_from_preprocessed_transcripts,
    plot_country_occurrences,
)

from debt_crisis.config import (
    BLD,
    SRC,
    COUNTRIES_UNDER_STUDY,
    MAPPING_COUNTRY_NAMES_TO_COUNTRY,
    TOP_LEVEL_DIR,
)

from debt_crisis.utilities import _check_for_missing_values_in_dataframe_column

import pandas as pd
import re
import random

random.seed(43)  # First set of transcripts was with seed 42


def task_create_random_code_snippets(
    depends_on=BLD
    / "data"
    / "gpt_sentiment_data"
    / "df_gpt_sentiment_training_dataset_cleaned.pkl",
    produces=BLD / "data" / "gpt_sentiment_data" / "df_random_transcript_snippets.xlsx",
):
    gpt_file = pd.read_pickle(depends_on)

    random_sample = gpt_file.sample(200)

    random_sample.to_excel(produces, index=False)


# def task_create_gpt_sentiment_index_dataset(
#     depends_on=BLD / "data" / "df_transcripts_clean_step_1.pkl",
#     country_names_file_path=SRC / "data" / "country_names" / "country_names.xlsx",
#     mapping_country_names_to_country=MAPPING_COUNTRY_NAMES_TO_COUNTRY,
#     produces=BLD
#     / "data"
#     / "gpt_sentiment_data"
#     / "df_gpt_sentiment_training_dataset.pkl",
# ):
#     country_names_file = pd.read_excel(country_names_file_path)
#     country_names_set = create_set_with_all_country_words(country_names_file)
#     transcripts_data = pd.read_pickle(depends_on)

#     gpt_file = pd.DataFrame()

#     for index, row in transcripts_data.iterrows():
#         transcript_snippets = (
#             extract_gpt_training_dataset_from_preprocessed_transcripts(
#                 row, country_names_set
#             )
#         )

#         gpt_file = pd.concat([gpt_file, transcript_snippets])

#         print(index)

#     gpt_file.to_pickle(produces)


def task_clean_gpt_sentiment_index_dataset(
    depends_on=BLD
    / "data"
    / "gpt_sentiment_data"
    / "df_gpt_sentiment_training_dataset.pkl",
    mapping_country_names_to_country=MAPPING_COUNTRY_NAMES_TO_COUNTRY,
    produces=BLD
    / "data"
    / "gpt_sentiment_data"
    / "df_gpt_sentiment_training_dataset_cleaned.pkl",
):
    gpt_file = pd.read_pickle(depends_on).reset_index(drop=True)

    gpt_file["Country"] = gpt_file["Keyword"].map(mapping_country_names_to_country)

    _check_for_missing_values_in_dataframe_column(gpt_file, "Country")

    gpt_file["Snippet_ID"] = gpt_file.index + 1

    gpt_file.to_pickle(produces)


def task_plot_number_of_snippets_per_country(
    depends_on=BLD
    / "data"
    / "gpt_sentiment_data"
    / "df_gpt_sentiment_training_dataset_cleaned.pkl",
    produces=TOP_LEVEL_DIR
    / "Input_for_Paper"
    / "figures"
    / "number_of_snippets_per_country.pdf",
):
    gpt_file = pd.read_pickle(depends_on)
    figure = plot_country_occurrences(gpt_file)
    figure.savefig(produces)
