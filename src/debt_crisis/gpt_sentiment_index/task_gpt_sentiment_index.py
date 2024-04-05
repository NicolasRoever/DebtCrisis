from debt_crisis.gpt_sentiment_index.gpt_sentiment_index import (
    create_set_with_all_country_words,
    get_a_text_snippet_if_there_is_country_mentioned,
    extract_gpt_training_dataset_from_preprocessed_transcripts,
)

from debt_crisis.config import (
    BLD,
    SRC,
    COUNTRIES_UNDER_STUDY,
    MAPPING_COUNTRY_NAMES_TO_COUNTRY,
)

import pandas as pd
import re


def task_create_random_code_snippets(
    depends_on=BLD / "data" / "df_transcripts_clean_step_1.pkl",
    country_names_file_path=SRC / "data" / "country_names" / "country_names.xlsx",
    countries_under_study=COUNTRIES_UNDER_STUDY,
    produces=BLD / "data" / "gpt_sentiment_data" / "df_random_transcript_snippets.xlsx",
):
    country_names_file = pd.read_excel(country_names_file_path)

    country_words_set = create_set_with_all_country_words(country_names_file)

    data = pd.read_pickle(depends_on)

    final_output = pd.DataFrame()

    while len(final_output) < 200:
        single_snippet = get_a_text_snippet_if_there_is_country_mentioned(
            data, country_words_set
        )

        if single_snippet is not None:
            final_output = pd.concat([final_output, single_snippet])

            print(len(final_output))

    # Clean the dataframe

    final_output["Snippet"] = final_output["Snippet"].apply(lambda x: " ".join(x))

    final_output = final_output.drop_duplicates(subset=["Snippet"])

    final_output.to_excel(produces, index=False)


def task_create_gpt_sentiment_index_dataset(
    depends_on=BLD / "data" / "df_transcripts_clean_step_1.pkl",
    country_names_file_path=SRC / "data" / "country_names" / "country_names.xlsx",
    mapping_country_names_to_country=MAPPING_COUNTRY_NAMES_TO_COUNTRY,
    produces=BLD
    / "data"
    / "gpt_sentiment_data"
    / "df_gpt_sentiment_training_dataset.pkl",
):
    country_names_file = pd.read_excel(country_names_file_path)
    country_names_set = create_set_with_all_country_words(country_names_file)
    transcripts_data = pd.read_pickle(depends_on)

    gpt_file = pd.DataFrame()

    for index, row in transcripts_data.iterrows():
        transcript_snippets = (
            extract_gpt_training_dataset_from_preprocessed_transcripts(
                row, country_names_set
            )
        )

        gpt_file = pd.concat([gpt_file, transcript_snippets])

        print(index)

    gpt_file["Country"] = gpt_file["Keyword"].map(mapping_country_names_to_country)

    gpt_file.to_pickle(produces)
