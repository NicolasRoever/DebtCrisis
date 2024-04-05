from debt_crisis.gpt_sentiment_index.gpt_sentiment_index import (
    create_set_with_all_country_words,
    get_a_text_snippet_if_there_is_country_mentioned,
)

from debt_crisis.config import BLD, SRC, COUNTRIES_UNDER_STUDY

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
