from debt_crisis.gpt_sentiment_index.gpt_sentiment_index import (
    get_random_country_transcript_snippet,
)

from debt_crisis.config import BLD, SRC, COUNTRIES_UNDER_STUDY

import pandas as pd


def task_create_random_code_snippets(
    depends_on=BLD / "data" / "df_transcripts_raw.pkl",
    country_names_file_path=SRC / "data" / "country_names" / "country_names.xlsx",
    countries_under_study=COUNTRIES_UNDER_STUDY,
    produces=BLD / "data" / "gpt_sentiment_data" / "df_random_transcript_snippets.xlsx",
):
    country_names_file = pd.read_excel(country_names_file_path)

    data = pd.read_pickle(depends_on)

    # Initialize an empty DataFrame
    df = pd.DataFrame()

    # Loop until df has 200 rows
    while len(df) < 200:
        # Execute the function
        result = get_random_country_transcript_snippet(
            data=data,
            countries_under_study=countries_under_study,
            country_names_file=country_names_file,
            window_size=40,
        )

        # If the result is not empty, append it to df
        if result is not None:
            df = pd.concat([df, result])
            print(len(df))

    # Save the DataFrame to a file
    df.to_excel(produces, index=False)
