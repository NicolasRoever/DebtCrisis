from debt_crisis.sentiment_index.clean_sentiment_data import (
    get_country_appearance_index_from_transcript_text,
)

import pandas as pd
import random


def get_random_country_transcript_snippet(
    data, countries_under_study, country_names_file, window_size=40
):
    row = data.sample(1)

    # Get the transcript text
    transcript = row["Transcript"].values[0]

    # Split the transcript into words
    transcript_words = transcript.split()

    # Find all countries present in the text
    countries_in_text = [
        country for country in countries_under_study if country in transcript_words
    ]

    # If no countries are present in the text, return None
    if not countries_in_text:
        return None

    # Randomly select a country from the countries present in the text
    country = random.choice(countries_in_text)

    transcript_id = row["Transcript_ID"].values[0]

    country_names = obtain_country_names(country, country_names_file)

    # Find the indices of the country in the transcript
    country_indices = get_country_appearance_index_from_transcript_text(
        transcript_words, country_names
    )

    # Randomly select an index from country_indices
    index = random.choice(country_indices)

    # Get the start and end indices for the snippet
    start = max(0, index - window_size)
    end = min(len(transcript_words), index + window_size)

    # Get the snippet
    snippet = " ".join(transcript_words[start:end])

    # Create a single-row DataFrame
    result = pd.DataFrame(
        {"Country": [country], "Transcript_ID": [transcript_id], "Snippet": [snippet]}
    )

    return result


def obtain_country_names(country, country_names_file):
    """THis function extracts the country names from the country names file."""

    country_row = country_names_file[
        country_names_file["name"].str.lower() == country.lower()
    ]
    if not country_row.empty:
        country_names = set(country_row.iloc[0].values.tolist())
    else:
        country_names = set()

    return country_names
