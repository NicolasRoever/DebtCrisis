from debt_crisis.sentiment_index.clean_sentiment_data import (
    get_country_appearance_index_from_transcript_text,
)

import pandas as pd
import random
import re


def create_set_with_all_country_words(country_names_file):
    # Flatten the DataFrame to a single list
    country_words = country_names_file.values.flatten()

    # Remove NaN values
    country_words = [word for word in country_words if pd.notna(word)]

    # Create a set of unique words
    country_words_set = set(country_words)

    return country_words_set


def get_a_text_snippet_if_there_is_country_mentioned(
    full_data, country_words_set, context=50
):
    row = full_data.sample(1)
    transcript_id = row["Transcript_ID"].values[0]
    occuring_words = country_words_set.intersection(
        set(re.findall(r"\S+", row["Preprocessed_Transcript_Step_1"].values[0]))
    )

    if occuring_words:
        # Randomly pick a word
        word = random.choice(list(occuring_words))

        # Get the 'Transcript'
        transcript = row["Preprocessed_Transcript_Step_1"].values[0]

        # Split the 'Transcript' into words
        words = re.findall(r"\S+", transcript)

        # Find the index of the word
        index = words.index(word)

        # Get the 40 preceding and succeeding words
        start = max(0, index - context)
        end = min(len(words), index + context)
        snippet = words[start:end]

        # Create a single-row DataFrame
        result = pd.DataFrame(
            {
                "Keyword": [" ".join(word)],
                "Transcript_ID": [transcript_id],
                "Snippet": [snippet],
            }
        )

        return result

    else:
        return None
