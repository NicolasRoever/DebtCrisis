from debt_crisis.sentiment_index.clean_sentiment_data import (
    get_country_appearance_index_from_transcript_text,
)

import pandas as pd
import random
import re
import matplotlib.pyplot as plt
import seaborn as sns


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


def extract_gpt_training_dataset_from_preprocessed_transcripts(
    transcript_row, country_names_set, context=400
):
    """This function is written to be applied on every row of the dataframe preprocessed
    transcripts step 1. It extracts a new dataframe with the complete data we want to
    parse to a large language model.

    Args:
        transcript (str): The preprocessed transcript
        country_names_set (set): A set with all country names

    Returns:
        pd.DataFrame:
            columns: Keyword ('str'): The keyword based on which the snippet was chosen
                    Transcript_ID ('str'): The transcript ID
                    Snippet ('str'): The transcript snippet

    """

    transcript_id = transcript_row["Transcript_ID"]
    transcript_string = transcript_row["Preprocessed_Transcript_Step_1"]

    occuring_words = country_names_set.intersection(
        set(re.findall(r"\S+", transcript_string))
    )

    output = pd.DataFrame()

    for word in occuring_words:
        word_indexes = get_index_where_words_occur({word}, transcript_string)

        for index in word_indexes:
            start = max(0, index - context)
            end = min(len(transcript_string), index + context)
            snippet = transcript_string[start:end]

            # Create a single-row DataFrame
            result = pd.DataFrame(
                {
                    "Keyword": [word],
                    "Transcript_ID": [transcript_id],
                    "Snippet": [snippet],
                }
            )

            output = pd.concat([output, result])

    return output


def get_index_where_words_occur(set_of_words, text):
    """This function returns the index of the words in the text.

    Args:
        set_of_words (set): A set of words
        text (str): The text in which we want to find the words

    Returns:
        list: A list with the indexes of the words in the text

    """

    indexes = [
        m.start()
        for word in set_of_words
        for m in re.finditer(r"\b" + re.escape(word) + r"\b", text)
    ]

    return indexes


def plot_country_occurrences(data):
    """This function plots the occurrences of each country in the 'Country' column."""

    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"

    sns.set_style("white")

    # Capitalize the country names
    data["Country"] = data["Country"].str.capitalize()

    # Count the occurrences of each country
    country_counts = data["Country"].value_counts()

    # Calculate the percentage for each country
    total = len(data)
    percentages = 100 * country_counts / total

    # Sort the counts in descending order
    sorted_counts = country_counts.sort_values(ascending=False)

    # Create the plot
    fig = plt.figure(figsize=(8, 7))

    bars = plt.barh(sorted_counts.index, sorted_counts.values, color="#3c5488")
    plt.xlabel("Total Number of Occurrences")
    plt.yticks(fontsize=8)  # Adjust font size here

    # Add the percentages to the right of the bars
    for bar, percentage in zip(bars, percentages.sort_values(ascending=False)):
        plt.text(
            bar.get_width() + 200,
            bar.get_y() + bar.get_height() / 2,
            f" ({percentage:.1f} \\%)",
            va="center",
        )

    # Remove the top and right spines from plot
    sns.despine()

    return fig
