import pandas as pd
import pytask
import matplotlib.pyplot as plt
import pickle


def clean_sentiment_count_data(raw_count_data, sentiment_dictionary):
    """This function cleans the word count data."""
    count_data_clean = raw_count_data.T.reset_index()
    count_data_clean.columns = ["Word", "Count"]

    full_data = pd.merge(
        count_data_clean,
        sentiment_dictionary,
        on="Word",
        how="left",
        validate="one_to_one",
    )

    full_data = full_data[
        (full_data["Positive_Indicator"] != 0) | (full_data["Negative_Indicator"] != 0)
    ]

    full_data.sort_values(by="Count", ascending=False, inplace=True)

    return full_data


def plot_actual_word_frequency(data, indicator):
    """This function plots the actual word frequency taking the cleaned-count_data."""

    # Filter the DataFrame
    filtered_data = data[(data[indicator] == 1) & (data["Count"] > 600)]

    # Sort the DataFrame
    sorted_data = filtered_data.sort_values(by="Count", ascending=False)

    # Create the plot

    fig = plt.figure(figsize=(10, 6))

    plt.barh(sorted_data["Word"], sorted_data["Count"])
    plt.xlabel("Word")
    plt.ylabel("Count")
    plt.title(f"Word Counts for {indicator} Words")
    plt.yticks(fontsize=8)  # Adjust font size here

    return fig
