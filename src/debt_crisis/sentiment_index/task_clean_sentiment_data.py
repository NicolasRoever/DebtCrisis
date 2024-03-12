import pandas as pd
import pytask
import matplotlib.pyplot as plt
import pickle

from debt_crisis.config import BLD, SRC, NO_LONG_RUNNING_TASKS
from debt_crisis.sentiment_index.clean_sentiment_data import (
    combine_all_transcripts_into_dataframe,
    clean_transcript_data_df,
    tokenize_text_and_remove_non_alphabetic_characters_and_stop_words,
    clean_sentiment_dictionary_data,
    create_sentiment_dictionary_for_lookups,
)


@pytask.mark.skipif(NO_LONG_RUNNING_TASKS, reason="Skip long-running tasks.")
def task_combine_all_transcripts_into_initial_dataframe(
    data_directory=str(
        SRC / "data" / "transcripts" / "raw" / "Eikon 2002 - 2022",
    ),  # This should be fixed !!!
    produces=BLD / "data" / "df_transcripts_raw.pkl",
):
    full_dataframe = combine_all_transcripts_into_dataframe(data_directory)

    full_dataframe.to_pickle(produces)


@pytask.mark.skipif(NO_LONG_RUNNING_TASKS, reason="Skip long-running tasks.")
def task_clean_transcript_data_step_1(
    depends_on=BLD / "data" / "df_transcripts_raw.pkl",
    produces=BLD / "data" / "df_transcripts_clean_step_1.pkl",
):
    raw_data = pd.read_pickle(depends_on)
    cleaned_data = clean_transcript_data_df(raw_data)

    cleaned_data.to_pickle(produces)


# def task_clean_transcript_data_step_2(
#     depends_on=BLD / "data" / "df_transcripts_clean_step_1.pkl",
#     produces=BLD / "data" / "df_transcripts_clean_step_2.pkl"):

#     cleaned_data = pd.read_pickle(depends_on)
#     cleaned_data["Preprocessed_Transcript_Step_2"] = tokenize_text_and_remove_non_alphabetic_characters_and_stop_words(cleaned_data["Preprocessed_Transcript_Step_1"])

#     cleaned_data.to_pickle(produces)


def task_clean_sentiment_dictionary(
    depends_on=SRC
    / "data"
    / "sentiment_dictionary"
    / "Loughran-McDonald_MasterDictionary_1993-2021.csv",
    produces=BLD / "data" / "sentiment_dictionary_clean.pkl",
):
    raw_data = pd.read_csv(depends_on)
    cleaned_data = clean_sentiment_dictionary_data(raw_data)
    cleaned_data.to_pickle(produces)


def task_plot_raw_data_sentiment_dictionart_barplot(
    depends_on=BLD / "data" / "sentiment_dictionary_clean.pkl",
    produces=BLD / "figures" / "barplot_sentiment_dictionary.png",
):
    df = pd.read_pickle(depends_on)
    positive_sum = df["Positive"].sum()
    negative_sum = df["Negative"].sum()

    # Create a bar plot
    plt.bar(["Positive", "Negative"], [positive_sum, negative_sum])

    # Add labels and title
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.title("Sum of Positive and Negative Words")

    # Save the plot as a PNG file (replace 'figure.png' with your desired file name)
    plt.savefig(produces)


def task_plot_histogram_negative_words(
    depends_on=BLD / "data" / "sentiment_dictionary_clean.pkl",
    produces=BLD / "figures" / "histogram_negative_sentiment_dictionary.png",
):
    df = pd.read_pickle(depends_on)
    df["Negative"].plot(kind="hist")
    plt.xlabel("Count")
    plt.ylabel("Frequency")
    plt.title("Histogram of Negative Words")
    plt.savefig(produces)


def task_create_loughlan_mcdonald_dictionary_for_lookup(
    depends_on=BLD / "data" / "sentiment_dictionary_clean.pkl",
    produces=BLD / "data" / "sentiment_dictionary_lookup.pickle",
):
    cleaned_data = pd.read_pickle(depends_on)
    word_sentiment_dict = create_sentiment_dictionary_for_lookups(cleaned_data)
    # Exporting to a file using pickle
    with open(produces, "wb") as f:
        pickle.dump(word_sentiment_dict, f)
