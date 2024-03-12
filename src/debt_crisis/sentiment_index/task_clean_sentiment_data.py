import pandas as pd
import pytask
import matplotlib.pyplot as plt
import pickle

from debt_crisis.config import BLD, SRC, NO_LONG_RUNNING_TASKS, COUNTRIES_UNDER_STUDY
from debt_crisis.sentiment_index.clean_sentiment_data import (
    combine_all_transcripts_into_dataframe,
    clean_transcript_data_df,
    tokenize_text_and_remove_non_alphabetic_characters_and_stop_words,
    clean_sentiment_dictionary_data,
    create_sentiment_dictionary_for_lookups,
    create_country_sentiment_index_for_one_transcript_and_print_transcript_number,
)


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


def task_plot_sentiment_index_austria(
    depends_on=BLD / "data" / "df_transcripts_clean_step_2.pkl",
    produces=BLD / "figures" / "sentiment_index_austria.png",
):
    df = pd.read_pickle(depends_on)
    # Calculate the moving average of the sentiment index column over the last 3 months
    df.set_index("Date", inplace=True)

    df["moving_average"] = (
        df["Sentiment_Index_McDonald_austria"].rolling(window="90D").sum()
    )

    # Plot the moving average with date on the x-axis
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["moving_average"], label="Cumulative Sum(3 months)")
    plt.xlabel("Date")
    plt.ylabel("Moving Average of Sentiment Index")
    plt.title("Moving Average of Sentiment Index Over the Last 3 Months")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(produces)


# task_clean_transcript_data_step_2_dependencies = {
#     "df_transcripts_step_1" : BLD / "data" / "df_transcripts_clean_step_1.pkl",
#     "sentiment_dictionary" : BLD / "data" / "sentiment_dictionary_lookup.pickle",
#     "country_names_file" : SRC / "data" / "country_names" / "country_names.xlsx",
#     "words_environment" : 20
# }


# def task_clean_transcript_data_step_2(
#     depends_on=task_clean_transcript_data_step_2_dependencies,
#     countries_under_study=COUNTRIES_UNDER_STUDY,
#     produces=BLD / "data" / "df_transcripts_clean_step_2.pkl"):

#     #Load Data
#     cleaned_data = pd.read_pickle(depends_on["df_transcripts_step_1"])
#     lookup_dict = pickle.load(open(depends_on["sentiment_dictionary"], "rb"))
#     country_names_file = pd.read_excel(depends_on["country_names_file"])
#     words_environment = depends_on["words_environment"]

#     for country in countries_under_study:
#         print(country)

#         #Initialize Row counter
#         create_country_sentiment_index_for_one_transcript_and_print_transcript_number.row_counter = 0

#         #Make Sentiment Index
#         cleaned_data[f"Sentiment_Index_McDonald_{country}"] = cleaned_data["Preprocessed_Transcript_Step_1"].apply(
#         lambda x: create_country_sentiment_index_for_one_transcript_and_print_transcript_number(
#             x, lookup_dict, words_environment, country, country_names_file
#         ))

#     cleaned_data.to_pickle(produces)


# @pytask.mark.skipif(NO_LONG_RUNNING_TASKS, reason="Skip long-running tasks.")
# def task_combine_all_transcripts_into_initial_dataframe(
#     data_directory=str(
#         SRC / "data" / "transcripts" / "raw" / "Eikon 2002 - 2022",
#     ),  # This should be fixed !!!
#     produces=BLD / "data" / "df_transcripts_raw.pkl",
# ):
#     full_dataframe = combine_all_transcripts_into_dataframe(data_directory)

#     full_dataframe.to_pickle(produces)


# @pytask.mark.skipif(NO_LONG_RUNNING_TASKS, reason="Skip long-running tasks.")
# def task_clean_transcript_data_step_1(
#     depends_on=BLD / "data" / "df_transcripts_raw.pkl",
#     produces=BLD / "data" / "df_transcripts_clean_step_1.pkl",
# ):
#     raw_data = pd.read_pickle(depends_on)
#     cleaned_data = clean_transcript_data_df(raw_data)

#     cleaned_data.to_pickle(produces)
