import pandas as pd
from pytask import task
import pytask
import matplotlib.pyplot as plt
import pickle

from debt_crisis.config import (
    BLD,
    SRC,
    NO_LONG_RUNNING_TASKS,
    COUNTRIES_UNDER_STUDY,
    CONFIGURATION_SETTINGS,
)
from debt_crisis.sentiment_index.clean_sentiment_data import (
    combine_all_transcripts_into_dataframe,
    clean_transcript_data_df,
    tokenize_text_and_remove_non_alphabetic_characters_and_stop_words,
    clean_sentiment_dictionary_data,
    create_sentiment_dictionary_for_lookups,
    create_country_sentiment_index_for_one_transcript_and_print_transcript_number,
    calculate_loughlan_mcdonald_sentiment_index,
    create_word_count_dictionary,
)

from debt_crisis.utilities import _name_sentiment_index_output_file


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
    produces=BLD / "data" / "sentiment_dictionary_lookup.pkl",
):
    cleaned_data = pd.read_pickle(depends_on)
    word_sentiment_dict = create_sentiment_dictionary_for_lookups(cleaned_data)
    # Exporting to a file using pickle
    with open(produces, "wb") as f:
        pickle.dump(word_sentiment_dict, f)


def task_clean_McDonald_sentiment_index(
    depends_on=BLD
    / "data"
    / _name_sentiment_index_output_file(
        "mcdonald_sentiment_index", CONFIGURATION_SETTINGS, ".pkl"
    ),
    produces=BLD
    / "data"
    / _name_sentiment_index_output_file(
        "mcdonald_sentiment_index_cleaned", CONFIGURATION_SETTINGS, ".pkl"
    ),
):
    df = pd.read_pickle(depends_on)
    cleaned_data = df.melt(
        id_vars=["Date"], var_name="Country", value_name="McDonald_Sentiment_Index"
    )

    cleaned_data["Country"] = (
        cleaned_data["Country"].str.split("_", expand=True).iloc[:, -1]
    )

    cleaned_data.to_pickle(produces)


def task_plot_all_countries_sentiment_index_cumulative_sum(
    depends_on=BLD / "data" / "df_transcripts_clean_step_2.pkl",
    countries=COUNTRIES_UNDER_STUDY,
    produces=BLD / "figures" / "sentiment_index_all_countries_cum_sum.png",
):
    df = pd.read_pickle(depends_on)
    df.set_index("Date", inplace=True)

    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    for country in countries:
        # Calculate the moving average of the sentiment index column over the last 3 months
        df[f"moving_average_{country}"] = (
            df[f"Sentiment_Index_McDonald_{country.lower()}"]
            .rolling(window="90D")
            .sum()
        )
        ax.plot(
            df.index,
            df[f"moving_average_{country}"],
            label=f"Cumulative Sum(3 months) {country}",
        )

    handles, labels = ax.get_legend_handles_labels()

    lgd = ax.legend(
        handles,
        labels,
        loc="best",
        bbox_to_anchor=(0.5, -0.1),
        ncol=len(countries),
        fontsize="small",
        title="Countries",
        title_fontsize="medium",
        handlelength=2.5,
    )

    plt.grid(True)
    plt.xticks(rotation=45)

    fig.savefig(produces, bbox_extra_artists=(lgd,), bbox_inches="tight")


for country in COUNTRIES_UNDER_STUDY:

    @task(id=country)
    def task_plot_sentiment_index_cumulative_sum(
        depends_on=BLD
        / "data"
        / _name_sentiment_index_output_file(
            "mcdonald_sentiment_index", CONFIGURATION_SETTINGS, ".pkl"
        ),
        country=country,
        produces=BLD
        / "figures"
        / _name_sentiment_index_output_file(
            f"sentiment_index_{country}_cum_sum", CONFIGURATION_SETTINGS, ".png"
        ),
    ):
        df = pd.read_pickle(depends_on)
        # Calculate the moving average of the sentiment index column over the last 3 months
        df.set_index("Date", inplace=True)

        # Plot the moving average with date on the x-axis
        plt.figure(figsize=(10, 6))
        plt.plot(
            df.index,
            df[f"Sentiment_Index_McDonald_{country}"],
            label="Cumulative Sum / Earnings Calls last 3 Months",
        )
        plt.xlabel("Date")
        plt.ylabel("Sentiment Index")
        plt.title(
            f"Sentiment Index Over the Last 3 Months {country} in Moving Average Style"
        )
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(produces)
