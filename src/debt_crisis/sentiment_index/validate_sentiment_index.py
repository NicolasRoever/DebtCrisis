import pandas as pd
import pytask
import matplotlib.pyplot as plt
import pickle
import seaborn as sns


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

    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"

    sns.set_style("white")

    # Filter the DataFrame
    filtered_data = (
        data[(data[indicator] == 1)].sort_values(by="Count", ascending=False).head(20)
    )

    # Sort the DataFrame
    sorted_data = filtered_data.sort_values(by="Count", ascending=False)

    # Create the plot
    fig = plt.figure(figsize=(8, 7))

    plt.barh(sorted_data["Word"], sorted_data["Count"], color="#3c5488")
    plt.xlabel("Total Number of Occurences")
    plt.yticks(fontsize=8)  # Adjust font size here

    # Remove the top and right spines from plot
    sns.despine()

    return fig


def plot_sentiment_index_and_bond_yield_spread_for_country(
    first_step_regression_data,
    country,
    color_scheme=["#3c5488", "#e64b35", "#4dbbd5", "#00a087", "#f39b7f"],
):
    # Filter the data for the given country
    country_data = first_step_regression_data[
        first_step_regression_data["Country"] == country
    ]
    country_data = country_data.sort_values("Date")

    # Set the style of the plot
    sns.set_style("white")

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(6, 5))

    ax1.plot(
        country_data["Date"],
        country_data["Bond_Yield_Spread"],
        marker="o",
        color=color_scheme[0],
        label=f"Bond Yield Spread {country.capitalize()} ",
    )
    ax1.set_ylabel("Bond Yield Spread in Basis Points", fontsize=14)

    ax2 = ax1.twinx()
    ax2.plot(
        country_data["Date"],
        country_data["McDonald_Sentiment_Index"],
        marker="o",
        color=color_scheme[1],
        label=f"Sentiment Index {country.capitalize()} ",
    )
    ax2.set_ylabel("Sentiment Index", fontsize=14)
    ax2.invert_yaxis()  # Invert the right y-axis

    # Add a horizontal line at y=0
    # ax1.axhline(0, color='grey', linestyle=':')

    # Set labels
    plt.xlabel("Time", fontsize=14)

    # Keep only the y-axis and x-axis
    sns.despine(left=False, bottom=False, right=False, top=True)

    # Create a legend for both lines
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    # Use LaTeX style for the font
    plt.rc("text", usetex=True)

    plt.close(fig)

    # Align the zero of both y-axes
    # ax1.set_ylim(min(ax1.get_ylim()[0], ax2.get_ylim()[0]), max(ax1.get_ylim()[1], ax2.get_ylim()[1]))
    # ax2.set_ylim( 0.2, -1.05)

    return fig


def plot_sentiment_index_and_exuberance_index_for_country(
    first_step_regression_data,
    country,
    color_scheme=["#3c5488", "#e64b35", "#4dbbd5", "#00a087", "#f39b7f"],
):
    """Input for this function is the dataframe created by the task
    task_run_exuberance_index_regression_quarterly."""

    # Filter the data for the given country
    country_data = first_step_regression_data[
        first_step_regression_data["Country"] == country
    ]
    country_data = country_data.sort_values("Date")

    # Set the style of the plot
    sns.set_style("white")

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(6, 5))

    ax1.plot(
        country_data["Date"],
        country_data["Residuals_Exuberance_Regression"],
        linestyle="-",
        color=color_scheme[0],
        label=f"Residuale, i.e. Unfounded Sentiment",
        alpha=0.75,
    )
    ax1.set_ylabel("Index Value", fontsize=14)

    ax1.plot(
        country_data["Date"],
        country_data["McDonald_Sentiment_Index"],
        linestyle="-",
        color=color_scheme[1],
        label=f"Raw Sentiment Index",
        alpha=0.75,
    )

    # Add a horizontal line at y=0
    ax1.axhline(0, color="grey", linestyle=":")

    # Set labels
    plt.xlabel("Time", fontsize=14)

    # Keep only the y-axis and x-axis
    sns.despine(left=False, bottom=False, right=False, top=True)

    # Remove the right y-axis
    ax1.spines["right"].set_visible(False)

    # Create a legend for both lines
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc="upper right")

    # Use LaTeX style for the font
    plt.rc("text", usetex=True)

    # Align the zero of both y-axes
    # ax1.set_ylim(min(ax1.get_ylim()[0], ax2.get_ylim()[0]), max(ax1.get_ylim()[1], ax2.get_ylim()[1]))
    # ax2.set_ylim( 0.2, -1.05)

    return fig


def plot_mcdonald_sentiment_index_for_countries(
    first_step_regression_data,
    countries,
    color_scheme=["#3c5488", "#e64b35", "#4dbbd5", "#00a087", "#f39b7f"],
):
    """Input for this function is the dataframe created by the task
    task_run_exuberance_index_regression_quarterly."""

    # Filter the data for the given country
    country_data = first_step_regression_data[
        first_step_regression_data["Country"] == country
    ]
    country_data = country_data.sort_values("Date")

    # Set the style of the plot
    sns.set_style("white")

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(6, 5))

    ax1.plot(
        country_data["Date"],
        country_data["McDonald_Sentiment_Index"],
        linestyle="-",
        color=color_scheme[1],
        label=f"Raw 'Count' Sentiment Index",
        alpha=0.75,
    )

    # Add a horizontal line at y=0
    ax1.axhline(0, color="grey", linestyle=":")

    # Set labels
    plt.xlabel("Time", fontsize=14)

    # Keep only the y-axis and x-axis
    sns.despine(left=False, bottom=False, right=False, top=True)

    # Remove the right y-axis
    ax1.spines["right"].set_visible(False)

    # Create a legend for both lines
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc="upper right")

    # Use LaTeX style for the font
    plt.rc("text", usetex=True)

    # Align the zero of both y-axes
    # ax1.set_ylim(min(ax1.get_ylim()[0], ax2.get_ylim()[0]), max(ax1.get_ylim()[1], ax2.get_ylim()[1]))
    # ax2.set_ylim( 0.2, -1.05)

    return fig
