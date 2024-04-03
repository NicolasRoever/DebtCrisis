import pandas as pd
from scipy.stats import t, norm
import numpy as np
from datetime import datetime, timedelta
import statsmodels.api as sm
import statsmodels.formula.api as smf
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates


def estimate_normal_returns(
    sentiment_data, country, event_date, estimation_window=[-20, 0], model="mean_model"
):
    """This function estimates the normal return model for the event study.
    Args:
    sentiment_data (pandas.DataFrame): The cleaned sentiment data
    country (str): The country for which the event study is being conducted
    event_date (datetime): Event date
    estimation_window (list): Estimation window for the event study

    Returns:
    pandas.Dataframe:
            columns:    Normal returns (float) Estimated normal returns for full data
                        Date (datetime) Date of the data
    degrees_of_freedom: degrees_of_freedom (int) Degrees of freedom of the model
    residual variance: residual_variance (float) Residual variance of the model

    """

    # Convert event_date to datetime if it's not already
    if not isinstance(event_date, datetime):
        event_date = pd.to_datetime(event_date)

    # Define the start and end of the estimation window
    start_date = event_date + timedelta(days=estimation_window[0])
    end_date = event_date + timedelta(days=estimation_window[1])

    # Filter the stock and factor data for the estimation window
    sentiment_data_filter = sentiment_data[
        (sentiment_data["Date"] >= pd.to_datetime(start_date).to_pydatetime()[0])
        & (sentiment_data["Date"] <= pd.to_datetime(end_date).to_pydatetime()[0])
        & (sentiment_data["Country"] == country)
    ]

    # Create the model
    model = smf.ols(formula="McDonald_Sentiment_Index ~ 1", data=sentiment_data_filter)

    # Fit the model
    results = model.fit()

    # Use the fitted model to estimate the normal returns for the full data
    full_data = sentiment_data[sentiment_data["Country"] == country].copy()
    full_data["Normal_Returns"] = results.predict(full_data)

    degrees_of_freedom = estimation_window[1] - estimation_window[0]

    residual_variance = results.ssr / results.df_resid

    return full_data[["Date", "Normal_Returns"]], degrees_of_freedom, residual_variance


def run_event_study_single_event(
    sentiment_data,
    country,
    event_date,
    estimation_window=[-20, 0],
    event_window=[0, 10],
):
    """This function runs an event study for a single event.
    Args:
    sentiment_data: (pandas.DataFrame): The cleaned sentiment data
    country (str): The country for which the event study is being conducted
    event_date (datetime): Event date
    estimation_window (list): Estimation window for the event study
    event_window (list): Event window for the event study

    Returns:
    pandas.DataFrame: Event study results
    """

    # Estimate the normal returns
    normal_returns, degrees_of_freedom, residual_variance = estimate_normal_returns(
        sentiment_data, country, event_date, estimation_window
    )

    # Generate the event study output
    event_study_output = generate_event_study_output(
        sentiment_data,
        country,
        event_date,
        normal_returns,
        residual_variance,
        degrees_of_freedom,
        event_window,
    )

    return event_study_output


def generate_event_study_output(
    sentiment_data,
    country,
    event_date,
    normal_returns,
    residual_variance,
    degrees_of_freedom,
    event_window=[0, 20],
):
    """This function generates the output for the event study.
    Args:
    sentimetn_data (pandas.DataFrame): Clean Sentiment Data
    country (str): Country for the event study
    event_date (pandas.DataFrame): Event date
    normal_returns (pandas.DataFrame): Normal returns from the estimate_normal_returns function
    event_window (list): Event window for the event study

    Returns:
    pandas.DataFrame: Event study results
            columns:
                        Abnormal_Returns (float) Abnormal returns for the event window
                        Cumulative Abnormal Returns (float) Cumulative abnormal returns for the event window
                        CAR_Variance (float) Variance of the cumulative abnormal returns
                        t-statistic (float) t-statistic for the CAR
                        p-value (float) p-value for the CAR
                        Date (datetime) Date of the data
                        Event_ID (int) Unique identifier for the event
                        Daily_Return (float) Daily returns for the stock
                        Ticker (str) Ticker for the stock

    """

    # Define the start and end of the event window
    start_date = event_date[0] + timedelta(days=event_window[0])
    end_date = event_date[0] + timedelta(days=event_window[1])

    sentiment_data_country = sentiment_data[sentiment_data["Country"] == country]

    # Merge the stock data and normal returns
    merged_data = pd.merge(
        sentiment_data_country,
        normal_returns,
        on="Date",
        how="left",
        validate="one_to_one",
    )

    # Filter the merged data for the event window
    event_window_data = merged_data[
        (merged_data["Date"] >= start_date) & (merged_data["Date"] <= end_date)
    ]

    # Calculate the abnormal returns
    event_window_data["Abnormal_Returns"] = (
        event_window_data["McDonald_Sentiment_Index"]
        - event_window_data["Normal_Returns"]
    )

    # Calculate the cumulative abnormal returns
    event_window_data.loc[:, "Cumulative_Abnormal_Returns"] = event_window_data[
        "Abnormal_Returns"
    ].cumsum()

    # Calculate CAR_Variance
    event_window_data.loc[:, "CAR_Variance"] = (
        event_window_data.reset_index().index * residual_variance
    )
    print(
        "This is wrong! I took it from the eventstudy package, but we ignore the variance coming from estimation uncertainty. Will fix it later."
    )

    # Calculate CI's
    event_window_data.loc[:, "CI_upper_bound_95"] = event_window_data[
        "Cumulative_Abnormal_Returns"
    ] + t.ppf(0.975, degrees_of_freedom) * np.sqrt(event_window_data["CAR_Variance"])

    event_window_data.loc[:, "CI_lower_bound_95"] = event_window_data[
        "Cumulative_Abnormal_Returns"
    ] - t.ppf(0.975, degrees_of_freedom) * np.sqrt(event_window_data["CAR_Variance"])

    # Calculate t-statistic
    event_window_data.loc[:, "t-statistic"] = event_window_data[
        "Cumulative_Abnormal_Returns"
    ] / np.sqrt(event_window_data["CAR_Variance"])

    # Calculate p-value
    event_window_data.loc[:, "p-value"] = 1 - t.cdf(
        abs(event_window_data["t-statistic"]), degrees_of_freedom
    )

    event_window_data.loc[
        :, "Time_Relative_To_Event"
    ] = event_window_data.reset_index().index

    return event_window_data


def plot_CAR_over_time_with_ci_plotly(
    event_study_results, sentiment_data, country, event_date, confidence_level=0.95
):
    """Plot Cumulative Abnormal Returns (CAR) over time with confidence intervals using
    Plotly.

    Args:
    event_study_results (pandas.DataFrame): DataFrame generated by the function generate_event_study_output
    sentiment_data: (pandas.DataFrame): The cleaned sentiment data
    event_date (pd.Timestamp): Date of the event
    confidence_level (float, optional): Confidence level for confidence intervals, default is 0.95.

    Returns:
    plotly.graph_objects.Figure: Plotly figure object.

    """

    # Raw Data
    start_date = pd.to_datetime(event_date) + np.timedelta64(-20, "D")
    end_date = event_study_results["Date"].max()

    filtered_sentiment_data = sentiment_data[
        (sentiment_data["Country"] == country)
        & (sentiment_data["Date"] >= start_date[0])
        & (sentiment_data["Date"] <= end_date)
    ]

    # Plot

    # Create a new figure and a subplot
    fig, ax = plt.subplots(figsize=(13, 8))

    # Plot the data
    sns.lineplot(
        x="Date",
        y="McDonald_Sentiment_Index",
        data=filtered_sentiment_data,
        ax=ax,
        label="Actual Daily Returns",
        color="black",
        linestyle=":",
    )
    sns.lineplot(
        x="Date",
        y="McDonald_Sentiment_Index",
        data=event_study_results,
        ax=ax,
        label="Normal Returns",
        color="blue",
        linestyle="--",
    )
    sns.lineplot(
        x="Date",
        y="Abnormal_Returns",
        data=event_study_results,
        ax=ax,
        label="Abnormal Returns",
        color="green",
    )
    sns.lineplot(
        x="Date",
        y="Cumulative_Abnormal_Returns",
        data=event_study_results,
        ax=ax,
        label="Culmulative Abnormal Returns",
        color="red",
    )

    # Plot the confidence intervals
    ax.fill_between(
        event_study_results["Date"],
        event_study_results["CI_lower_bound_95"],
        event_study_results["CI_upper_bound_95"],
        color="grey",
        alpha=0.3,
    )

    # Add a vertical line for the event date
    ax.axvline(pd.to_datetime(event_date), color="red", linestyle="-", linewidth=1)

    # Set the title and labels
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("Returns")

    # Set the legend
    ax.legend(loc="upper right")

    # Format the x-axis to display dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    return fig
