import pandas as pd
import matplotlib.pyplot as plt

from debt_crisis.regression_analysis.regression_analysis import (
    create_dataset_step_one_regression,
    run_first_step_regression,
    run_second_step_regression,
    run_third_step_regression,
)

from debt_crisis.config import BLD


dependencies_task_create_dataset_step_one_regression = {
    "bond_yield_data": BLD / "data" / "financial_data" / "eu_yields_10y_cleaned.pkl",
    "debt_to_gdp_data": BLD / "data" / "financial_data" / "debt-to-gdp_EU_cleaned.pkl",
    "sentiment_index_data": BLD / "data" / "mcdonald_sentiment_index_cleaned.pkl",
    "ratings_data": BLD / "data" / "financial_data" / "processed_ratings.pkl",
    "gdp_data": BLD / "data" / "financial_data" / "gdp_data_cleaned.pkl",
    "current_account_data": BLD
    / "data"
    / "financial_data"
    / "current_account_EU_cleaned.pkl",
    "stoxx_data": BLD / "data" / "financial_data" / "stoxx50_vstoxx.pkl",
}


def task_create_dataset_step_one_regression(
    depends_on=dependencies_task_create_dataset_step_one_regression,
    produces=BLD / "data" / "step_one_regression_dataset.pkl",
):
    bond_yield_data = pd.read_pickle(depends_on["bond_yield_data"])

    debt_to_gdp_data = pd.read_pickle(depends_on["debt_to_gdp_data"])

    sentiment_index_data = pd.read_pickle(depends_on["sentiment_index_data"])

    ratings_data = pd.read_pickle(depends_on["ratings_data"])

    gdp_data = pd.read_pickle(depends_on["gdp_data"])

    current_account_data = pd.read_pickle(depends_on["current_account_data"])

    stoxx_data = pd.read_pickle(depends_on["stoxx_data"])

    dataset = create_dataset_step_one_regression(
        bond_yield_data,
        debt_to_gdp_data,
        sentiment_index_data,
        ratings_data,
        gdp_data,
        current_account_data,
        stoxx_data,
    )

    dataset.to_pickle(produces)


def task_run_first_step_regression(
    depends_on=BLD / "data" / "step_one_regression_dataset.pkl",
    produces=[
        BLD / "models" / "first_step_regression.txt",
        BLD / "data" / "step_one_regression_dataset_output.pkl",
    ],
):
    # Load the data
    data = pd.read_pickle(depends_on)

    # Run the regression
    model = run_first_step_regression(data)

    # Get the summary of the model
    model_summary = model.summary()

    # Save the summary as a text file
    with open(produces[0], "w") as file:
        file.write(model_summary.as_text())

    data["Fitted_Values_Step_One_Regression"] = model.fittedvalues
    data["Residuals_Step_One_Regression"] = model.resid

    data.to_pickle(produces[1])


def task_run_second_step_regression(
    depends_on=BLD / "data" / "step_one_regression_dataset_output.pkl",
    produces=[
        BLD / "models" / "second_step_regression.txt",
        BLD / "data" / "step_two_regression_dataset_output.pkl",
    ],
):
    data = pd.read_pickle(depends_on)

    model = run_second_step_regression(data)

    model_summary = model.summary()

    with open(produces[0], "w") as file:
        file.write(model_summary.as_text())

    data["Residuals_Step_Two_Regression"] = model.resid

    data.to_pickle(produces[1])


def task_run_third_step_regression(
    depends_on=BLD / "data" / "step_two_regression_dataset_output.pkl",
    produces=BLD / "models" / "third_step_regression.txt",
):
    data = pd.read_pickle(depends_on)

    model = run_third_step_regression(data)

    model_summary = model.summary()

    with open(produces, "w") as file:
        file.write(model_summary.as_text())


def task_plot_sentiment_index_vs_bond_yield(
    depends_on=BLD / "data" / "step_one_regression_dataset.pkl",
    produces=BLD / "figures" / "sentiment_index_vs_bond_yield.png",
):
    # Load the data
    data = pd.read_pickle(depends_on)

    # Create a scatter plot
    plt.scatter(data["McDonald_Sentiment_Index"], data["Bond_Yield"], s=10)

    # Annotate points with a McDonald Sentiment Index of less than -0.4
    for i, row in data[data["McDonald_Sentiment_Index"] < -0.4].iterrows():
        plt.annotate(
            f"{row['Country']}, {row['Date']}",
            (row["McDonald_Sentiment_Index"], row["Bond_Yield"]),
        )

    # Set the title and labels
    plt.title("McDonald Sentiment Index vs Bond Yield")
    plt.xlabel("McDonald Sentiment Index")
    plt.ylabel("Bond Yield")

    # Save the plot
    plt.savefig(produces)
