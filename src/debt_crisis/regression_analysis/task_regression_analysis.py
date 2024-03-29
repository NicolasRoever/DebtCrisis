import pandas as pd
import matplotlib.pyplot as plt
from pytask import task
import numpy as np

from debt_crisis.regression_analysis.regression_analysis import (
    create_dataset_step_one_regression_eurostat_data,
    run_first_step_regression_eurostat,
    run_second_step_regression_eurostat,
    run_third_step_regression_eurostat,
    create_dataset_step_one_regression_quarterly_data,
    run_step_two_regression_quarterly_data,
    run_step_three_regression_quarterly_data,
    create_data_set_event_study,
    run_event_study_regression,
    plot_event_study_coefficients,
    plot_event_study_coefficients_for_multiple_countries_in_one_plot,
    run_exuberance_index_regression_quarterly_data,
    merge_event_study_coefficients_and_exuberance_index_data,
    run_regression_exuberance_indicator_vs_event_study_coefficients_for_all_countries,
    plot_fitted_values_from_exuberance_unfounded_bond_yield_regression,
    run_exuberance_index_regression_event_study_data,
    plot_unfounded_spreads_vs_unfounded_sentiment,
    plot_unfounded_spreads_vs_daily_sentiment_index,
)

from debt_crisis.config import (
    BLD,
    EVENT_STUDY_COUNTRIES,
    EVENT_STUDY_TIME_PERIOD,
    TOP_LEVEL_DIR,
    CONFIGURATION_SETTINGS,
    EVENT_STUDY_PLOT_COUNTRIES,
)

from debt_crisis.utilities import _name_sentiment_index_output_file


# -----------------Event Study-----------------#


task_create_event_study_dataset_dependencies = {
    "quarterly_macro_data": BLD
    / "data"
    / "financial_data"
    / "Quarterly Macroeconomic Variables_cleaned.pkl",
    "sentiment_index_data": BLD
    / "data"
    / _name_sentiment_index_output_file(
        "mcdonald_sentiment_index_cleaned", CONFIGURATION_SETTINGS, ".pkl"
    ),
}


def task_create_event_study_dataset(
    depends_on=task_create_event_study_dataset_dependencies,
    event_study_countries=EVENT_STUDY_COUNTRIES,
    event_study_time_period=EVENT_STUDY_TIME_PERIOD,
    produces=BLD
    / "data"
    / "event_study_approach"
    / _name_sentiment_index_output_file(
        "event_study_dataset", CONFIGURATION_SETTINGS, ".pkl"
    ),
):
    quarterly_macro_data = pd.read_pickle(depends_on["quarterly_macro_data"])
    sentiment_index_data = pd.read_pickle(depends_on["sentiment_index_data"])

    dataset = create_data_set_event_study(
        quarterly_macro_data,
        sentiment_index_data,
        event_study_countries,
        event_study_time_period=event_study_time_period,
    )

    dataset.to_pickle(produces)


def task_run_bond_yield_event_study(
    depends_on=BLD
    / "data"
    / "event_study_approach"
    / _name_sentiment_index_output_file(
        "event_study_dataset", CONFIGURATION_SETTINGS, ".pkl"
    ),
    event_study_countries=EVENT_STUDY_COUNTRIES,
    event_study_time_period=EVENT_STUDY_TIME_PERIOD,
    produces=[
        BLD
        / "models"
        / _name_sentiment_index_output_file(
            "event_study_regression", CONFIGURATION_SETTINGS, ".txt"
        ),
        BLD
        / "data"
        / "event_study_approach"
        / _name_sentiment_index_output_file(
            "event_study_coefficients_data", CONFIGURATION_SETTINGS, ".pkl"
        ),
        BLD
        / "data"
        / "event_study_approach"
        / _name_sentiment_index_output_file(
            "event_study_full_model_data", CONFIGURATION_SETTINGS, ".pkl"
        ),
    ],
):
    data = pd.read_pickle(depends_on)

    model, regression_dataset = run_event_study_regression(
        data, event_study_countries, event_study_time_period
    )

    regression_dataset.to_pickle(produces[2])

    model_summary = model.summary()

    with open(produces[0], "w") as file:
        file.write(model_summary.as_text())

    # Make coefficient data

    coefficient_data = pd.DataFrame()

    coefficient_data["Variable"] = model.params.index
    coefficient_data["Coefficient"] = model.params.values
    coefficient_data["Standard Errors"] = model.bse.values

    coefficient_data.to_pickle(produces[1])


for index, country_list in enumerate(EVENT_STUDY_PLOT_COUNTRIES):
    plot_number = index + 1

    @task(id=str(index))
    def task_plot_event_study_coefficients_for_all_countries_in_one_plot(
        depends_on=BLD
        / "data"
        / "event_study_approach"
        / _name_sentiment_index_output_file(
            "event_study_coefficients_data", CONFIGURATION_SETTINGS, ".pkl"
        ),
        countries=country_list,
        produces=[
            BLD
            / "figures"
            / "event_study"
            / f"event_study_coefficients_{plot_number}.png",
            TOP_LEVEL_DIR
            / "Input_for_Paper"
            / "figures"
            / f"event_study_coefficients_{plot_number}.png",
        ],
    ):
        data = pd.read_pickle(depends_on)

        plot = plot_event_study_coefficients_for_multiple_countries_in_one_plot(
            data, countries
        )

        plot.savefig(produces[0])
        plot.savefig(produces[1])


for country in EVENT_STUDY_COUNTRIES:

    @task(id=country)
    def task_plot_event_study_coefficients(
        depends_on=BLD
        / "data"
        / "event_study_approach"
        / _name_sentiment_index_output_file(
            "event_study_coefficients_data", CONFIGURATION_SETTINGS, ".pkl"
        ),
        country=country,
        produces=BLD
        / "figures"
        / "event_study"
        / _name_sentiment_index_output_file(
            f"event_study_coefficients_{country}", CONFIGURATION_SETTINGS, ".png"
        ),
    ):
        data = pd.read_pickle(depends_on)

        figure = plot_event_study_coefficients(data, country)

        figure.savefig(produces)


# -------------------Sentiment Indicator-------------------#


def task_run_exuberance_index_regression_with_event_study_data(
    depends_on=BLD
    / "data"
    / "event_study_approach"
    / _name_sentiment_index_output_file(
        "event_study_dataset", CONFIGURATION_SETTINGS, ".pkl"
    ),
    produces=[
        BLD
        / "models"
        / _name_sentiment_index_output_file(
            "exuberance_index_regression_quarterly", CONFIGURATION_SETTINGS, ".txt"
        ),
        BLD
        / "data"
        / "sentiment_exuberance"
        / _name_sentiment_index_output_file(
            "exuberance_index_regression_event_study_data",
            CONFIGURATION_SETTINGS,
            ".pkl",
        ),
        TOP_LEVEL_DIR
        / "Input_for_Paper"
        / "tables"
        / "exuberance_index_regression_summary.tex",
    ],
):
    # Load the data
    data = pd.read_pickle(depends_on).dropna()

    # Run the regression
    model, regression_table_latex = run_exuberance_index_regression_event_study_data(
        data
    )

    with open(produces[2], "w") as file:
        file.write(regression_table_latex.as_latex())

    # Get the summary of the model
    model_summary = model.summary()

    # Save the summary as a text file
    with open(produces[0], "w") as file:
        file.write(model_summary.as_text())

    data["Fitted_Values_Exuberance_Regression"] = model.fittedvalues
    data["Residuals_Exuberance_Regression"] = model.resid

    data.to_pickle(produces[1])


def task_run_regression_event_study_coefficients_vs_exuberance_index(
    depends_on=[
        BLD
        / "data"
        / "event_study_approach"
        / _name_sentiment_index_output_file(
            "event_study_coefficients_data", CONFIGURATION_SETTINGS, ".pkl"
        ),
        BLD
        / "data"
        / "sentiment_exuberance"
        / _name_sentiment_index_output_file(
            "exuberance_index_regression_event_study_data",
            CONFIGURATION_SETTINGS,
            ".pkl",
        ),
    ],
    produces=[
        BLD / "models" / "event_study_coefficients_vs_exuberance_index_regression.csv",
        TOP_LEVEL_DIR
        / "Input_for_Paper"
        / "tables"
        / "exuberance_vs_event_study_coefficients_regression_summary.tex",
        BLD
        / "data"
        / "exuberance_vs_event_study_coefficients_regression_fitted_values.pkl",
    ],
):
    coefficients_data = pd.read_pickle(depends_on[0])
    exuberance_data = pd.read_pickle(depends_on[1])

    merged_data = merge_event_study_coefficients_and_exuberance_index_data(
        coefficients_data, exuberance_data
    )

    (
        result,
        fitted_values,
    ) = run_regression_exuberance_indicator_vs_event_study_coefficients_for_all_countries(
        merged_data
    )

    result["Correlation"] = np.sqrt(result["R_Squared"])

    result.to_csv(produces[0])

    with open(produces[1], "w", encoding="utf-8") as file:
        file.write(result.to_latex())

    fitted_values.to_pickle(produces[2])


def task_plot_fitted_values_from_exuberance_unfounded_bond_yield_regression(
    depends_on=BLD
    / "data"
    / "exuberance_vs_event_study_coefficients_regression_fitted_values.pkl",
    produces=[
        BLD
        / "figures"
        / "fitted_values_exuberance_unfounded_bond_yield_regression.png",
        TOP_LEVEL_DIR
        / "Input_for_Paper"
        / "figures"
        / "fitted_values_exuberance_unfounded_bond_yield_regression.png",
    ],
):
    data = pd.read_pickle(depends_on)

    plot = plot_fitted_values_from_exuberance_unfounded_bond_yield_regression(
        data, ["greece", "portugal", "spain", "italy"]
    )

    plot.savefig(produces[0])
    plot.savefig(produces[1])


for country in EVENT_STUDY_COUNTRIES:
    country = country

    @task(id=country)
    def task_plot_unfounded_spreads_vs_unfounded_sentiment(
        depends_on=[
            BLD
            / "data"
            / "event_study_approach"
            / _name_sentiment_index_output_file(
                "event_study_coefficients_data", CONFIGURATION_SETTINGS, ".pkl"
            ),
            BLD
            / "data"
            / "sentiment_exuberance"
            / _name_sentiment_index_output_file(
                "exuberance_index_regression_event_study_data",
                CONFIGURATION_SETTINGS,
                ".pkl",
            ),
        ],
        country=country,
        produces=BLD
        / "figures"
        / "final regression"
        / f"unfounded_spreads_vs_unfounded_sentiment_{country}.png",
    ):
        coefficients_data = pd.read_pickle(depends_on[0])
        exuberance_data = pd.read_pickle(depends_on[1])

        plot = plot_unfounded_spreads_vs_unfounded_sentiment(
            coefficients_data, exuberance_data, country
        )

        plot.savefig(produces)


for country in EVENT_STUDY_COUNTRIES:
    country = country

    @task(id=country)
    def task_plot_unfounded_spreads_vs_daily_sentiment(
        depends_on=[
            BLD
            / "data"
            / "event_study_approach"
            / _name_sentiment_index_output_file(
                "event_study_coefficients_data", CONFIGURATION_SETTINGS, ".pkl"
            ),
            BLD
            / "data"
            / _name_sentiment_index_output_file(
                "mcdonald_sentiment_index_cleaned", CONFIGURATION_SETTINGS, ".pkl"
            ),
        ],
        country=country,
        produces=BLD
        / "figures"
        / "final regression"
        / f"unfounded_spreads_vs_daily_sentiment_{country}.png",
    ):
        coefficients_data = pd.read_pickle(depends_on[0])
        sentiment_data = pd.read_pickle(depends_on[1])

        plot = plot_unfounded_spreads_vs_daily_sentiment_index(
            coefficients_data, sentiment_data, country
        )

        plot.savefig(produces)


# ------------------------------------------------------------------------------------------
# Quarterly Macroeconomic Data Implementation


dependencies_task_create_dataset_step_one_regression_quarterly_macro_data = {
    "quarterly_macro_data": BLD
    / "data"
    / "financial_data"
    / "Quarterly Macroeconomic Variables_cleaned.pkl",
    "sentiment_index_data": BLD / "data" / "mcdonald_sentiment_index_cleaned.pkl",
}


def task_create_dataset_step_one_regression_quarterly_macro_data(
    depends_on=dependencies_task_create_dataset_step_one_regression_quarterly_macro_data,
    produces=BLD / "data" / "step_one_regression_dataset_quarterly_data.pkl",
):
    quarterly_macro_data = pd.read_pickle(depends_on["quarterly_macro_data"])

    sentiment_index_data = pd.read_pickle(depends_on["sentiment_index_data"])

    dataset = create_dataset_step_one_regression_quarterly_data(
        quarterly_macro_data, sentiment_index_data
    )

    dataset.to_pickle(produces)


def task_run_second_step_regression_quarterly(
    depends_on=BLD / "data" / "step_one_regression_dataset_output_quarterly.pkl",
    produces=[
        BLD / "models" / "second_step_regression_quarterly.txt",
        BLD / "data" / "step_two_regression_dataset_output_quarterly.pkl",
    ],
):
    data = pd.read_pickle(depends_on)

    model = run_step_two_regression_quarterly_data(data)

    model_summary = model.summary()

    with open(produces[0], "w") as file:
        file.write(model_summary.as_text())

    data["Residuals_Step_Two_Regression"] = model.resid
    data["Fitted_Values_Step_Two_Regression"] = model.fittedvalues

    data.to_pickle(produces[1])


def task_run_step_three_regression_quarterly(
    depends_on=BLD / "data" / "step_two_regression_dataset_output_quarterly.pkl",
    produces=BLD / "models" / "third_step_regression_quarterly.txt",
):
    data = pd.read_pickle(depends_on)

    model = run_step_three_regression_quarterly_data(data)

    model_summary = model.summary()

    with open(produces, "w") as file:
        file.write(model_summary.as_text())


def task_plot_fitted_values_vs_real_yield(
    depends_on=BLD / "data" / "step_two_regression_dataset_output_quarterly.pkl",
    produces=BLD / "figures" / "fitted_values_vs_real_yield_greece.png",
):
    data = pd.read_pickle(depends_on)
    data_filter = data[data["Country"] == "greece"]

    fig, ax = plt.subplots()
    ax.plot(
        data_filter["Date"],
        data_filter["Fitted_Values_Step_Two_Regression"],
        label="Fitted Values",
    )
    ax.plot(
        data_filter["Date"], data_filter["10y_Maturity_Bond_Yield"], label="Real Yield"
    )
    ax.plot(
        data_filter["Date"], data_filter["Public_Debt_as_%_of_GDP"], label="Residuals"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title("Fitted Values vs Real Yield for Greece")
    ax.legend()

    fig.savefig(produces)


# ------------------------------------------------------------------------------------------
# Eurostat Data Implementation


dependencies_task_create_dataset_step_one_regression_eurostat_data = {
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
    depends_on=dependencies_task_create_dataset_step_one_regression_eurostat_data,
    produces=BLD / "data" / "step_one_regression_dataset_eurostat.pkl",
):
    bond_yield_data = pd.read_pickle(depends_on["bond_yield_data"])

    debt_to_gdp_data = pd.read_pickle(depends_on["debt_to_gdp_data"])

    sentiment_index_data = pd.read_pickle(depends_on["sentiment_index_data"])

    ratings_data = pd.read_pickle(depends_on["ratings_data"])

    gdp_data = pd.read_pickle(depends_on["gdp_data"])

    current_account_data = pd.read_pickle(depends_on["current_account_data"])

    stoxx_data = pd.read_pickle(depends_on["stoxx_data"])

    dataset = create_dataset_step_one_regression_eurostat_data(
        bond_yield_data,
        debt_to_gdp_data,
        sentiment_index_data,
        ratings_data,
        gdp_data,
        current_account_data,
        stoxx_data,
    )

    dataset.to_pickle(produces)


def task_run_first_step_regression_eurostat(
    depends_on=BLD / "data" / "step_one_regression_dataset_eurostat.pkl",
    produces=[
        BLD / "models" / "first_step_regression_eurostat.txt",
        BLD / "data" / "step_one_regression_dataset_output_eurostat.pkl",
    ],
):
    # Load the data
    data = pd.read_pickle(depends_on)

    # Run the regression
    model = run_first_step_regression_eurostat(data)

    # Get the summary of the model
    model_summary = model.summary()

    # Save the summary as a text file
    with open(produces[0], "w") as file:
        file.write(model_summary.as_text())

    data["Fitted_Values_Step_One_Regression"] = model.fittedvalues
    data["Residuals_Step_One_Regression"] = model.resid

    data.to_pickle(produces[1])


def task_run_second_step_regression_eurostat(
    depends_on=BLD / "data" / "step_one_regression_dataset_output_eurostat.pkl",
    produces=[
        BLD / "models" / "second_step_regression_eurostat.txt",
        BLD / "data" / "step_two_regression_dataset_output_eurostat.pkl",
    ],
):
    data = pd.read_pickle(depends_on)

    model = run_second_step_regression_eurostat(data)

    model_summary = model.summary()

    with open(produces[0], "w") as file:
        file.write(model_summary.as_text())

    data["Residuals_Step_Two_Regression"] = model.resid

    data.to_pickle(produces[1])


def task_run_third_step_regression(
    depends_on=BLD / "data" / "step_two_regression_dataset_output_eurostat.pkl",
    produces=BLD / "models" / "third_step_regression.txt",
):
    data = pd.read_pickle(depends_on)

    model = run_third_step_regression_eurostat(data)

    model_summary = model.summary()

    with open(produces, "w") as file:
        file.write(model_summary.as_text())


def task_plot_sentiment_index_vs_bond_yield(
    depends_on=BLD / "data" / "step_one_regression_dataset_eurostat.pkl",
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
