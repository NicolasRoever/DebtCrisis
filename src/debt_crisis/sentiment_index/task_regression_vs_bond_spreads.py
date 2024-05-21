from debt_crisis.sentiment_index.regression_vs_bond_spreads import (
    run_exuberance_index_regression_event_study_data,
)


from debt_crisis.utilities import _name_sentiment_index_output_file

from debt_crisis.config import CONFIGURATION_SETTINGS, TOP_LEVEL_DIR, BLD


import pandas as pd


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
            "exuberance_index_regression_event_study", CONFIGURATION_SETTINGS, ".txt"
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
        file.write(regression_table_latex.render_latex())

    # Get the summary of the model
    model_summary = model.summary()

    # Save the summary as a text file
    with open(produces[0], "w") as file:
        file.write(model_summary.as_text())

    data["Fitted_Values_Exuberance_Regression"] = model.fittedvalues
    data["Residuals_Exuberance_Regression"] = model.resid

    data.to_pickle(produces[1])
