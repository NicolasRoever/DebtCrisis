from debt_crisis.event_study_sentiment_index.event_study_sentiment_index import (
    estimate_normal_returns,
    generate_event_study_output,
    run_event_study_single_event,
    plot_CAR_over_time_with_ci_plotly,
)

from debt_crisis.config import BLD, CONFIGURATION_SETTINGS

from debt_crisis.utilities import _name_sentiment_index_output_file

import pandas as pd
from pytask import task


task_run_event_study_on_daily_sentiment_index_dependencies = {
    "sentiment_data_path": BLD
    / "data"
    / "event_study_sentiment_index"
    / _name_sentiment_index_output_file(
        "mcdonald_sentiment_index_cleaned", CONFIGURATION_SETTINGS, ".pkl"
    )
}


for country in ["portugal", "italy", "spain", "greece", "ireland"]:
    country = country

    event_date = (pd.to_datetime("2010-10-19"),)

    @task(id=country)
    def task_run_event_study_on_daily_sentiment_index(
        depends_on=task_run_event_study_on_daily_sentiment_index_dependencies,
        event_date=event_date,
        country=country,
        produces=BLD
        / "figures"
        / _name_sentiment_index_output_file(
            f"figure_event_study_{country}_{event_date}", CONFIGURATION_SETTINGS, ".png"
        ),
    ):
        sentiment_data = pd.read_pickle(depends_on["sentiment_data_path"])

        event_study_output = run_event_study_single_event(
            sentiment_data, country, event_date
        )

        figure = plot_CAR_over_time_with_ci_plotly(
            event_study_output, sentiment_data, country, event_date
        )

        figure.savefig(produces)
