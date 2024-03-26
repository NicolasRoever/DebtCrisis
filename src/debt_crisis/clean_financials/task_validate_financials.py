from debt_crisis.config import BLD, ALL_COUNTRIES_IN_QUARTERLY_MACRO_DATA
from debt_crisis.clean_financials.validate_financials import (
    plot_time_series_variables_from_different_datasources,
    plot_bond_yield_spreads_for_all_countries,
    plot_bond_yield_for_country,
)
import pandas as pd
from pytask import task

task_plot_bond_yields_two_datasources_dependencies = {
    "datasource_1": BLD / "data" / "step_one_regression_dataset_eurostat.pkl",
    "datasource_2": BLD / "data" / "step_one_regression_dataset_quarterly_data.pkl",
}


def task_plot_bond_yields_greece_two_datasources(
    depends_on=task_plot_bond_yields_two_datasources_dependencies,
    produces=BLD / "figures" / "bond_yields_greece_both_datasources.png",
):
    datasource_1 = pd.read_pickle(depends_on["datasource_1"])
    datasource_1_filter = datasource_1[datasource_1["Country"] == "greece"]

    datasource_2 = pd.read_pickle(depends_on["datasource_2"])
    datasource_2_filter = datasource_2[datasource_2["Country"] == "greece"]

    plot = plot_time_series_variables_from_different_datasources(
        datasource_1_filter,
        "Bond_Yield",
        datasource_2_filter,
        "10y_Maturity_Bond_Yield",
    )

    plot.savefig(produces)


def task_plot_all_bond_yields(
    depends_on=BLD / "data" / "event_study_approach" / "event_study_dataset.pkl",
    produces=BLD / "figures" / "all_bond_yields.png",
):
    data = pd.read_pickle(depends_on)

    plot = plot_bond_yield_spreads_for_all_countries(data)

    plot.savefig(produces)


for country in ALL_COUNTRIES_IN_QUARTERLY_MACRO_DATA:

    @task(id=country)
    def task_plot_bond_yield_for_given_country(
        depends_on=BLD / "data" / "event_study_approach" / "event_study_dataset.pkl",
        country=country,
        produces=BLD / "figures" / "raw_bond_yields" / f"bond_yield_{country}.png",
    ):
        data = pd.read_pickle(depends_on)
        plot = plot_bond_yield_for_country(data, country)
        plot.savefig(produces)
