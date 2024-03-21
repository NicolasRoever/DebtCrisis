from debt_crisis.config import BLD
from debt_crisis.clean_financials.validate_financials import (
    plot_time_series_variables_from_different_datasources,
)
import pandas as pd

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
