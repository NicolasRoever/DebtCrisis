import pandas as pd
from pytask import task

from debt_crisis.clean_financials.clean_financials import import_quarterly_data
from debt_crisis.config import BLD, QUARTERLY_DATA_PATHS

for index, filepath in enumerate(QUARTERLY_DATA_PATHS):

    @task(id=f"filename_{index}")
    def task_import_qarterly_data(
        depends_on=filepath,
        produces=BLD / "data" / "financial_data" / filepath.stem,
    ):
        if depends_on.suffix == ".csv":
            raw_data = pd.read_csv(depends_on)
        elif depends_on.suffix in [".xls", ".xlsx"]:
            raw_data = pd.read_excel(depends_on)
        else:
            msg = "Unsupported file format. Only CSV and Excel files are supported."
            raise ValueError(
                msg,
            )

        clean_data = import_quarterly_data(raw_data)

        clean_data.to_pickle(produces)
