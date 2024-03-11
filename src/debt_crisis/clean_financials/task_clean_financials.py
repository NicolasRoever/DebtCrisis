from src.debt_crisis.clean_financials.clean_financials import import_quarterly_data

from src.debt_crisis.config import BLD, SRC, QUARTERLY_DATA_PATHS

from pathlib import Path

from pytask import task

import pandas as pd 

for index, filepath in enumerate(QUARTERLY_DATA_PATHS):
    @task(id=f"filename_{index}")
    def task_import_qarterly_data(depends_on=filepath,
                                produces=BLD / "data" / "financial_data" / filepath.stem):
        

        if depends_on.suffix == '.csv':

            raw_data = pd.read_csv(depends_on)
        elif depends_on.suffix in ['.xls', '.xlsx']:

            raw_data = pd.read_excel(depends_on)
        else:
            raise ValueError("Unsupported file format. Only CSV and Excel files are supported.")


        clean_data = import_quarterly_data(raw_data)

    
        clean_data.to_pickle(produces)

