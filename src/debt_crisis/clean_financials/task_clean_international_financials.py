from debt_crisis.config import (
    FRED_API_KEY,
    FRED_INTEREST_RATE_SERIES,
    BLD,
    SRC,
    OECD_QUARTERLY_GDP_XML_QUERY_LINK,
    OECD_QUARTERLY_CURRENT_ACCOUNT_QUERY_LINK,
)
from debt_crisis.clean_financials.clean_international_financials import (
    clean_quarterly_macroeconomic_variables,
)

import pandas as pd


def task_import_quarterly_macroeconomic_variables(
    depends_on=SRC
    / "data"
    / "financial_data"
    / "Quarterly Macroeconomic Variables.xlsx",
    produces=BLD
    / "data"
    / "financial_data"
    / "Quarterly Macroeconomic Variables_cleaned.pkl",
):
    raw_data = pd.read_excel(depends_on)
    clean_data = clean_quarterly_macroeconomic_variables(raw_data)
    clean_data.to_pickle(produces)
