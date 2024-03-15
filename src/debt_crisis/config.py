"""All the general configuration of the project."""
from pathlib import Path

import spacy

SRC = Path(__file__).parent.resolve()
BLD = SRC.joinpath("..", "..", "bld").resolve()

TEST_DIR = SRC.joinpath("..", "..", "tests").resolve()
PAPER_DIR = SRC.joinpath("..", "..", "paper").resolve()


__all__ = ["BLD", "SRC", "TEST_DIR", "GROUPS"]

QUARTERLY_DATA_PATHS = [
    SRC.joinpath(Path("data/financial_data/debt-to-gdp_EU.csv")),
    SRC.joinpath(Path("data/financial_data/gdp_EU.csv")),
    SRC.joinpath(Path("data/financial_data/stoxx50_vstoxx.xlsx")),
    SRC.joinpath(Path("data/financial_data/eu_yields_10y.xlsx")),
    SRC.joinpath(Path("data/financial_data/current account_EU.csv")),
]

PATH_TO_TRANSCRIPTS = Path("src/debt_crisis/data/transcripts/raw/Eikon 2002 - 2022/")

RATINGS_DATA_PATH = SRC / "data" / "financial_data" / "ratings"

COUNTRIES_UNDER_STUDY = [
    "austria",
    "belgium",
    "bulgaria",
    "croatia",
    "cyprus",
    "czechia",
    "denmark",
    "estonia",
    "finland",
    "france",
    "germany",
    "greece",
    "hungary",
    "ireland",
    "italy",
    "latvia",
    "lithuania",
    "luxembourg",
    "malta",
    "netherlands",
    "poland",
    "portugal",
    "romania",
    "slovakia",
    "slovenia",
    "spain",
    "sweden",
]

NLP_MODEL = spacy.load("en_core_web_sm")

VARIABLES_IN_GLOBAL_FISCAL_RAW_DATA = [
    "ggdy",
    "pby",
    "cby",
    "fby",
    "dfggd",
    "dffb",
    "ggdma",
    "fbma",
    "fxsovsh",
    "secnres",
    "fordebtsh",
    "concggd",
    "avglife",
    "debtduey",
    "xtdebty",
    "fxdebtall",
    "prdebty",
    "pscy",
    "stdebtall",
    "stdebtres",
    "xtdebtres",
    "xtdebtrxg",
    "sovrate",
]

GLOBAL_FISCAL_RAW_DATA_PATH = SRC / "data" / "financial_data" / "Fiscal-space-data.xlsx"

MAPPING_CURRENCY_TO_COUNTRY = {
    "AUD": "Australia",
    "CAD": "Canada",
    "CHF": "Switzerland",
    "DKK": "Denmark",
    "EUR": "Germany",
    "GBP": "United Kingdom",
    "JPY": "Japan",
    "NOK": "Norway",
    "NZD": "New Zealand",
    "SEK": "Sweden",
    "BRL": "Brazil",
    "CLP": "Chile",
    "CNY": "China",
    "COP": "Colombia",
    "HUF": "Hungary",
    "IDR": "Indonesia",
    "ILS": "Israel",
    "INR": "India",
    "KRW": "Korea",
    "MXN": "Mexico",
    "MYR": "Malaysia",
    "PEN": "Peru",
    "PHP": "Philippines",
    "PLN": "Poland",
    "RUB": "Russia",
    "THB": "Thailand",
    "TRY": "Turkey",
    "ZAR": "South Africa",
}


NO_LONG_RUNNING_TASKS = True
