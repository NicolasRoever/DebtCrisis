"""All the general configuration of the project."""
from pathlib import Path

import spacy

# -------------Configurations----------------#

EVENT_STUDY_TIME_PERIOD = ["2008Q1", "2014Q1"]

NO_LONG_RUNNING_TASKS = True

CONFIGURATION_SETTINGS = {
    "sentiment_index_calculation_method": "negative_and_positive",  # "negative_and_positive" or "negative"
    "words_in_environment": 20,
}


# -------------Paths----------------#

SRC = Path(__file__).parent.resolve()
BLD = SRC.joinpath("..", "..", "bld").resolve()
TOP_LEVEL_DIR = SRC.joinpath("..", "..").resolve()

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

ALL_COUNTRIES_IN_QUARTERLY_MACRO_DATA = [
    "colombia",
    "costa rica",
    "netherlands",
    "chile",
    "latvia",
    "united kingdom",
    "denmark",
    "luxembourg",
    "austria",
    "new zealand",
    "australia",
    "israel",
    "italy",
    "finland",
    "norway",
    "slovenia",
    "lithuania",
    "india",
    "slovak republic",
    "greece",
    "hungary",
    "portugal",
    "iceland",
    "spain",
    "germany",
    "switzerland",
    "japan",
    "belgium",
    "sweden",
    "canada",
    "ireland",
    "usa",
    "mexico",
    "france",
    "russia",
    "czech republic",
    "south korea",
    "south africa",
    "poland",
]

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

EUROPEAN_COUNTRIES_IN_QUARTERLY_MACROECONOMIC_VARIABLES = [
    "austria",
    "belgium",
    "denmark",
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
    "netherlands",
    "poland",
    "portugal",
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

EVENT_STUDY_COUNTRIES = [
    "netherlands",
    "austria",
    "italy",
    "finland",
    "greece",
    "portugal",
    "spain",
    "germany",
    "belgium",
    "ireland",
    "france",
]

EVENT_STUDY_PLOT_COUNTRIES = [
    [
        "netherlands",
        "austria",
        "france",
        "finland",
        "germany",
        "belgium",
    ],
    ["greece", "portugal", "spain", "italy", "ireland"],
]


FRED_API_KEY = "a6c090d9708fcd388e74204168cc7f43"


FRED_INTEREST_RATE_SERIES = {"Japan": "IRLTLT01JPM156N", "Germany": "IRLTLT01DEM156N"}


OECD_QUARTERLY_GDP_XML_QUERY_LINK = "https://sdmx.oecd.org/public/rest/data/OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_USD,1.0/Q...S1..B1GQ.....V..?dimensionAtObservation=AllDimensions"


OECD_QUARTERLY_DEBT_AS_PERCENT_GDP_QUERY_LINK = "https://sdmx.oecd.org/public/rest/data/OECD.SDD.NAD,DSD_NASEC20@DF_T7PSD_Q,1.0/Q....S13.....FD4.T.PT_B1GQ......?dimensionAtObservation=AllDimensions"

OECD_QUARTERLY_CURRENT_ACCOUNT_QUERY_LINK = "https://sdmx.oecd.org/public/rest/data/OECD.SDD.TPS,DSD_BOP@DF_BOP,1.0/..CA.B..Q.USD_EXC.N?dimensionAtObservation=AllDimensions"
