"""All the general configuration of the project."""
from pathlib import Path

import spacy
import random

# -------------Configurations----------------#

EVENT_STUDY_TIME_PERIOD = ["2008Q1", "2014Q1"]

NO_LONG_RUNNING_TASKS = False

CONFIGURATION_SETTINGS = {
    "sentiment_index_calculation_method": "negative_and_positive",  # "negative_and_positive" or "negative"
    "words_in_environment": 20,
    "event_study_time_period": ["2008Q1", "2014Q1"],
}

EVENT_STUDY_MODEL_LIST = [
    "Q('10y_Maturity_Bond_Yield') ~ Q('Public_Debt_as_%_of_GDP')+ GDP_in_Current_Prices_Growth + Moody_Rating_PD + "
    "VIX_Daily_Close_Quarterly_Mean + Q('10y_Maturity_Bond_Yield_US') + C(Country) +",
    "Q('10y_Maturity_Bond_Yield') ~ Q('Public_Debt_as_%_of_GDP')+ GDP_in_Current_Prices_Growth + Moody_Rating_PD + "
    "VIX_Daily_Close_Quarterly_Mean + Q('10y_Maturity_Bond_Yield_US') + C(Country) + C(Date) +",
    "Q('10y_Maturity_Bond_Yield') ~ Q('Public_Debt_as_%_of_GDP')+ GDP_in_Current_Prices_Growth  + "
    "VIX_Daily_Close_Quarterly_Mean + Q('10y_Maturity_Bond_Yield_US') + C(Country) + C(Date) +",
    "Q('10y_Maturity_Bond_Yield') ~ Q('Public_Debt_as_%_of_GDP')+ GDP_in_Current_Prices_Growth + Moody_Rating_PD + "
    "VIX_Daily_Close_Quarterly_Mean + Q('10y_Maturity_Bond_Yield_US') + C(Country) + Q('3_Month_US_Treasury_Yield_Quarterly_Mean') + ",
    "Q('10y_Maturity_Bond_Yield') ~ Q('Public_Debt_as_%_of_GDP')+ GDP_in_Current_Prices_Growth + Moody_Rating_PD + "
    "VIX_Daily_Close_Quarterly_Mean + Q('10y_Maturity_Bond_Yield_US') + C(Country) + Q('NASDAQ_Daily_Close_Quarterly_Mean') + ",
    "Q('10y_Maturity_Bond_Yield') ~ Q('Public_Debt_as_%_of_GDP')+ GDP_in_Current_Prices_Growth + Moody_Rating_PD + "
    "VIX_Daily_Close_Quarterly_Mean + Q('10y_Maturity_Bond_Yield_US') + C(Country) + Q('Current_Account_in_USD') + ",
    "Q('10y_Maturity_Bond_Yield') ~ Q('Public_Debt_as_%_of_GDP')+ GDP_in_Current_Prices_Growth + Moody_Rating_PD + "
    "VIX_Daily_Close_Quarterly_Mean + Q('10y_Maturity_Bond_Yield_US') + C(Country) + Q('Eurostat_CPI_Annualised Growth_Rate') + ",
]

random.seed(42)


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

MAPPING_MOODY_RATING_TO_PD = {
    "Aaa": 0.02,
    "Aa": 0.03,
    "A": 0.07,
    "Baa": 0.18,
    "Ba": 0.7,
    "B": 2.0,
    "Caa": 14.0,
    "Ca": 17.0,
    "C": 20.0,
    "D": ">20.0",
}


MAPPING_COUNTRY_NAMES_TO_COUNTRY = {
    "austrian": "austria",
    "belgian": "belgium",
    "bulgarian": "bulgaria",
    "croatian": "croatia",
    "cypriot": "cyprus",
    "czech": "czechia",
    "danish": "denmark",
    "estonian": "estonia",
    "finnish": "finland",
    "french": "france",
    "german": "germany",
    "greek": "greece",
    "hungarian": "hungary",
    "irish": "ireland",
    "italian": "italy",
    "latvian": "latvia",
    "lithuanian": "lithuania",
    "luxembourgish": "luxembourg",
    "maltese": "malta",
    "dutch": "netherlands",
    "polish": "poland",
    "portuguese": "portugal",
    "romanian": "romania",
    "slovak": "slovakia",
    "slovenian": "slovenia",
    "spanish": "spain",
    "swedish": "sweden",
    "austrians": "austria",
    "belgians": "belgium",
    "bulgarians": "bulgaria",
    "croatians": "croatia",
    "cypriots": "cyprus",
    "czechs": "czechia",
    "danes": "denmark",
    "estonians": "estonia",
    "finns": "finland",
    "french": "france",
    "germans": "germany",
    "greeks": "greece",
    "hungarians": "hungary",
    "irish": "ireland",
    "italians": "italy",
    "latvians": "latvia",
    "lithuanians": "lithuania",
    "luxembourgers": "luxembourg",
    "maltese": "malta",
    "dutch": "netherlands",
    "polish": "poland",
    "portuguese": "portugal",
    "romanians": "romania",
    "slovaks": "slovakia",
    "slovenians": "slovenia",
    "spaniards": "spain",
    "swedes": "sweden",
    "vienna": "austria",
    "sofia": "bulgaria",
    "zagreb": "croatia",
    "nicosia": "cyprus",
    "prague": "czechia",
    "copenhagen": "denmark",
    "tallinn": "estonia",
    "helsinki": "finland",
    "paris": "france",
    "berlin": "germany",
    "athens": "greece",
    "budapest": "hungary",
    "dublin": "ireland",
    "rome": "italy",
    "riga": "latvia",
    "vilnius": "lithuania",
    "luxembourg": "luxembourg",
    "valetta": "malta",
    "amsterdam": "netherlands",
    "warsaw": "poland",
    "lisbon": "portugal",
    "bucharest": "romania",
    "bratislava": "slovakia",
    "ljubljana": "slovenia",
    "madrid": "spain",
    "stockholm": "sweden",
    "croats": "croatia",
    "luxembourger": "luxembourg",
    "holland": "netherlands",
    "slovene": "slovenia",
    "austria": "austria",
    "belgium": "belgium",
    "bulgaria": "bulgaria",
    "croatia": "croatia",
    "cyprus": "cyprus",
    "czechia": "czechia",
    "denmark": "denmark",
    "estonia": "estonia",
    "finland": "finland",
    "france": "france",
    "germany": "germany",
    "greece": "greece",
    "hungary": "hungary",
    "ireland": "ireland",
    "italy": "italy",
    "latvia": "latvia",
    "lithuania": "lithuania",
    "luxembourg": "luxembourg",
    "malta": "malta",
    "netherlands": "netherlands",
    "poland": "poland",
    "portugal": "portugal",
    "romania": "romania",
    "slovakia": "slovakia",
    "slovenia": "slovenia",
    "spain": "spain",
    "sweden": "sweden",
}


FRED_API_KEY = "a6c090d9708fcd388e74204168cc7f43"


FRED_INTEREST_RATE_SERIES = {"Japan": "IRLTLT01JPM156N", "Germany": "IRLTLT01DEM156N"}


OECD_QUARTERLY_GDP_XML_QUERY_LINK = "https://sdmx.oecd.org/public/rest/data/OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_USD,1.0/Q...S1..B1GQ.....V..?dimensionAtObservation=AllDimensions"


OECD_QUARTERLY_DEBT_AS_PERCENT_GDP_QUERY_LINK = "https://sdmx.oecd.org/public/rest/data/OECD.SDD.NAD,DSD_NASEC20@DF_T7PSD_Q,1.0/Q....S13.....FD4.T.PT_B1GQ......?dimensionAtObservation=AllDimensions"

OECD_QUARTERLY_CURRENT_ACCOUNT_QUERY_LINK = "https://sdmx.oecd.org/public/rest/data/OECD.SDD.TPS,DSD_BOP@DF_BOP,1.0/..CA.B..Q.USD_EXC.N?dimensionAtObservation=AllDimensions"

LINK_RATINGS = "http://ratingshistory.info"

LINK_RATINGS_2 = "https://www.wikirating.com/data-analytics/"

LINK_RATINGS_3 = "https://github.com/gvschweinitz/ES_18_The-Joint-Dynamics-of-Sovereign-Ratings-and-Government-Bond-Yields/tree/main"


MOODY_RATING_CONVERSION = {
    "Aaa": 21,
    "Aa1": 20,
    "Aa2": 19,
    "Aa3": 18,
    "A1": 17,
    "A2": 16,
    "A3": 15,
    "Baa1": 14,
    "Baa2": 13,
    "Baa3": 12,
    "Ba1": 11,
    "Ba2": 10,
    "Ba3": 9,
    "B1": 8,
    "B2": 7,
    "B3": 6,
    "Caa1": 5,
    "Caa2": 4,
    "Caa3": 3,
    "Ca": 2,
    "C": 1,
}
