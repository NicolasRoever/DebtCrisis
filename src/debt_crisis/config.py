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
]

PATH_TO_TRANSCRIPTS = Path("src/debt_crisis/data/transcripts/raw/Eikon 2002 - 2022/")

NLP_MODEL = spacy.load("en_core_web_sm")


NO_LONG_RUNNING_TASKS = True
