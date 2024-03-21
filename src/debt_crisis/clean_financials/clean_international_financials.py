from debt_crisis.config import FRED_API_KEY

from fredapi import Fred
import pandas as pd
import requests
import xmltodict


def clean_quarterly_macroeconomic_variables(raw_data):
    """THis function cleans the quarterly macroeconomic variables dataset."""

    cleaned_data = pd.DataFrame()

    cleaned_data["date"] = pd.to_datetime(raw_data["Date"])
    cleaned_data["Country"] = raw_data["Country"].str.lower()

    for column in raw_data.columns:
        if column not in cleaned_data.columns:
            cleaned_data[column] = raw_data[column]

    return cleaned_data
