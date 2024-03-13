import pandas as pd
import pytest

from src.debt_crisis.clean_financials.clean_financials import (
    import_quarterly_data,
    clean_bond_yield_spreads,
)


def test_import_quarterly_data():
    # Create sample data
    raw_data_df = pd.DataFrame(
        {
            "Date": ["2022-01-01", "2022-04-01", "2022-07-01", "2022-10-01"],
            "Denmark": [100, 200, 300, 400],
            "Germany": [10, 20, 30, 40],
        }
    )

    # Call the function
    actual_result = import_quarterly_data(raw_data_df)

    # Define the expected result DataFrame
    expected_result = pd.DataFrame(
        {"Denmark": [100.0, 200.0, 300.0, 400.0], "Germany": [10, 20, 30, 40]},
        index=pd.to_datetime(["2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31"]),
    )

    expected_result.index.name = "Date"

    # Assert that the actual result matches the expected result
    pd.testing.assert_frame_equal(actual_result, expected_result, check_dtype=False)


def test_clean_bond_yield_spreads():
    # Create a sample dataframe
    data = {
        "date": ["2002-03-31", "2002-04-30"],
        "germany": [5.253, 5.300],
        "italy": [5.503, 5.600],
        "france": [5.331, 5.400],
        "spain": [5.417, 5.500],
    }
    df = pd.DataFrame(data)

    # Convert 'date' column to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Call the function
    cleaned_data = clean_bond_yield_spreads(df)

    # Create the expected output
    expected_data = {
        "date": [
            "2002-03-31",
            "2002-04-30",
            "2002-03-31",
            "2002-04-30",
            "2002-03-31",
            "2002-04-30",
            "2002-03-31",
            "2002-04-30",
        ],
        "country": [
            "germany",
            "germany",
            "italy",
            "italy",
            "france",
            "france",
            "spain",
            "spain",
        ],
        "bond_yield": [5.253, 5.300, 5.503, 5.600, 5.331, 5.400, 5.417, 5.500],
    }
    expected_df = pd.DataFrame(expected_data)
    expected_df["date"] = pd.to_datetime(expected_df["date"])

    # Check that the cleaned data is as expected
    pd.testing.assert_frame_equal(cleaned_data, expected_df)
