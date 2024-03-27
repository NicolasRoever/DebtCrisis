import pandas as pd
import pytest
import numpy as np

from src.debt_crisis.clean_financials.clean_financials import (
    import_quarterly_data,
    clean_bond_yield_spreads,
    clean_one_rating_file,
    generate_long_format_rating_data,
    merge_alphabetic_with_numeric_rating_data,
    clean_gdp_data,
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


def test_clean_one_rating_file():
    # Create a sample DataFrame
    test_input = pd.DataFrame(
        {
            "Date": ["01.10.02", "13.12.05", None, None, "13.02.15", "24.06.16"],
            "Issuer Rating": [
                "AAA",
                "WD",
                "Access to S&P data denied",
                "Access to S&P data denied",
                "AA+",
                "Aa1",
            ],
            "Rating Source": ["FIS", "FIS", None, None, "FDL", "MIS"],
        }
    )

    # Create the expected output DataFrame
    expected_output = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["01.10.02", "13.02.15", "24.06.16"], format="mixed"
            ),
            "rating": ["AAA", "AA+", "Aa1"],
            "source": ["FIS", "FDL", "MIS"],
            "Country": ["austria", "austria", "austria"],
        }
    )
    expected_output.set_index("Date", inplace=True)

    # Call the function with the sample DataFrame and a sample filename
    actual_output = clean_one_rating_file(test_input, "austria")

    # Check that the function correctly cleaned the data
    pd.testing.assert_frame_equal(actual_output, expected_output)


def test_generate_long_format_rating_data():
    test_imported_data = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["01.10.10", "13.02.15", "2016-06-24"], format="mixed"
            ),
            "Rating": ["AAA", "AA+", "Aa1"],
            "Country": ["austria", "austria", "sweden"],
        }
    )

    # Create output dataframe
    dates = pd.date_range(start="2002-01-01", end="2023-01-01")

    df_austria = pd.DataFrame(dates, columns=["Date"])
    df_austria["Country"] = "austria"
    df_austria["Rating_Letter_Moody"] = np.nan
    df_austria.loc[
        df_austria["Date"] >= pd.to_datetime("2010-01-10"), "Rating_Letter_Moody"
    ] = "AAA"
    df_austria.loc[
        df_austria["Date"] >= pd.to_datetime("2015-02-13"), "Rating_Letter_Moody"
    ] = "AA+"

    df_sweden = pd.DataFrame(dates, columns=["Date"])
    df_sweden["Country"] = "sweden"
    df_sweden["Rating_Letter_Moody"] = np.nan
    df_sweden.loc[
        df_sweden["Date"] >= pd.to_datetime("2016-06-24"), "Rating_Letter_Moody"
    ] = "Aa1"

    expected_output = pd.concat([df_austria, df_sweden], ignore_index=True)

    actual_output = generate_long_format_rating_data(test_imported_data)

    pd.testing.assert_frame_equal(actual_output, expected_output)


def test_merge_alphabetic_with_numeric_rating_data():
    long_format_rating_data = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["01.10.10", "13.02.15", "2016-06-24"], format="mixed"
            ),
            "Country": ["austria", "austria", "sweden"],
            "Rating_Letter_MIS": ["Aaa", "Aaa", "Aa1"],
            "Rating_Letter_FIS": ["AAA", "AAA", "AAA"],
            "Rating_Letter_FDL": ["A-", "A-", "A-"],
        }
    )

    rating_conversion_data = pd.DataFrame(
        {
            "source": ["MIS", "FIS", "FDL", "MIS"],
            "rating": ["Aaa", "AAA", "A-", "Aa1"],
            "score": [21, 22, 23, 20],
        }
    )

    expected_output = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["01.10.10", "13.02.15", "2016-06-24"], format="mixed"
            ),
            "Country": ["austria", "austria", "sweden"],
            "Rating_Letter_MIS": ["Aaa", "Aaa", "Aa1"],
            "Rating_Letter_FIS": ["AAA", "AAA", "AAA"],
            "Rating_Letter_FDL": ["A-", "A-", "A-"],
            "Rating_Numeric_MIS": [21, 21, 20],
            "Rating_Numeric_FIS": [22, 22, 22],
            "Rating_Numeric_FDL": [23, 23, 23],
        }
    )

    actual_output = merge_alphabetic_with_numeric_rating_data(
        long_format_rating_data, rating_conversion_data
    )

    pd.testing.assert_frame_equal(actual_output, expected_output, check_dtype=False)


def test_clean_gdp_data():
    gdp_data = pd.DataFrame(
        {
            "Date": ["2020-03-31", "2020-06-30", "2020-09-30"],
            "Austria": [1, 2, 3],
            "Belgium": [2, 3, 4],
        }
    )

    expected_output = pd.DataFrame(
        {
            "Date": ["2020-03-31", "2020-06-30", "2020-09-30"] * 2,
            "Country": [
                "austria",
                "austria",
                "austria",
                "belgium",
                "belgium",
                "belgium",
            ],
            "GDP": [1, 2, 3, 2, 3, 4],
            "GDP_Growth": [np.nan, 1.0, 0.5, np.nan, 0.5, 0.3333333333333333],
            "GDP_Growth_Lead": [1.0, 0.5, np.nan, 0.5, 0.3333333333333333, np.nan],
        }
    )

    actual_output = clean_gdp_data(gdp_data)

    pd.testing.assert_frame_equal(actual_output, expected_output)
