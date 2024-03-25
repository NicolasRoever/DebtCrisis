import pandas as pd
import pytest

from src.debt_crisis.regression_analysis.regression_analysis import (
    add_dummy_columns_for_quarter_year_combination,
)


def test_add_dummy_columns_for_quarter_year_combination():
    # Test data
    df = pd.DataFrame(
        {
            "Date": pd.date_range(start="2009-01-01", end="2011-12-31", freq="Q"),
            "Country": ["Greece", "Spain"] * 6,
        }
    )

    # Expected output
    expected = df.copy()
    quarters = pd.period_range(start="2009Q1", end="2011Q4", freq="Q")
    for quarter in quarters:
        start_date = quarter.start_time
        end_date = quarter.end_time
        expected[f"Dummy_Greece_{quarter}"] = (
            (expected["Date"] >= start_date)
            & (expected["Date"] <= end_date)
            & (expected["Country"] == "Greece")
        ).astype(int)

    # Call the function and get the result
    result = add_dummy_columns_for_quarter_year_combination(
        df, "2009Q1", "2011Q4", "Greece"
    )

    # Assert that the result equals the expected output
    pd.testing.assert_frame_equal(result, expected)
