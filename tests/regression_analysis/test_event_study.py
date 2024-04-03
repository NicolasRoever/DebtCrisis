import pytest
from debt_crisis.regression_analysis.event_study import (
    extract_column_names_from_regression_formula,
)


def test_extract_column_names_from_regression_formula():
    # Define a test formula
    formula = "Q('y') ~ Q('x1') + C(x2) + x3 + Q('x4') + C(x5)"

    # Call the function with the test formula
    actual_result = extract_column_names_from_regression_formula(formula)

    # Define the expected result
    expected_result = ["x1", "x2", "x3", "x4", "x5"]

    # Check if the result matches the expected result
    assert actual_result == expected_result
