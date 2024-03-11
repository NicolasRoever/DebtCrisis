import pandas as pd

from src.debt_crisis.clean_financials.clean_financials import import_quarterly_data  



def test_import_quarterly_data():
    # Create sample data
    raw_data_df = pd.DataFrame({
        "Date": ["2022-01-01", "2022-04-01", "2022-07-01", "2022-10-01"],
        "Denmark": [100, 200, 300, 400],  
        "Germany": [10, 20, 30, 40]
    })

    # Call the function
    actual_result = import_quarterly_data(raw_data_df)

    # Define the expected result DataFrame
    expected_result = pd.DataFrame({
        "Denmark": [100.0, 200.0, 300.0, 400.0],
        "Germany": [10, 20, 30, 40]
    }, index=pd.to_datetime(["2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31"]))

    expected_result.index.name = 'Date'

    # Assert that the actual result matches the expected result
    pd.testing.assert_frame_equal(actual_result, expected_result, check_dtype=False)