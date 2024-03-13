import pandas as pd


# Define funtion to read quarterly data
def import_quarterly_data(raw_data):
    """This function imports all data from EUROSTAT which was downloaded in the
    quarterly format."""

    df = pd.DataFrame()

    # Get date column
    date_column = "Date" if "Date" in raw_data.columns else "date"
    df["Date"] = pd.to_datetime(raw_data[date_column]) + pd.offsets.QuarterEnd()

    # Append other columns as numeric
    for column in raw_data.columns:
        if column != date_column:  # Skip the Date column
            df[column] = raw_data[column].astype(float)

    return df


def clean_bond_yield_spreads(imported_data):
    """This function cleans the bond yield spreads data."""

    # Reshape the data
    cleaned_data = imported_data.melt(
        id_vars="Date", var_name="Country", value_name="Bond_Yield"
    )

    return cleaned_data
