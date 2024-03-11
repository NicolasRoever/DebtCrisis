import pandas as pd



# Define funtion to read quarterly data
def import_quarterly_data(raw_data):
    """This function imports all data from EUROSTAT which was downloaded in the
    quarterly format."""

    df = pd.DataFrame()

    # Append other columns as numeric
    for column in raw_data.columns:
        if column != "Date":  # Skip the Date column
            df[column] = raw_data[column].astype(float)

  
    df["Date"] = pd.to_datetime(raw_data["Date"]) + pd.offsets.QuarterEnd() 

    # Set date as index
    df.set_index("Date", inplace=True)

    

    return df