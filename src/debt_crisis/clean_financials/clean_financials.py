import pandas as pd

import os
import pandas as pd
import numpy as np


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


def clean_current_account_data(imported_data):
    """This function cleans the current account data."""

    # Reshape the data
    cleaned_data = imported_data.melt(
        id_vars="Date", var_name="Country", value_name="Current_Account_Balance"
    )

    cleaned_data["Country"] = cleaned_data["Country"].str.lower()

    return cleaned_data


def import_ratings_data(directory_path):
    """This function takes in the directory path to the rating files and returns a
    single data frame with all the ratings.

    Args: directory_path (pathlib.Path object): The path to the directory containing the rating files.

    Returns: ratings (pandas.DataFrame): A data frame containing all the ratings.
        columns: Date (datetime): the date of the rating
                    Rating (str): the rating
                    Source (str): the source of the rating
                    Country (str): The Country of the rating

    """

    ratings = pd.DataFrame()

    for filename in os.listdir(directory_path):
        if filename.endswith(".xlsx") and filename.startswith("Ratings_"):
            # Read the excel file into a pandas data frame
            filepath = os.path.join(directory_path, filename)
            raw_ratings = pd.read_excel(
                filepath, usecols=["Date", "Issuer Rating", "Rating Source"]
            )

            # Get the country of the ratings
            country = filename.split("_", 1)[1].split(".")[0]

            # Clean the rating file

            cleaned_rating_file = clean_one_rating_file(raw_ratings, country)

            # Concatenate the temporary data frame to the main data frame
            ratings = pd.concat([ratings, cleaned_rating_file], ignore_index=False)

    return ratings


def clean_one_rating_file(temp_ratings, country):
    """This function cleans a DataFrame which is just the read in version of the
    original .xlsx file.

    It filters out rows with NULL values in the 'Date' column or 'WD' / "RD" in the 'Issuer Rating' column,
    converts the 'Date' column to datetime format and sets it as index, extracts the country from the filename,
    and renames the columns.

    Args:
        temp_ratings (pandas.DataFrame): The DataFrame containing the ratings data.
            columns: Date (str): the date of the rating
                     Issuer Rating (str): the issuer rating
                     Rating Source (str): the source of the rating

        filename (str): The name of the file from which the data was read. It should start with "Ratings_"
                         followed by the country name.

    Returns:
        temp_ratings (pandas.DataFrame): The cleaned DataFrame.
            columns: Date (datetime): the date of the rating
                     rating (str): the rating
                     source (str): the source of the rating
                     Country (str): The country of the rating

    """

    # Filter out rows with NULL values in the 'Date' column or 'WD' / "RD" in the 'Issuer Rating' column
    temp_ratings = temp_ratings[
        pd.notnull(temp_ratings["Date"])
        & (~temp_ratings["Issuer Rating"].isin(["WD", "RD", "NR"]))
    ]

    # Convert the 'Date' column to datetime format and set as index
    temp_ratings["Date"] = pd.to_datetime(temp_ratings["Date"], format="mixed")
    temp_ratings.set_index("Date", inplace=True)

    # Extract the word after "Ratings_" in the file name and rename the columns
    temp_ratings.rename(
        columns={"Issuer Rating": "rating", "Rating Source": "source"}, inplace=True
    )
    temp_ratings["Country"] = country.lower()

    return temp_ratings


def generate_long_format_rating_data(imported_data):
    """This function takes in the dataframe generated by the import_ratings_data
    function. It returns a dataframe in long format, where for every country the rating
    from a given source is updated once there is a new rating in the imported data.

    Args: imported_data (pandas.DataFrame): The dataframe generated by the

    Returns: cleaned_data (pandas.DataFrame): A dataframe
        columns: Date (datetime): the date of the rating
                    Rating_Letter_MIS (str): the rating
                    - the columns for the other sources are similarly named
                    Country (str): The Country of the rating

    """

    # Create a date range from 2002-01-01 to 2023-01-01
    dates = pd.date_range(start="2002-01-01", end="2023-01-01")

    # Get the unique countries from the imported data
    countries = imported_data["Country"].unique()

    # Initialize an empty DataFrame to store the results
    result = pd.DataFrame()

    # Process each country
    for country in countries:
        # Filter the imported data for the current country
        country_data = imported_data[imported_data["Country"] == country]

        # Create a DataFrame for the current country with the date range
        df_country = pd.DataFrame(dates, columns=["Date"])
        df_country["Country"] = country

        # Initialize the rating columns with np.nan
        df_country["Rating_Letter_MIS"] = np.nan
        df_country["Rating_Letter_FIS"] = np.nan
        df_country["Rating_Letter_FDL"] = np.nan

        # Update the rating columns based on the imported data
        for index, row in country_data.iterrows():
            source_column = "Rating_Letter_" + row["source"]
            df_country.loc[df_country["Date"] >= row["Date"], source_column] = row[
                "rating"
            ]

        # Append the DataFrame for the current country to the result DataFrame
        result = pd.concat([result, df_country], ignore_index=True)

    return result


def merge_alphabetic_with_numeric_rating_data(
    long_format_rating_data, rating_conversion_data
):
    """This function takes in the long_format rating data and adds columns with the
    respective numeric conversion.

    Args: long_format_rating_data (pandas.DataFrame): The dataframe generated by the generate_long_format_rating_data function
            rating_conversion_data (pandas.DataFrame): The dataframe containing the conversion of the rating letters to numbers

    Returns: merged_data (pandas.DataFrame): A dataframe containing the long format rating data with the numeric conversion

    """

    # Copy the long format rating data to avoid modifying the original DataFrame
    merged_data = long_format_rating_data.copy()

    # For each source in the rating conversion data
    for source in rating_conversion_data["source"].unique():
        # Extract the relevant conversion data for the current source
        conversion_data = rating_conversion_data[
            rating_conversion_data["source"] == source
        ]

        # Rename the columns for merging
        conversion_data = conversion_data.rename(
            columns={
                "rating": "Rating_Letter_" + source,
                "score": "Rating_Numeric_" + source,
            }
        )

        # Merge the dataframes on the rating letter column
        merged_data = pd.merge(
            merged_data, conversion_data, on="Rating_Letter_" + source, how="left"
        )

        # Drop the source column from the merged data
        merged_data = merged_data.drop(columns=["source"])

    return merged_data


def apply_numeric_rating(row, rating_conversion_data):
    """This function takes in a row of the long format rating data and applies the
    numeric rating conversion."""
    # Process each source in the rating conversion data
    for _, conv_row in rating_conversion_data.iterrows():
        # Get the source, rating, and score
        source = conv_row["source"]
        rating = conv_row["rating"]
        score = conv_row["score"]

        # Update the numeric rating column for the current source
        # The numeric rating is the score where the letter rating matches the rating from the rating conversion data
        if row["Rating_Letter_" + source] == rating:
            row["Rating_Numeric_" + source] = score

    return row


def clean_gdp_data(gdp_data):
    """This function reshapes the GDP data to long format and calculates the GDP growth
    and one period GDP growth lead.

    Args:
        gdp_data (pandas.DataFrame): The GDP data in wide format.

    Returns:
        long_format_gdp_data (pandas.DataFrame): The GDP data in long format with the GDP growth and one period GDP growth lead.

    """

    # Reshape the data to long format
    long_format_gdp_data = gdp_data.melt(
        id_vars="Date", var_name="Country", value_name="GDP"
    )

    # Sort the data by country and date
    long_format_gdp_data = long_format_gdp_data.sort_values(["Country", "Date"])

    # Calculate the GDP growth
    long_format_gdp_data["GDP_Growth"] = long_format_gdp_data.groupby("Country")[
        "GDP"
    ].pct_change()

    # Calculate the one period GDP growth lead
    long_format_gdp_data["GDP_Growth_Lead"] = long_format_gdp_data.groupby("Country")[
        "GDP_Growth"
    ].shift(-1)

    long_format_gdp_data["Country"] = long_format_gdp_data["Country"].str.lower()

    return long_format_gdp_data
