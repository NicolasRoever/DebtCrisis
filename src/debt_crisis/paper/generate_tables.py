import pandas as pd


def generate_summary_statistics_table_event_study(data):
    table_body = convert_dataframe_content_to_latex_table_body(data)

    number_of_observations = data["Number_of_Observations"].sum()

    table = f"""
    \\begin{{center}}
    \\begin{{table}}[h!] \\caption{{Descriptive Statistics of Data Used in Event Study}}
    \\label{{table:descriptives_event_study}}
    \\scalebox{{1}}{{
    \\begin{{tabular}}{{p{{3cm}}p{{0.2cm}}p{{3cm}}p{{3cm}}p{{1.5cm}}p{{0.2cm}}p{{2cm}}p{{1.5cm}}}}
    \\toprule
    & \multicolumn{{6}}{{c}}{{Average}} & & \\\\
    \\textbf{{Country}} & \\textbf{{Public Debt as \\% of GDP}} & \\textbf{{10 Year Sovereign Bond Yield}} & \\textbf{{10 Year Sovereign Bond Yield Spread}} & \\textbf{{Real GDP Growth}} & \\textbf{{Mode Moody Sovereign Rating}}& \\textbf{{Number Observations}} \\\\
    {table_body}
    \\midrule
    \\begin{{minipage}}{{15cm}}
    \\footnotesize{{\\textbf{{Notes:}} The table presents summary statistics for the dataset we compiled for our event study. We have a total of {number_of_observations} observations as there are some missing data (i.e. in the bond yields we obtained from the OECD).}}
    \\end{{minipage}}
       \\end{{tabular}}
    }}
    \\end{{table}}
    \\end{{center}}
    """

    return table


def generate_descriptive_statistics_from_full_event_study_dataset(data):
    # Create Dexcriptive Statistics

    # Group the data by 'Country' and calculate the mean of the specified columns
    average_data = data.groupby("Country")[
        [
            "Public_Debt_as_%_of_GDP",
            "10y_Maturity_Bond_Yield",
            "GDP_in_Current_Prices_Growth",
        ]
    ].mean()

    # Calculate the most frequent 'Rating_Moody_Last_Quarter_Day' for each country
    average_data["Most Frequent Rating_Moody_Last_Quarter_Day"] = data.groupby(
        "Country"
    )["Rating_Moody_Last_Quarter_Day"].agg(pd.Series.mode)

    # Calculate the number of observations for each country
    average_data["Number_of_Observations"] = data.groupby("Country").size()

    average_data = average_data.round(2)

    # Reset the index
    average_data = average_data.reset_index()

    average_data["Country"] = average_data["Country"].str.title()

    # Insert empty columns for breaks
    average_data.insert(2, "Break1", "")
    average_data.insert(5, "Break2", "")

    return average_data


def convert_dataframe_content_to_latex_table_body(data):
    # Convert each row to a string with ' & ' as the separator
    data_string = data.apply(lambda row: " & ".join(row.astype(str)), axis=1)

    # Join all rows into a single string with ' \\\\\n' as the separator
    data_string = " \\\\".join(data_string)

    # Add ' \\\\' at the end of the string
    data_string += " \\\\"

    return data_string
