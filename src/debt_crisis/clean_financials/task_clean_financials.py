import pandas as pd
from pytask import task
import matplotlib.pyplot as plt

from debt_crisis.clean_financials.clean_financials import (
    import_quarterly_data,
    clean_bond_yield_spreads,
    import_ratings_data,
    generate_long_format_rating_data,
    merge_alphabetic_with_numeric_rating_data,
    clean_gdp_data,
)

from debt_crisis.utilities import _make_missing_values_heatmap

from debt_crisis.config import BLD, QUARTERLY_DATA_PATHS, SRC, RATINGS_DATA_PATH

for index, filepath in enumerate(QUARTERLY_DATA_PATHS):

    @task(id=f"filename_{index}")
    def task_import_qarterly_data(
        depends_on=filepath,
        produces=BLD / "data" / "financial_data" / f"{filepath.stem}.pkl",
    ):
        if depends_on.suffix == ".csv":
            raw_data = pd.read_csv(depends_on)
        elif depends_on.suffix in [".xls", ".xlsx"]:
            raw_data = pd.read_excel(depends_on)
        else:
            msg = "Unsupported file format. Only CSV and Excel files are supported."
            raise ValueError(
                msg,
            )

        clean_data = import_quarterly_data(raw_data)

        clean_data.to_pickle(produces)


def task_plot_debt_gdp_timeseries(
    depends_on=BLD / "data" / "financial_data" / "debt-to-gdp_EU.pkl",
    produces=BLD / "figures" / "debt_gdp_timeseries.png",
):
    # Load the data
    df = pd.read_pickle(depends_on)

    fig, ax = plt.subplots(figsize=(15, 10))  # Increase the size of the figure

    # Plot each column
    for column in df.columns:
        ax.plot(df.index, df[column], label=column)

    # Move the legend inside the plot
    ax.legend(loc="upper right")

    # Save the figure
    fig.savefig(produces)


def task_clean_bond_yield_spread_data(
    depends_on=BLD / "data" / "financial_data" / "eu_yields_10y.pkl",
    produces=BLD / "data" / "financial_data" / "eu_yields_10y_cleaned.pkl",
):
    # Load the data
    df = pd.read_pickle(depends_on)

    cleaned_data = clean_bond_yield_spreads(df)

    cleaned_data.to_pickle(produces)


def task_clean_debt_to_gdp_data(
    depends_on=BLD / "data" / "financial_data" / "debt-to-gdp_EU.pkl",
    produces=BLD / "data" / "financial_data" / "debt-to-gdp_EU_cleaned.pkl",
):
    # Load the data
    df = pd.read_pickle(depends_on)

    cleaned_data = df.melt(
        id_vars="Date", var_name="Country", value_name="Debt_to_GDP_Ratio"
    )

    cleaned_data["Country"] = cleaned_data["Country"].str.lower()
    # Save the cleaned data
    cleaned_data.to_pickle(produces)


def task_plot_bond_yields(
    depends_on=BLD / "data" / "financial_data" / "eu_yields_10y_cleaned.pkl",
    produces=BLD / "figures" / "bond_yields.png",
):
    # Load the data
    df = pd.read_pickle(depends_on)

    # Filter the data for the specified countries
    countries = ["greece", "italy", "spain", "germany", "france", "ireland"]
    df = df[df["Country"].isin(countries)]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the bond yields for each country
    for country in countries:
        country_data = df[df["Country"] == country]
        ax.plot(country_data["Date"], country_data["Bond_Yield"], label=country)

    # Add a legend
    ax.legend()

    # Save the figure
    fig.savefig(produces)


def task_plot_debt_to_gdp(
    depends_on=BLD / "data" / "financial_data" / "debt-to-gdp_EU_cleaned.pkl",
    produces=BLD / "figures" / "debt_to_gdp.png",
):
    # Load the data
    df = pd.read_pickle(depends_on)

    # Filter the data for the specified countries
    countries = ["greece", "italy", "spain", "germany", "france", "ireland"]
    df = df[df["Country"].isin(countries)]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the bond yields for each country
    for country in countries:
        country_data = df[df["Country"] == country]
        ax.plot(country_data["Date"], country_data["Debt_to_GDP_Ratio"], label=country)

    # Add a legend
    ax.legend()

    # Save the figure
    fig.savefig(produces)


def task_clean_current_account_data(
    depends_on=BLD / "data" / "financial_data" / "current account_EU.pkl",
    produces=BLD / "data" / "financial_data" / "current_account_EU_cleaned.pkl",
):
    """This function cleans the current account data."""

    raw_data = pd.read_pickle(depends_on)

    # Reshape the dat
    cleaned_data = raw_data.melt(
        id_vars="Date", var_name="Country", value_name="Current_Account_Balance"
    )

    cleaned_data["Country"] = cleaned_data["Country"].str.lower()

    cleaned_data.to_pickle(produces)


def task_import_ratings_data(
    produces=BLD / "data" / "financial_data" / "imported_ratings.pkl",
):
    imported_ratings = import_ratings_data(RATINGS_DATA_PATH)

    imported_ratings.to_pickle(produces)


def task_clean_rating_data(
    depends_on=BLD / "data" / "financial_data" / "imported_ratings.pkl",
    rating_conversion_data_path=SRC
    / "data"
    / "financial_data"
    / "ratings"
    / "conversion_ratings.xlsx",
    produces=BLD / "data" / "financial_data" / "processed_ratings.pkl",
):
    # Load the imported rating data
    imported_rating_data = pd.read_pickle(depends_on)

    imported_rating_data["Date"] = imported_rating_data.index

    # Load the rating conversion data
    rating_conversion_data = pd.read_excel(rating_conversion_data_path)

    # Process the rating data
    long_format_rating_data = generate_long_format_rating_data(imported_rating_data)

    # Add numeric ratings
    long_format_rating_data_with_numeric = merge_alphabetic_with_numeric_rating_data(
        long_format_rating_data, rating_conversion_data
    )

    _make_missing_values_heatmap(long_format_rating_data_with_numeric, "Rating Data")

    long_format_rating_data_with_numeric.to_pickle(produces)


def task_clean_gdp_data(
    depends_on=BLD / "data" / "financial_data" / "gdp_EU.pkl",
    produces=BLD / "data" / "financial_data" / "gdp_data_cleaned.pkl",
):
    """This function reshapes the GDP data to long format and calculates the GDP
    growth."""

    # Load the GDP data
    gdp_data = pd.read_pickle(depends_on)

    cleaned_data = clean_gdp_data(gdp_data)

    cleaned_data.to_pickle(produces)
