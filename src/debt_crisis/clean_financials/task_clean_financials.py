import pandas as pd
from pytask import task
import matplotlib.pyplot as plt

from debt_crisis.clean_financials.clean_financials import (
    import_quarterly_data,
    clean_bond_yield_spreads,
)
from debt_crisis.config import BLD, QUARTERLY_DATA_PATHS

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
