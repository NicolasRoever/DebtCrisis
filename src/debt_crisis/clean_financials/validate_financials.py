import matplotlib.pyplot as plt


def plot_time_series_variables_from_different_datasources(
    datasource_1, column_name_1, datasource_2, column_name_2
):
    """This function plots two time series variables from different datasources."""
    fig, ax = plt.subplots()
    ax.plot(datasource_1["Date"], datasource_1[column_name_1], label=column_name_1)
    ax.plot(datasource_2["Date"], datasource_2[column_name_2], label=column_name_2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()

    return fig


def plot_bond_yield_spreads_for_all_countries(data):
    countries = data["Country"].unique()

    fig = plt.figure()

    for country in countries:
        country_data = data[data["Country"] == country]
        plt.plot(country_data["Date"], country_data["Bond_Yield_Spread"], label=country)

    plt.xlabel("Date")
    plt.ylabel("Bond Yield Spread")
    plt.legend()

    return fig


def plot_bond_yield_for_country(data, country):
    country_data = data[data["Country"] == country]
    fig = plt.figure()
    plt.plot(country_data["Date"], country_data["Bond_Yield_Spread"], label=country)
    plt.xlabel("Date")
    plt.ylabel("Bond Yield Spread")
    plt.legend()

    return fig
