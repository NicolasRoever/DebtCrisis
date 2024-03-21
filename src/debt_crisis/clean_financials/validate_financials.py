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
