# import pandas as pd
# import matplotlib.pyplot as plt
# from pytask import task
# import numpy as np


# from debt_crisis.regression_analysis.event_study import (
#     create_data_set_event_study,
#     run_event_study_regression,
#     plot_event_study_coefficients,
#     plot_event_study_coefficients_for_multiple_countries_in_one_plot,
#     generate_regresssion_table_for_list_of_configurations,
#     run_al_amine_regression,
#     create_comparison_table_our_results_vs_al_amine,
# )

# from debt_crisis.config import (
#     BLD,
#     EVENT_STUDY_COUNTRIES,
#     EVENT_STUDY_TIME_PERIOD,
#     TOP_LEVEL_DIR,
#     CONFIGURATION_SETTINGS,
#     EVENT_STUDY_PLOT_COUNTRIES,
#     EVENT_STUDY_MODEL_LIST,
#     MOODY_RATING_CONVERSION,
# )

# from debt_crisis.utilities import _name_sentiment_index_output_file


# # -----------------Event Study-----------------#


# task_create_event_study_dataset_dependencies = {
#     "quarterly_macro_data": BLD
#     / "data"
#     / "financial_data"
#     / "Quarterly Macroeconomic Variables_cleaned.pkl",
#     "sentiment_index_data": BLD
#     / "data"
#     / _name_sentiment_index_output_file(
#         "mcdonald_sentiment_index_cleaned", CONFIGURATION_SETTINGS, ".pkl"
#     ),
# }


# def task_create_event_study_dataset(
#     depends_on=task_create_event_study_dataset_dependencies,
#     event_study_countries=EVENT_STUDY_COUNTRIES,
#     event_study_time_period=EVENT_STUDY_TIME_PERIOD,
#     produces=BLD
#     / "data"
#     / "event_study_approach"
#     / _name_sentiment_index_output_file(
#         "event_study_dataset", CONFIGURATION_SETTINGS, ".pkl"
#     ),
# ):
#     quarterly_macro_data = pd.read_pickle(depends_on["quarterly_macro_data"])
#     sentiment_index_data = pd.read_pickle(depends_on["sentiment_index_data"])

#     dataset = create_data_set_event_study(
#         quarterly_macro_data,
#         sentiment_index_data,
#         event_study_countries,
#         event_study_time_period=event_study_time_period,
#     )

#     dataset.to_pickle(produces)


# def task_run_bond_yield_event_study(
#     depends_on=BLD
#     / "data"
#     / "event_study_approach"
#     / _name_sentiment_index_output_file(
#         "event_study_dataset", CONFIGURATION_SETTINGS, ".pkl"
#     ),
#     event_study_countries=EVENT_STUDY_COUNTRIES,
#     event_study_time_period=EVENT_STUDY_TIME_PERIOD,
#     produces=[
#         BLD
#         / "models"
#         / _name_sentiment_index_output_file(
#             "event_study_regression", CONFIGURATION_SETTINGS, ".txt"
#         ),
#         BLD
#         / "data"
#         / "event_study_approach"
#         / _name_sentiment_index_output_file(
#             "event_study_coefficients_data", CONFIGURATION_SETTINGS, ".pkl"
#         ),
#         BLD
#         / "data"
#         / "event_study_approach"
#         / _name_sentiment_index_output_file(
#             "event_study_full_model_data", CONFIGURATION_SETTINGS, ".pkl"
#         ),
#     ],
# ):
#     data = pd.read_pickle(depends_on)

#     model, regression_dataset = run_event_study_regression(
#         data, event_study_countries, event_study_time_period
#     )

#     regression_dataset.to_pickle(produces[2])

#     model_summary = model.summary()

#     with open(produces[0], "w") as file:
#         file.write(model_summary.as_text())

#     # Make coefficient data

#     coefficient_data = pd.DataFrame()

#     coefficient_data["Variable"] = model.params.index
#     coefficient_data["Coefficient"] = model.params.values
#     coefficient_data["Standard Errors"] = model.bse.values

#     coefficient_data.to_pickle(produces[1])


# for index, country_list in enumerate(EVENT_STUDY_PLOT_COUNTRIES):
#     plot_number = index + 1

#     @task(id=str(index))
#     def task_plot_event_study_coefficients_for_all_countries_in_one_plot(
#         depends_on=BLD
#         / "data"
#         / "event_study_approach"
#         / _name_sentiment_index_output_file(
#             "event_study_coefficients_data", CONFIGURATION_SETTINGS, ".pkl"
#         ),
#         countries=country_list,
#         produces=[
#             BLD
#             / "figures"
#             / "event_study"
#             / f"event_study_coefficients_{plot_number}.png",
#             TOP_LEVEL_DIR
#             / "Input_for_Paper"
#             / "figures"
#             / f"event_study_coefficients_{plot_number}.png",
#         ],
#     ):
#         data = pd.read_pickle(depends_on)

#         plot = plot_event_study_coefficients_for_multiple_countries_in_one_plot(
#             data, countries
#         )

#         plot.savefig(produces[0])
#         plot.savefig(produces[1])


# for country in EVENT_STUDY_COUNTRIES:

#     @task(id=country)
#     def task_plot_event_study_coefficients(
#         depends_on=BLD
#         / "data"
#         / "event_study_approach"
#         / _name_sentiment_index_output_file(
#             "event_study_coefficients_data", CONFIGURATION_SETTINGS, ".pkl"
#         ),
#         country=country,
#         produces=BLD
#         / "figures"
#         / "event_study"
#         / _name_sentiment_index_output_file(
#             f"event_study_coefficients_{country}", CONFIGURATION_SETTINGS, ".png"
#         ),
#     ):
#         data = pd.read_pickle(depends_on)

#         figure = plot_event_study_coefficients(data, country)

#         figure.savefig(produces)


# def task_generate_data_for_regression_table(
#     depends_on=BLD
#     / "data"
#     / "event_study_approach"
#     / _name_sentiment_index_output_file(
#         "event_study_dataset", CONFIGURATION_SETTINGS, ".pkl"
#     ),
#     event_study_configuration_list=EVENT_STUDY_MODEL_LIST,
#     event_study_time_period=CONFIGURATION_SETTINGS["event_study_time_period"],
#     event_study_countries=EVENT_STUDY_COUNTRIES,
#     produces=BLD
#     / "data"
#     / "event_study_approach"
#     / "event_study_regression_table_data.pkl",
# ):
#     event_study_data = pd.read_pickle(depends_on)

#     regression_table_data = generate_regresssion_table_for_list_of_configurations(
#         event_study_data,
#         event_study_configuration_list,
#         event_study_countries,
#         event_study_time_period,
#     )

#     regression_table_data.to_pickle(produces)


# def task_run_comparison_regression_to_al_amine(
#     depends_on=[
#         BLD
#         / "data"
#         / "event_study_approach"
#         / _name_sentiment_index_output_file(
#             "event_study_dataset", CONFIGURATION_SETTINGS, ".pkl"
#         ),
#         MOODY_RATING_CONVERSION,
#     ],
#     produces=TOP_LEVEL_DIR
#     / "Input_for_Paper"
#     / "tables"
#     / "event_study_comparison_to_al_amine.tex",
# ):
#     data = pd.read_pickle(depends_on[0])

#     model, regression_data = run_al_amine_regression(data, depends_on[1])

#     table = create_comparison_table_our_results_vs_al_amine(model, regression_data)

#     with open(produces, "w") as file:
#         file.write(table)
