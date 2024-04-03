# import pandas as pd
# from pytask import task
# import pytask
# import matplotlib.pyplot as plt
# import pickle

# from debt_crisis.config import (
#     BLD,
#     SRC,
#     NO_LONG_RUNNING_TASKS,
#     COUNTRIES_UNDER_STUDY,
#     CONFIGURATION_SETTINGS,
# )
# from debt_crisis.sentiment_index.clean_sentiment_data import (
#     combine_all_transcripts_into_dataframe,
#     clean_transcript_data_df,
#     tokenize_text_and_remove_non_alphabetic_characters_and_stop_words,
#     clean_sentiment_dictionary_data,
#     create_sentiment_dictionary_for_lookups,
#     create_country_sentiment_index_for_one_transcript_and_print_transcript_number,
#     calculate_loughlan_mcdonald_sentiment_index,
#     create_word_count_dictionary,
# )

# from debt_crisis.utilities import _name_sentiment_index_output_file


# @pytask.mark.skipif(NO_LONG_RUNNING_TASKS, reason="Skip long-running tasks.")
# def task_combine_all_transcripts_into_initial_dataframe(
#     data_directory=str(
#         SRC / "data" / "transcripts" / "raw" / "Eikon 2002 - 2022",
#     ),
#     produces=BLD / "data" / "df_transcripts_raw.pkl",
# ):
#     full_dataframe = combine_all_transcripts_into_dataframe(data_directory)

#     full_dataframe.to_pickle(produces)


# @pytask.mark.skipif(NO_LONG_RUNNING_TASKS, reason="Skip long-running tasks.")
# def task_calculate_McDonald_sentiment_index(
#     depends_on=BLD
#     / "data"
#     / _name_sentiment_index_output_file(
#         "df_transcripts_clean_step_2", CONFIGURATION_SETTINGS, ".pkl"
#     ),
#     countries=COUNTRIES_UNDER_STUDY,
#     produces=BLD
#     / "data"
#     / _name_sentiment_index_output_file(
#         "mcdonald_sentiment_index", CONFIGURATION_SETTINGS, ".pkl"
#     ),
# ):
#     df = pd.read_pickle(depends_on)

#     sentiment_index_data = calculate_loughlan_mcdonald_sentiment_index(df, countries)

#     sentiment_index_data.to_pickle(produces)


# task_clean_transcript_data_step_2_dependencies = {
#     "df_transcripts_step_1": BLD / "data" / "df_transcripts_clean_step_1.pkl",
#     "sentiment_dictionary": BLD / "data" / "sentiment_dictionary_lookup.pkl",
#     "country_names_file": SRC / "data" / "country_names" / "country_names.xlsx",
#     "words_environment": 20,
# }


# @pytask.mark.skipif(NO_LONG_RUNNING_TASKS, reason="Skip long-running tasks.")
# def task_clean_transcript_data_step_2(
#     depends_on=task_clean_transcript_data_step_2_dependencies,
#     countries_under_study=COUNTRIES_UNDER_STUDY,
#     produces=[
#         BLD
#         / "data"
#         / _name_sentiment_index_output_file(
#             "df_transcripts_clean_step_2", CONFIGURATION_SETTINGS, ".pkl"
#         ),
#         BLD
#         / "data"
#         / _name_sentiment_index_output_file(
#             "filled_word_count_dict", CONFIGURATION_SETTINGS, ".pkl"
#         ),
#     ],
# ):
#     # Load Data
#     cleaned_data = pd.read_pickle(depends_on["df_transcripts_step_1"])
#     lookup_dict = pickle.load(open(depends_on["sentiment_dictionary"], "rb"))
#     word_count_dict = create_word_count_dictionary(lookup_dict)
#     country_names_file = pd.read_excel(depends_on["country_names_file"])
#     words_environment = depends_on["words_environment"]

#     for country in countries_under_study:
#         print(country)

#         # Initialize Row counter
#         create_country_sentiment_index_for_one_transcript_and_print_transcript_number.row_counter = (
#             0
#         )

#         # Make Sentiment Index
#         cleaned_data[f"Sentiment_Index_McDonald_{country}"] = cleaned_data[
#             "Preprocessed_Transcript_Step_1"
#         ].apply(
#             lambda x: create_country_sentiment_index_for_one_transcript_and_print_transcript_number(
#                 x,
#                 lookup_dict,
#                 words_environment,
#                 country,
#                 country_names_file,
#                 word_count_dict,
#             )
#         )

#     cleaned_data.to_pickle(produces[0])

#     word_count_df = pd.DataFrame.from_dict(word_count_dict, orient="index")
#     word_count_df = word_count_df.transpose().rename_axis("Keys").reset_index()
#     word_count_df.to_pickle(produces[1])


# @pytask.mark.skipif(NO_LONG_RUNNING_TASKS, reason="Skip long-running tasks.")
# def task_clean_transcript_data_step_1(
#     depends_on=BLD / "data" / "df_transcripts_raw.pkl",
#     produces=BLD / "data" / "df_transcripts_clean_step_1.pkl",
# ):
#     raw_data = pd.read_pickle(depends_on)
#     cleaned_data = clean_transcript_data_df(raw_data)

#     cleaned_data.to_pickle(produces)


# # def task_plot_raw_data_sentiment_dictionart_barplot(
# #     depends_on=BLD / "data" / "sentiment_dictionary_clean.pkl",
# #     produces=BLD / "figures" / "barplot_sentiment_dictionary.png",
# # ):
# #     df = pd.read_pickle(depends_on)
# #     positive_sum = df["Positive"].sum()
# #     negative_sum = df["Negative"].sum()

# #     # Create a bar plot
# #     plt.bar(["Positive", "Negative"], [positive_sum, negative_sum])

# #     # Add labels and title
# #     plt.xlabel("Sentiment")
# #     plt.ylabel("Count")
# #     plt.title("Sum of Positive and Negative Words")

# #     # Save the plot as a PNG file (replace 'figure.png' with your desired file name)
# #     plt.savefig(produces)
