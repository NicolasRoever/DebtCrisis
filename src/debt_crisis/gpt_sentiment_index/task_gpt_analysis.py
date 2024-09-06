from debt_crisis.gpt_sentiment_index.gpt_index_analysis import (
    clean_llm_output_data,
    calculate_gpt_sentiment_index,
    plot_sentiment_indices_and_bond_yields,
    create_correlation_matrix_between_sentiment_indices_and_yields,
)


from debt_crisis.config import SRC, BLD

import pandas as pd

from pytask import task


country_list = [
    {"name": "greece", "file_name": "sentiment_data_greece_output_v006.csv"},
    {"name": "portugal", "file_name": "sentiment_data_portugal_output_v005.csv"},
    {"name": "germany", "file_name": "sentiment_data_germany_output_v001.csv"},
    {"name": "france", "file_name": "sentiment_data_france_output_v001.csv"},
    {"name": "italy", "file_name": "sentiment_data_italy_output_v001.csv"},
    {"name": "ireland", "file_name": "sentiment_data_ireland_output_v001.csv"},
    {"name": "netherlands", "file_name": "sentiment_data_netherlands_output_v001.csv"},
    {"name": "austria", "file_name": "sentiment_data_austria_output_v001.csv"},
    {"name": "hungary", "file_name": "sentiment_data_hungary_output_v001.csv"},
    {"name": "poland", "file_name": "sentiment_data_poland_output_v001.csv"},
    {"name": "denmark", "file_name": "sentiment_data_denmark_output_v001.csv"},
    {"name": "sweden", "file_name": "sentiment_data_sweden_output_v001.csv"},
]


for country_dictionary in country_list:
    country_name = country_dictionary["name"]
    file_name = country_dictionary["file_name"]

    @task
    def task_clean_llm_output_data(
        depends_on={
            "llm_output": SRC / "data" / "GPT_Output_Data" / file_name,
            "raw_transcript_data": BLD / "data" / "df_transcripts_raw.pkl",
            "training_data": BLD
            / "data"
            / "gpt_sentiment_data"
            / "df_gpt_sentiment_training_dataset_cleaned.pkl",
        },
        produces=BLD
        / "data"
        / "GPT_Output_Data"
        / f"sentiment_data_clean_{country_name}.pkl",
    ):
        llm_output_data = pd.read_csv(depends_on["llm_output"])
        transcript_data = pd.read_pickle(depends_on["raw_transcript_data"])
        training_data = pd.read_pickle(depends_on["training_data"])

        llm_output_data_clean = clean_llm_output_data(
            llm_output_data=llm_output_data,
            raw_transcript_data=transcript_data,
            training_data=training_data,
        )

        llm_output_data_clean.to_pickle(produces)


country_list = [
    "greece",
    "portugal",
    "germany",
    "france",
    "italy",
    "ireland",
    "netherlands",
    "austria",
    "hungary",
    "poland",
    "denmark",
    "sweden",
]


def task_create_full_clean_sentiment_output(
    depends_on=BLD / "data" / "GPT_Output_Data" / f"sentiment_data_clean_portugal.pkl",
    country_list=country_list,
    produces=BLD / "data" / "GPT_Output_Data" / f"sentiment_data_clean_full.pkl",
):
    sentiment_data_full = pd.DataFrame()

    for country_name in country_list:
        sentiment_data = pd.read_pickle(
            BLD
            / "data"
            / "GPT_Output_Data"
            / f"sentiment_data_clean_{country_name}.pkl"
        )

        sentiment_data_full = pd.concat([sentiment_data_full, sentiment_data])

    sentiment_data_full.to_pickle(produces)


for country_name in country_list:

    @task
    def task_plot_sentiment_index_and_bond_yield(
        depends_on={
            "llm_output_data_clean": BLD
            / "data"
            / "GPT_Output_Data"
            / f"sentiment_data_clean_{country_name}.pkl",
            "mcdonald_sentiment_data": BLD
            / "data"
            / "mcdonald_sentiment_index_negative_and_positive_20_.pkl",
            "bond_yield_spread": BLD
            / "data"
            / "financial_data"
            / "Quarterly Macroeconomic Variables_cleaned.pkl",
        },
        country=country_name,
        produces=BLD
        / "figures"
        / "sentiment_index_all"
        / f"sentiment_index_{country_name}.pdf",
    ):
        # Import Data

        bond_yield_spread = pd.read_pickle(depends_on["bond_yield_spread"])
        bond_yield_spread_filter = bond_yield_spread[
            bond_yield_spread["Country"] == country
        ]
        bond_yield_spread_filter["Date"] = pd.to_datetime(
            bond_yield_spread_filter["Date"]
        )
        quarterly_dates = bond_yield_spread_filter["Date"]

        llm_output_data_clean = pd.read_pickle(depends_on["llm_output_data_clean"])

        llm_sentiment_index_data = calculate_gpt_sentiment_index(
            preprocessed_data=llm_output_data_clean, country_under_study=country
        )
        llm_quarter_data = llm_sentiment_index_data[
            llm_sentiment_index_data["Date"].isin(quarterly_dates)
        ]

        mcdonald_sentiment_data = pd.read_pickle(depends_on["mcdonald_sentiment_data"])
        mcdonald_quarter_data = mcdonald_sentiment_data[
            mcdonald_sentiment_data["Date"].isin(quarterly_dates)
        ]

        plot = plot_sentiment_indices_and_bond_yields(
            gpt_sentiment_index=llm_quarter_data,
            mcdonald_sentiment_index=mcdonald_quarter_data,
            bond_yield_data=bond_yield_spread_filter,
            country_under_study=country,
        )

        plot.savefig(produces)


for country_name in country_list:

    @task(id=country_name)
    def task_create_correlation_matrix(
        depends_on={
            "llm_output_data_clean": BLD
            / "data"
            / "GPT_Output_Data"
            / f"sentiment_data_clean_{country_name}.pkl",
            "mcdonald_sentiment_data": BLD
            / "data"
            / "mcdonald_sentiment_index_negative_and_positive_20_.pkl",
            "bond_yield_spread": BLD
            / "data"
            / "financial_data"
            / "Quarterly Macroeconomic Variables_cleaned.pkl",
        },
        country=country_name,
        produces=BLD
        / "figures"
        / "correlation_matrix"
        / f"correlation_matrix_{country_name}.pdf",
    ):
        # Import Data

        bond_yield_spread = pd.read_pickle(depends_on["bond_yield_spread"])
        bond_yield_spread_filter = bond_yield_spread[
            bond_yield_spread["Country"] == country
        ]
        bond_yield_spread_filter["Date"] = pd.to_datetime(
            bond_yield_spread_filter["Date"]
        )
        quarterly_dates = bond_yield_spread_filter["Date"]

        llm_output_data_clean = pd.read_pickle(depends_on["llm_output_data_clean"])

        llm_sentiment_index_data = calculate_gpt_sentiment_index(
            preprocessed_data=llm_output_data_clean, country_under_study=country
        )
        llm_quarter_data = llm_sentiment_index_data[
            llm_sentiment_index_data["Date"].isin(quarterly_dates)
        ]

        mcdonald_sentiment_data = pd.read_pickle(depends_on["mcdonald_sentiment_data"])
        mcdonald_quarter_data = mcdonald_sentiment_data[
            mcdonald_sentiment_data["Date"].isin(quarterly_dates)
        ]

        correlation_matrix = (
            create_correlation_matrix_between_sentiment_indices_and_yields(
                gpt_sentiment_index=llm_quarter_data,
                mcdonald_sentiment_index=mcdonald_quarter_data,
                bond_yield_data=bond_yield_spread_filter,
                country_under_study=country,
            )
        )

        correlation_matrix.savefig(produces)
