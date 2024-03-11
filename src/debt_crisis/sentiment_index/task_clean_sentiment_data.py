from debt_crisis.config import (
    BLD,
    SRC,
)
from debt_crisis.sentiment_index.clean_sentiment_data import (
    combine_all_transcripts_into_dataframe,
)


def task_combine_all_transcripts_into_initial_dataframe(
    data_directory=str(
        SRC / "data" / "transcripts" / "raw" / "Eikon 2002 - 2022",
    ),  # This should be fixed !!!
    produces=BLD / "data" / "df_transcripts_raw.pkl",
):
    full_dataframe = combine_all_transcripts_into_dataframe(data_directory)

    full_dataframe.to_pickle(produces)
