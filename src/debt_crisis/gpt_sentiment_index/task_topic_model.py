from debt_crisis.config import SRC, BLD
from debt_crisis.gpt_sentiment_index.topic_model import (
    get_embeddings_for_text_snippets,
    prepare_rationalizes_for_creating_embeddings,
)

import pandas as pd
import numpy as np
import csv


# def task_generate_lemmatized_text_from_rationales(
#         depends_on = {
#             "llm_output_data": BLD / "data" / 'GPT_Output_Data' / 'sentiment_data_clean_full.pkl',
#             "country_names": SRC / "data" / "country_names" / "country_names.csv"
#         },
#         produces =  BLD / "data" / 'topic_model_intermediate' / 'text_snippets_lemmatized_v003.csv'
#         ):


#     llm_output_data = pd.read_pickle(depends_on["llm_output_data"])

#     country_names = pd.read_csv(depends_on["country_names"])

#     rationales = llm_output_data['Rationale_for_Prediction'].tolist()


#     lemmatized_rationales = prepare_rationalizes_for_creating_embeddings(
#         rationales=rationales,
#         country_names=country_names
#     )

#     with open(produces, 'w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(lemmatized_rationales)


def task_get_embedings_from_lemmatized_rationales(
    depends_on=BLD
    / "data"
    / "topic_model_intermediate"
    / "text_snippets_lemmatized_v003.csv",
    produces=BLD / "data" / "topic_model_intermediate" / "embeddings_v003.npy",
):
    with open(depends_on, "r") as file:
        reader = csv.reader(file)
        lemmatized_rationales = next(reader)

    embeddings = get_embeddings_for_text_snippets(lemmatized_rationales)

    np.save(produces, embeddings)
