import pandas as pd
import logging
import numpy as np
from bertopic import BERTopic
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired
from IPython.display import display
import random
from transformers import BertTokenizer, BertModel
import torch
from typing import List, Set
from concurrent.futures import ThreadPoolExecutor
from umap import UMAP
from nltk.stem import PorterStemmer
import spacy
from tqdm import tqdm

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")
# Set up loggings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------------------
# Functions for Preprocessing


def create_set_with_all_country_words(country_names_file):
    # Flatten the DataFrame to a single list
    country_words = country_names_file.values.flatten()

    # Remove NaN values
    country_words = [word for word in country_words if pd.notna(word)]

    # Create a set of unique words
    return set(country_words)


def prepare_rationalizes_for_creating_embeddings(
    rationales: List[str], country_names: pd.DataFrame
) -> List[str]:
    """This function prepares the rationales for creating embeddings. It removes the
    country words and lemmatizes the text.

    Args:
    rationales: A list of rationales.
    country_names: A DataFrame containing the country names.

    Returns:
    A list of rationales with the country names replaced by a placeholder.

    """

    country_names_set = create_set_with_all_country_words(country_names)

    rationales_without_country_names = remove_country_names(
        texts=rationales, country_names=country_names_set
    )

    # Add a progress bar to the pandas apply method
    tqdm.pandas(desc="Lemmatizing Texts")

    # Apply lemmatization with a progress bar
    lemmatized_rationales = [
        lemmatize_text(rationale)
        for rationale in tqdm(
            rationales_without_country_names, desc="Lemmatizing Texts"
        )
    ]

    return lemmatized_rationales


def remove_country_names(texts: List[str], country_names: Set[str]) -> List[str]:
    """Remove country names from a list of text snippets.

    Parameters:
    texts (List[str]): A list of text snippets from which to remove country names.
    country_names (Set[str]): A set of country names to be removed from the texts.
                              Country names may contain variations separated by semicolons.

    Returns:
    List[str]: A list of text snippets with country names removed.

    """
    # Flatten the country names by splitting on semicolons and creating a new set
    flattened_country_names = set()
    for name in country_names:
        # Split names by semicolon and strip whitespace
        flattened_country_names.update(name.split(";"))

    # Clean up any leading or trailing spaces in country names
    flattened_country_names = {name.strip() for name in flattened_country_names}

    # Create a regex pattern to match any of the country names
    pattern = re.compile(
        r"\b(" + "|".join(map(re.escape, flattened_country_names)) + r")\b",
        re.IGNORECASE,
    )

    # Replace country names in each text
    modified_texts = [pattern.sub("", text) for text in texts]

    return modified_texts


def lemmatize_text(text: str) -> str:
    """Lemmatize the given text using spaCy.

    Parameters:
    text (str): A string containing the text to be lemmatized.

    Returns:
    str: A string containing the lemmatized version of the input text,
            with only alphabetic tokens included.

    """
    # Process the text using spaCy
    doc = nlp(text)
    # Extract the lemma for each token and filter out non-alphabetic tokens
    lemmatized_text = " ".join(token.lemma_ for token in doc if token.is_alpha)
    return lemmatized_text


def get_embeddings_for_text_snippets(
    texts: List[str],
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
        "yiyanghkust/finbert-pretrain"
    ),
    model: BertModel = BertModel.from_pretrained("yiyanghkust/finbert-pretrain"),
    max_length: int = 128,
    batch_size: int = 32,
    use_gpu: bool = False,
) -> np.ndarray:
    """Generate embeddings for a list of texts using a pre-trained BERT-based model.

    Parameters:
    texts (List[str]): A list of text snippets to be encoded into embeddings.
    tokenizer (BertTokenizer, optional): The tokenizer to process the input texts.
    model (BertModel, optional): The BERT-based model to generate embeddings.
    max_length (int, optional): The maximum length for tokenization. Defaults to 128.
    batch_size (int, optional): The number of texts to process in a batch. Defaults to 32.
    use_gpu (bool, optional): Whether to use GPU acceleration. Defaults to False.

    Returns:
    np.ndarray: A 2D numpy array where each row corresponds to the embedding of a text snippet.

    """
    if use_gpu and torch.cuda.is_available():
        model = model.to("cuda")

    embeddings = []
    total_texts = len(texts)

    # Process texts in batches
    for start_idx in range(0, total_texts, batch_size):
        end_idx = min(start_idx + batch_size, total_texts)
        batch_texts = texts[start_idx:end_idx]

        # Tokenize and encode the text data
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )

        if use_gpu and torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate embeddings without gradients
        with torch.no_grad():
            outputs = model(**inputs)

        # Mean pooling of token embeddings
        mean_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(mean_embeddings)

        # Print progress every 20 texts
        if (end_idx) % 20 == 0:
            progress = end_idx / total_texts * 100
            print(f"Progress: {progress:.2f}% ({end_idx}/{total_texts})")

    # Stack embeddings into a single numpy array
    return np.vstack(embeddings)
