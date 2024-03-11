import os
import re

import pandas as pd
import regex as re
from spacy.tokens import Doc

from debt_crisis.config import NLP_MODEL


def extract_date_from_transcript(transcript):
    """This function extracts the date information from a given raw transcript.
    Args: Raw Transcript as string.

    Returns: Date string
    """
    # Regular expression pattern to match the date information
    pattern = r"([A-Z][a-z]+ \d{1,2}, \d{4})"

    # Search for the pattern in the transcript
    match = re.search(pattern, transcript, re.IGNORECASE)

    if match:
        return match.group(1)  # Extracted date
    else:
        return None


# Define function to extract date of conference call and company name from file name and text from txt files
def extract_data_from_file(file_path):
    """This function loads all transcript data and returns a dictionary with the date,
    quarter, year, company name and transcript."""
    # Extract date and company name from file name
    date_list = os.path.basename(file_path).split("-")[:3]
    date_str = "-".join(date_list)
    company = os.path.basename(file_path).split("-")[3]

    # Transform date string into datetime object and remove time
    date = pd.to_datetime(date_str, format="%Y-%b-%d").date()

    # Extract text from txt files
    with open(file_path, encoding="utf-8") as f:
        transcript = f.read()

    return {"date": date, "company": company, "transcript": transcript}


def combine_all_transcripts_into_dataframe(root_transcript_directory):
    """This function goes over all transcripts saved in the data folder under src/debt_crisis/data/transcripts/raw/Eikon 2002 - 2022
    and combines them into one dataframe.

    Args: root_transcript_directory: Path to the root directory of the transcripts: src/debt_crisis/data/transcripts/raw/Eikon 2002 - 2022

    Returns: Dataframe with all transcripts
        Columns: Date (pd.DateTime): Date of the earnings call extracted from the file name
                Company (str): Name of the company as ticker symbol
                    Transcript (str): Text of the transcript
    """
    # Initialize list to store data
    data_list = []

    # Loop over files in directory and subdirectories and apply function to extract data from files
    for root, _dirs, files in os.walk(root_transcript_directory):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                data_list.append(extract_data_from_file(file_path))

    # Create data frame from the list
    return pd.DataFrame(data_list).sort_values(by="date").reset_index(drop=True)


def preprocess_transcript_text(raw_transcript_text, nlp_model=NLP_MODEL):
    """THis function takes in a raw transcript and makes standard preprocessing."""
    # Pre-compile regular expressions
    presentation_regex = re.compile(r".*Presentation\n-*\n|PRELIMINARY TRANSCRIPT:.*")
    speaker_regex = re.compile(r"^.*\[\d+\].*$\n?")
    dash_equals_regex = re.compile(r"[=-]{3,}")
    number_date_regex = re.compile(
        r"\d{1,2}(\s*(th|st|nd|rd))?(\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec))?(\s*\d{2,4})?|\d+",
    )

    # Remove content before "Presentation" and after "PRELIMINARY TRANSCRIPT:"
    text = presentation_regex.sub("", raw_transcript_text)

    # Remove lines with speaker descriptions in Q&A
    text = speaker_regex.sub("", text)

    # Remove consecutive "-" and "=" occurring more than twice
    text = dash_equals_regex.sub("", text)

    # Remove dates and numbers
    text = number_date_regex.sub("", text)

    # Convert to lowercase and remove non-alphabetic characters
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())

    # Tokenize raw_transcript_text and remove stop words using spaCy
    doc = Doc(nlp_model.vocab, words=text.split())

    return [
        token.text
        for token in nlp_model.pipeline[0](doc)
        if not token.is_stop and token.is_alpha
    ]


def tokenize_text_and_remove_non_alphabetic_characters_and_stop_words(
    text,
    nlp_model=NLP_MODEL,
):
    """This function takes in a string of text and tokenizes it using the spaCy model
    and removes stop words and non-alphabetic characters.

    Args: text (str): Raw transcript text
          nlp_model (spacy model): SpaCy model to use for tokenization

    Returns: List of tokens

    """
    # Extract the tokenizer from the pipeline
    tokenizer = None
    for component_name, component_func in nlp_model.pipeline:
        if component_name == "tok2vec":
            tokenizer = component_func
            break

    if tokenizer is None:
        msg = "Tokenizer not found in the spaCy pipeline"
        raise ValueError(msg)

    doc = Doc(nlp_model.vocab, words=text.split())

    return [token.text for token in tokenizer(doc) if not token.is_alpha]
