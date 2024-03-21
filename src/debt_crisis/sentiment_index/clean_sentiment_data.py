import os
import pandas as pd
import regex as re
import numpy as np
from datetime import timedelta

from debt_crisis.config import NLP_MODEL


def clean_transcript_data_df(raw_dataframe):
    """This function takes in a raw dataframe and returns a cleaned version.

    Args: raw_dataframe (pd.DataFrame): Raw dataframe with transcripts as created by the task combine_all_transcripts_into_initial_dataframe

    Returns pd.DataFrame: Cleaned dataframe
    columns: Date (pd.DateTime): Date of the earnings call extracted from the file name
            Company (str): Name of the company as ticker symbol
            Transcript (str): Text of the transcript
            Cleaned_Transcript (str): Preprocessed text of the transcript

    """

    cleaned_data = pd.DataFrame()

    cleaned_data["Date"] = pd.to_datetime(raw_dataframe["Date"])
    cleaned_data["Company"] = raw_dataframe["Company"]

    cleaned_data["Raw_Transcript"] = raw_dataframe["Transcript"]

    def preprocess_and_print(transcript_text):
        print(f"Processing row {preprocess_and_print.row_counter}")
        preprocess_and_print.row_counter += 1
        return preprocess_transcript_text(transcript_text)

    # Initialize a counter attribute for the function
    preprocess_and_print.row_counter = 0

    cleaned_data["Preprocessed_Transcript_Step_1"] = raw_dataframe["Transcript"].apply(
        preprocess_and_print
    )

    return cleaned_data


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

    return {"Date": date, "Company": company, "Transcript": transcript}


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
    data = pd.DataFrame(data_list).sort_values(by="Date").reset_index(drop=True)

    data["Transcript_ID"] = data.index

    return data


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

    return text


def tokenize_text_and_remove_non_alphabetic_characters_and_stop_words(
    column, nlp_model=NLP_MODEL
):
    """This function takes in a column of text and tokenizes each text using the spaCy
    model, and removes stop words and non-alphabetic characters.

    Args:
        column (pd.Series): Series of raw transcript texts
        nlp_model (spacy model): SpaCy model to use for tokenization

    Returns:
        pd.Series: Series of tokenized texts

    """
    # Use SpaCy's pipe method for batch processing
    processed_texts = list(nlp_model.pipe(column))

    # Process each document in the processed_texts
    tokenized_texts = []
    for doc in processed_texts:
        filtered_tokens = [
            token.text for token in doc if not token.is_stop and not token.is_punct
        ]
        tokenized_texts.append(" ".join(filtered_tokens))

    return pd.Series(tokenized_texts, index=column.index)


def clean_sentiment_dictionary_data(raw_data):
    """This function takes in a raw dictionary and returns a cleaned version."""

    cleaned_data = pd.DataFrame()

    cleaned_data["Word"] = raw_data["Word"].str.lower()
    cleaned_data["Positive"] = raw_data["Positive"]
    cleaned_data["Negative"] = raw_data["Negative"]
    cleaned_data["Positive_Indicator"] = np.where(cleaned_data["Positive"] > 0, 1, 0)
    cleaned_data["Negative_Indicator"] = np.where(cleaned_data["Negative"] > 0, 1, 0)

    return cleaned_data


def create_sentiment_dictionary_for_lookups(cleaned_data):
    """THis function takes in a cleaned sentiment dictionary and returns a dictionary
    with the word as key and the sentiment value as value."""

    word_sentiment_dict = {}  # Dictionary to store word sentiment values

    for index, row in cleaned_data.iterrows():
        word = row["Word"]
        positive = row["Positive_Indicator"]
        negative = row["Negative_Indicator"]

        # Calculate the value as positive - negative for the word
        sentiment_value = positive - negative

        # Store the sentiment value in the dictionary
        word_sentiment_dict[word] = sentiment_value

    return word_sentiment_dict


def create_word_count_dictionary(sentiment_dict):
    """This function takes in a cleaned sentiment dictionary and returns a dictionary
    with the word as key and 0 as the value."""

    word_count_dict = {word: 0 for word in sentiment_dict}
    return word_count_dict


def create_country_sentiment_index_for_one_transcript_and_print_transcript_number(
    transcript,
    lookup_dict,
    words_environment,
    country,
    country_names_file,
    word_count_dict,
):
    print(
        f"Processing row {create_country_sentiment_index_for_one_transcript_and_print_transcript_number.row_counter}"
    )
    create_country_sentiment_index_for_one_transcript_and_print_transcript_number.row_counter += (
        1
    )
    return create_country_sentiment_index_for_one_transcript(
        transcript,
        lookup_dict,
        words_environment,
        country,
        country_names_file,
        word_count_dict,
    )


def create_country_sentiment_index_for_one_transcript(
    transcript,
    lookup_dict,
    words_environment,
    country,
    country_names_file,
    word_count_dict,
):
    """This function takes in an earnings call transcript and a lookup dictionary and
    returns a sentiment index for the transcript. The sentiment index is calculated as
    the sum of the sentiment values of the words in the transcript.

    Args: transcript (str): Earnings call transcript
        lookup_dict (dict): Dictionary with words as keys and sentiment values as values
        words_envirnoment (int): number of words before and after the country to consider
        country (str): country to consider
        country_names_file (pd.Dataframe): file with the names of the countries
        word_count_dict (dict): dictionary where we store the number of occurence of the word

    Returns: int: Sentiment index for the transcript

    """

    # First, I get all name versions of the country
    country_row = country_names_file[
        country_names_file["name"].str.lower() == country.lower()
    ]
    if not country_row.empty:
        country_names = set(country_row.iloc[0].values.tolist())
    else:
        country_names = set()

    transcript_words = re.findall(r"\b\w+\b", transcript.lower())

    # Now, I get the indices where these words appear in the transcript
    country_indices = get_country_appearance_index_from_transcript_text(
        transcript_words, country_names
    )

    # Calculate sentiment index based on the words around the country occurrences
    sentiment_index = 0

    for index in country_indices:
        start = max(0, index - words_environment)
        end = min(len(transcript), index + words_environment)
        context_words = transcript_words[start:end]

        for word in context_words:
            if word in lookup_dict:
                sentiment_index += lookup_dict[word]
                word_count_dict[word] += 1  # This adds one to the count dictionary

    return sentiment_index


def get_country_appearance_index_from_transcript_text(transcript_words, country_names):
    """Get a list of indices where any of the country names or their alternate names are
    found in the transcript.

    Args:
        transcript (str): Earnings call transcript
        country_names (set): Set of country names and alternate names

    Returns:
        list: List of indices where country names or alternate names are found

    """

    # Initialize a list to store indices
    country_indices = []

    # Iterate through words and check for country names or alternate names
    for i, word in enumerate(transcript_words):
        if word in country_names:
            country_indices.append(i)

    return country_indices


def calculate_loughlan_mcdonald_sentiment_index(
    preprocessed_data, countries_under_study, day_window=90
):
    """This function calculates the sentiment index taking as input the preprocessed
    data generated by earlier functions in this script.

    Args:
        preprocessed_data (pd.DataFrame): Dataframe with the preprocessed data (name is df_transcripts_clean_step_2.pkl)
        countries_under_study (list): List of countries to consider
        day_window (int): Number of days to consider for the sentiment index

    Returns:
        pd.DataFrame: Dataframe with the sentiment index
        columns: Date (pd.DateTime): Date of sentiment index
                Sentiment_Index_country (int): Sentiment index for the country (there is one of such columns for every country under study.)

    """

    # Create date range from January 2003 to January 2023
    date_range = pd.date_range(start="1/1/2003", end="1/1/2023")

    # Initialize a DataFrame with 'Date' column
    result_df = pd.DataFrame(date_range, columns=["Date"])

    # Set 'Date' as index for efficient lookup
    result_df.set_index("Date", inplace=True)
    preprocessed_data.set_index("Date", inplace=True)

    # Iterate over each date
    for date in date_range:
        # Iterate over each country
        for country in countries_under_study:
            # Calculate the sum of the Sentiment_Index_McDonald_{country} column over the prior day_window days
            end_date = date
            start_date = end_date - timedelta(
                days=day_window
            )  # start date is day_window days before the end date

            # Extract the data for the window
            window_data = preprocessed_data.loc[
                start_date:end_date, f"Sentiment_Index_McDonald_{country}"
            ]

            # Calculate the sentiment index
            sentiment_index = (
                window_data.sum() / len(window_data) if len(window_data) > 0 else np.nan
            )

            # Add the sentiment index to the result DataFrame
            result_df.loc[date, f"Sentiment_Index_McDonald_{country}"] = sentiment_index

    return result_df.reset_index()
