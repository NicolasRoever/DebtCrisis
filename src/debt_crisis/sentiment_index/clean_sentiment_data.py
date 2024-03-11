import re


def extract_date_from_transcript(transcript):
    '''This function extracts the date information from a given raw transcript.
    Args: Raw Transcript as string

    Returns: Date string
    '''
    # Regular expression pattern to match the date information
    pattern = r'([A-Z][a-z]+ \d{1,2}, \d{4})'

    # Search for the pattern in the transcript
    match = re.search(pattern, transcript, re.IGNORECASE)

    if match:
        date = match.group(1)  # Extracted date
        return date
    else:
        return None

