# src/data/preprocess_data.py

import re
import html
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm.notebook import tqdm

def setup_nltk():
    """
    Download the necessary NLTK resources.
    """

    nltk.download('stopwords')
    nltk.download('wordnet')

# Call the setup function once to download NLTK resources
setup_nltk()


def clean_text(text: str) -> str:
    """
    Clean a text string by removing URLs, mentions, and hashtags.
    
    Args:
        text (str): Input text string.
    
    Returns:
        str: Cleaned text string.
    """
    # Decode HTML entities
    text = html.unescape(text)

    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)

    # Remove special characters and numbers
    text = re.sub(r'[^A-Za-z\s]+', '', text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra spaces
    text = text.strip()    
    
    return text

def preprocess_text(text: str) -> str:
    """
    Preprocess a text string by removing special characters, numbers, and stopwords.
    
    Args:
        text (str): Input text string.
    
    Returns:
        str: Processed text string.
    """
    # Split the text into words
    words = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Join the words back into a single string
    processed_text = ' '.join(words)
    
    return processed_text

def preprocess_data(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Preprocess a DataFrame containing text data.

    Args:
        df (pd.DataFrame): Input DataFrame containing a 'text' column.
        column (str): Name of the column to clean.

    Returns:
        pd.DataFrame: Processed DataFrame with an additional 'clean_text' column.
    """
    # Initialize tqdm
    tqdm.pandas()

    # Clean the text
    df['clean_text'] = df[column].apply(clean_text)

    # Preprocess the text
    df['clean_text'] = df['clean_text'].apply(preprocess_text)

    return df