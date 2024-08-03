# src/data/load_raw_data.py

import pandas as pd
import os
import logging
from typing import Optional

def load_sentiment140_raw(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load the Sentiment140 dataset from a raw CSV file.
    
    Args:
        file_path (str): Path to the raw CSV file.
    
    Returns:
        Optional[pd.DataFrame]: DataFrame containing the dataset, or None if an error occurs.
    """
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return None

    try:
        # Load the dataset
        df = pd.read_csv(file_path, encoding='latin-1', header=None, 
                         names=['label', 'id', 'date', 'query', 'user', 'text'])

        # Map sentiment labels to 0 and 1
        df['label'] = df['label'].map({0: 0, 4: 1})

        # Drop unnecessary columns
        df = df[['text', 'label']]

        return df
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        return None

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    file_path = 'data/raw/training.1600000.processed.noemoticon.csv'
    df = load_sentiment140_raw(file_path)

    if df is not None:
        # Display basic information about the DataFrame
        print(df.info())
        print(df.head())
    else:
        logging.error("Failed to load the dataset.")