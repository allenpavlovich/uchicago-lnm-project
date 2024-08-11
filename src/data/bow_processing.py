# src/data/bow_processing.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os

def create_bow_features(input_csv, max_features=1000):
    """
    Creates Bag of Words features from the cleaned text data and adds sentiment labels.

    Args:
        input_csv: Path to the CSV file containing cleaned text data.
        max_features: Maximum number of features (words) to include.

    Returns:
        bow_df: DataFrame with BoW features and sentiment labels.
        cv: The trained CountVectorizer model.
    """
    # Load the cleaned data
    df = pd.read_csv(input_csv)

    # Drop NaN values
    df = df.dropna(subset=['clean_text'])
    
    # Initialize CountVectorizer
    cv = CountVectorizer(max_features=max_features)

    # Fit and transform the cleaned text data to create BoW features
    X = cv.fit_transform(df['clean_text'])  # Keep X as a sparse matrix

    # Print the shape of the BoW matrix
    print("Shape of the BoW matrix:", X.shape)

    # Convert to DataFrame with sparse columns
    bow_df = pd.DataFrame.sparse.from_spmatrix(X, columns=cv.get_feature_names_out())

    # Add the sentiment labels to the DataFrame
    bow_df['label'] = pd.Series(df['label'].values, dtype='Sparse[int]')

    return bow_df, cv

def save_bow_features(bow_df, bow_model, output_path, model_path):
    """
    Saves the BoW features and CountVectorizer model as pickle files.

    Args:
        bow_df: DataFrame with BoW features and sentiment labels.
        bow_model: The trained CountVectorizer model.
        output_path: Path to save the BoW features as a pickle file.
        model_path: Path to save the CountVectorizer model as a pickle file.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save the BoW features as a compressed pickle file
    bow_df.to_pickle(output_path, compression='gzip')
    print(f"BoW features saved to {output_path}")

    # Save the CountVectorizer model as a pickle file
    with open(model_path, 'wb') as model_file:
        pickle.dump(bow_model, model_file)
    print(f"BoW model saved to {model_path}")

def load_bow_features(bow_path, model_path):
    """
    Loads the BoW features and CountVectorizer model from pickle files.

    Args:
        bow_path: Path to the pickle file with BoW features.
        model_path: Path to the pickle file with the CountVectorizer model.

    Returns:
        bow_df: DataFrame with BoW features and sentiment labels.
        cv: The CountVectorizer model.
    """
    # Load the BoW features
    bow_df = pd.read_pickle(bow_path)
    print(f"BoW features loaded from {bow_path}")

    # Load the CountVectorizer model
    with open(model_path, 'rb') as model_file:
        cv = pickle.load(model_file)
    print(f"BoW model loaded from {model_path}")

    return bow_df, cv

# Main function to create and save BoW features and model
def process_and_save_bow(input_csv, bow_output_path, model_output_path, max_features=1000):
    """
    Processes the input CSV to create BoW features and saves the results.

    Args:
        input_csv: Path to the CSV file containing cleaned text data.
        bow_output_path: Path to save the BoW features as a pickle file.
        model_output_path: Path to save the CountVectorizer model as a pickle file.
        max_features: Maximum number of features (words) to include.
    """
    bow_df, cv = create_bow_features(input_csv, max_features)
    save_bow_features(bow_df, cv, bow_output_path, model_output_path)

if __name__ == "__main__":
    input_csv = '../../data/processed/cleaned_data.csv'
    bow_output_path = '../../src/features/Bow.pkl'
    model_output_path = '../../models/Bow_model.pkl'
    process_and_save_bow(input_csv, bow_output_path, model_output_path)