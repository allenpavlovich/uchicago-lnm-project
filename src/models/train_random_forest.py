# src/models/train_random_forest.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import gzip
import logging

# Set up logging
logging.basicConfig(
    filename='training_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def load_bow_features(bow_path, model_path):
    """Load the BoW features and CountVectorizer model from the specified files."""
    with gzip.open(bow_path, 'rb') as f:
        bow_df = pd.read_pickle(f)
    vectorizer = joblib.load(model_path)
    return bow_df, vectorizer

def prepare_features_and_labels(bow_df):
    """Separate the BoW features and labels."""
    logging.info('Preparing features and labels...')
    X = bow_df.drop('label', axis=1)
    y = bow_df['label'].sparse.to_dense()
    non_nan_indices = y.dropna().index
    X = X.loc[non_nan_indices]
    y = y.loc[non_nan_indices]
    logging.info(f'Prepared features and labels. X shape: {X.shape}, y shape: {y.shape}')
    return X, y

def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """Train a Random Forest model on the given features and labels."""
    logging.info('Training Random Forest model...')
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    logging.info('Random Forest model training complete.')
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance on the test dataset."""
    logging.info('Evaluating model...')
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    logging.info(f'Model evaluation complete. Accuracy: {accuracy}')
    logging.info(f'Classification Report:\n{report}')
    logging.info(f'Confusion Matrix:\n{conf_matrix}')
    return accuracy, report, conf_matrix

def save_model(model, model_path):
    """Save the trained Random Forest model to disk."""
    logging.info(f'Saving model to {model_path}...')
    joblib.dump(model, model_path)
    logging.info('Model saved.')

def run_training_pipeline(bow_path, model_path, output_model_path):
    """Full pipeline to load data, train the model, evaluate it, and save the model."""
    logging.info('Starting training pipeline...')
    bow_df, vectorizer = load_bow_features(bow_path, model_path)
    X, y = prepare_features_and_labels(bow_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_random_forest(X_train, y_train)

    accuracy, report, conf_matrix = evaluate_model(model, X_test, y_test)
    logging.info(f"Final Model Accuracy: {accuracy}")

    save_model(model, output_model_path)
    logging.info('Training pipeline complete.')

if __name__ == "__main__":
    run_training_pipeline(
        bow_path='../../src/features/Bow.pkl',
        model_path='../../models/Bow_model.pkl',
        output_model_path='../../models/random_forest_model.pkl'
    )