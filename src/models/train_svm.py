# src/models/train_svm.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_data(file_path):
    """Load the cleaned data from the specified file."""
    return pd.read_csv(file_path)

def vectorize_text(df, max_features=1000):
    """Perform TF-IDF vectorization on the 'clean_text' column."""
    tfidf = TfidfVectorizer(max_features=max_features)
    X = tfidf.fit_transform(df['clean_text'])
    return X, tfidf

def train_svm(X, y):
    """Train an SVM model on the given features and labels."""
    model = SVC(kernel='linear', C=1, verbose= True)
    model.fit(X, y)
    return model

def evaluate_model(model, X, y):
    """Evaluate the model's performance on the given dataset."""
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)
    return accuracy, report

def save_model_and_vectorizer(model, vectorizer, model_path, vectorizer_path):
    """Save the trained model and TF-IDF vectorizer to disk."""
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

def run_training_pipeline(data_path, model_path, vectorizer_path):
    """Full pipeline to load data, train the model, evaluate it, and save artifacts."""
    df = load_data(data_path)
    df = df.dropna(subset=['clean_text'])
    X, tfidf = vectorize_text(df)
    y = df['label']
    model = train_svm(X, y)
    accuracy, report = evaluate_model(model, X, y)
    print(f"Model Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")
    save_model_and_vectorizer(model, tfidf, model_path, vectorizer_path)

if __name__ == "__main__":
    run_training_pipeline(
        data_path='../../data/processed/cleaned_data.csv',
        model_path='../../models/svm_model.pkl',
        vectorizer_path='../../models/tfidf_vectorizer_svm.pkl'
    )
