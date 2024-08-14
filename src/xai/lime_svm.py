import joblib
import lime
import lime.lime_text
import numpy as np
import pandas as pd

def load_model_and_vectorizer(model_path, vectorizer_path):
    """Load the trained model and TF-IDF vectorizer."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def load_data(data_path):
    """Load the cleaned data."""
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['clean_text'])
    return df

def explain_with_lime(model, vectorizer, text_instance):
    """Use LIME to explain a single prediction."""
    lime_explainer = lime.lime_text.LimeTextExplainer(class_names=['Negative', 'Positive'])
    X = vectorizer.transform([text_instance])
    
    # Create a custom predict function that outputs probabilities
    def predict_fn(texts):
        decision_scores = model.decision_function(vectorizer.transform(texts))
        return np.array([1 - decision_scores, decision_scores]).T

    # Generate LIME explanation
    explanation = lime_explainer.explain_instance(text_instance, predict_fn, num_features=10)
    explanation.save_to_file('../../reports/figures/xai/svm/lime_explanation_svm.html')

def run_lime_explanation(data_path, model_path, vectorizer_path, text_sample):
    """Run the LIME explanation pipeline."""
    df = load_data(data_path)
    model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)
    explain_with_lime(model, vectorizer, text_sample)