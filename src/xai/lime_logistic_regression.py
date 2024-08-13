import pandas as pd
import joblib
from lime.lime_text import LimeTextExplainer

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

def explain_with_lime(model, df, vectorizer):
    """Apply LIME to explain individual predictions."""
    explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])
    sample_texts = df['clean_text'].sample(5, random_state=42).tolist()

    for text in sample_texts:
        explanation = explainer.explain_instance(text, model.predict_proba, num_features=10)
        explanation.show_in_notebook(text=text)

def run_lime_explanation(data_path, model_path, vectorizer_path):
    """Run the LIME explanation pipeline."""
    df = load_data(data_path)
    model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)
    explain_with_lime(model, df, vectorizer)