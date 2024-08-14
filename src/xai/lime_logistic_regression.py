import joblib
import lime
import lime.lime_text
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

def explain_with_lime(model, vectorizer, text_sample, num_features=10):
    """Apply LIME to explain the model predictions."""
    explainer = lime.lime_text.LimeTextExplainer(class_names=['Negative', 'Positive'])

    # LIME expects the raw text, so we create a prediction function
    predict_fn = lambda x: model.predict_proba(vectorizer.transform(x)).astype(float)
    explanation = explainer.explain_instance(text_sample, predict_fn, num_features=num_features)

    # Save LIME explanation as an HTML file
    explanation.save_to_file('../../reports/figures/xai/log_reg/lime_explanation_logreg.html')

def run_lime_explanation(data_path, model_path, vectorizer_path, text_sample):
    """Run the LIME explanation pipeline."""
    df = load_data(data_path)
    model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)
    explain_with_lime(model, vectorizer, text_sample)