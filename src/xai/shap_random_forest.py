import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

def load_model_and_vectorizer(model_path, vectorizer_path):
    """Load the trained model and vectorizer."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def load_data(data_path):
    """Load the cleaned data."""
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['clean_text'])
    return df

def explain_with_shap(model, X, vectorizer):
    """Apply SHAP to explain the model predictions."""
    X_dense = X.toarray()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_dense)

    # SHAP summary plot
    shap.summary_plot(shap_values, features=X_dense, feature_names=vectorizer.get_feature_names_out())
    plt.savefig('../../reports/figures/xai/rand_forst/shap_summary_rand_forst.png')

    # SHAP feature importance plot
    shap.summary_plot(shap_values, features=X_dense, feature_names=vectorizer.get_feature_names_out(), plot_type='bar')
    plt.savefig('../../reports/figures/xai/rand_forst/shap_feature_importance_rand_forst.png')

def run_shap_explanation(data_path, model_path, vectorizer_path):
    """Run the SHAP explanation pipeline."""
    df = load_data(data_path)
    model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)
    X = vectorizer.transform(df['clean_text'])
    explain_with_shap(model, X, vectorizer)