import joblib
import pandas as pd
import matplotlib.pyplot as plt

def load_model_and_vectorizer(model_path, vectorizer_path):
    """Load the trained model and TF-IDF vectorizer."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def plot_top_coefficients(model, vectorizer):
    """Plot the top 5 positive and top 5 negative coefficients."""
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_.flatten()

    # Create a DataFrame of coefficients
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    
    # Get top 5 positive and top 5 negative coefficients
    top_positive = coef_df.nlargest(5, 'Coefficient')
    top_negative = coef_df.nsmallest(5, 'Coefficient')
    
    # Combine the top positive and negative coefficients
    top_coefs = pd.concat([top_positive, top_negative])

    # Plot the coefficients
    plt.figure(figsize=(10, 6))
    top_coefs.plot(kind='barh', x='Feature', y='Coefficient', legend=False, color=['red' if x < 0 else 'blue' for x in top_coefs['Coefficient']])
    plt.title('Top 5 Positive and Negative Coefficients in SVM')
    plt.xlabel('Coefficient')
    plt.ylabel('Feature')
    plt.savefig('../../reports/figures/xai/svm/svm_top_coefficients.png')
    plt.show()

def run_coefficients_analysis(model_path, vectorizer_path):
    """Run the coefficients analysis pipeline."""
    model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)
    plot_top_coefficients(model, vectorizer)
