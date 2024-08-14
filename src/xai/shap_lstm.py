import joblib
import shap
import pandas as pd
import tensorflow as tf

def load_model_and_tokenizer(model_path, tokenizer_path):
    """Load the trained LSTM model and tokenizer."""
    model = tf.keras.models.load_model(model_path)
    tokenizer = joblib.load(tokenizer_path)
    return model, tokenizer

def load_data(data_path):
    """Load the cleaned data."""
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['clean_text'])
    return df

def explain_with_shap(model, tokenizer, df, max_sequence_length=150, num_samples=200):
    """Use SHAP to explain the model predictions on a sample of data."""
    # Tokenize and pad sequences
    X = tokenizer.texts_to_sequences(df['clean_text'].iloc[:num_samples])
    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_sequence_length)
    
    # Ensure the SHAP values are computed correctly
    def model_predict(input_data):
        return model.predict(input_data).flatten()

    # Create SHAP explainer
    explainer = shap.KernelExplainer(model_predict, X)
    shap_values = explainer.shap_values(X, nsamples=num_samples)

    # Map indices to actual words
    word_index = {v: k for k, v in tokenizer.word_index.items()}
    
    # Update the feature_names to match the shape of X
    feature_names = []
    for i in range(max_sequence_length):
        word = word_index.get(i + 1, f"word{i + 1}")  # Adding 1 to match the 1-based index
        feature_names.append(word)

    # Generate the SHAP summary plot
    shap.summary_plot(shap_values, X, feature_names=feature_names)
    shap.save_html('shap_summary.html', shap_values)

def run_shap_explanation(data_path, model_path, tokenizer_path):
    """Run the SHAP explanation pipeline."""
    df = load_data(data_path)
    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)
    explain_with_shap(model, tokenizer, df)