import joblib
import lime
import lime.lime_text
import numpy as np
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
    df = df.dropna(subset(['clean_text']))
    return df

def explain_with_lime(model, tokenizer, text_instance, max_sequence_length=150):
    """Use LIME to explain a single prediction."""
    lime_explainer = lime.lime_text.LimeTextExplainer(class_names=['Negative', 'Positive'])
    
    # Tokenize and pad the input text
    sequence = tokenizer.texts_to_sequences([text_instance])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_sequence_length)
    
    # LIME expects a function that returns a 2D array with class probabilities
    def predict_fn(texts):
        sequences = tokenizer.texts_to_sequences(texts)
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_sequence_length)
        probabilities = model.predict(padded_sequences)
        return np.hstack([1 - probabilities, probabilities])  # Returns [prob_neg, prob_pos]
    
    explanation = lime_explainer.explain_instance(text_instance, predict_fn, num_features=10)
    explanation.save_to_file('lime_explanation_lstm.html')

def run_lime_explanation(data_path, model_path, tokenizer_path, text_sample):
    """Run the LIME explanation pipeline."""
    df = load_data(data_path)
    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)
    explain_with_lime(model, tokenizer, text_sample)
