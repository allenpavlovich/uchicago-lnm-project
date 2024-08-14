import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import joblib

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

def get_attention_weights(model, tokenizer, text_sample, max_sequence_length=150):
    """Extract attention weights from the model for a given text sample."""
    sequence = tokenizer.texts_to_sequences([text_sample])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_sequence_length)
    
    # Assuming the model returns attention weights as an output along with predictions
    attention_model = tf.keras.models.Model(inputs=model.input, outputs=[model.output, model.get_layer('attention').output])
    _, attention_weights = attention_model.predict(padded_sequence)
    
    return attention_weights.squeeze(), padded_sequence.squeeze()

def plot_attention_heatmap(attention_weights, sequence, tokenizer):
    """Plot the attention heatmap over the input sequence."""
    words = [word for word, index in tokenizer.index_word.items() if index in sequence]
    attention_weights = attention_weights[-len(words):]  # Adjust for sequence length

    plt.figure(figsize=(10, 2))
    sns.heatmap(np.expand_dims(attention_weights, axis=0), cmap='viridis', annot=True, xticklabels=words, cbar=False)
    plt.title("Attention Weights")
    plt.xlabel("Words")
    plt.show()

def run_attention_visualization(data_path, model_path, tokenizer_path, text_sample):
    """Run the attention visualization pipeline."""
    df = load_data(data_path)
    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)
    attention_weights, sequence = get_attention_weights(model, tokenizer, text_sample)
    plot_attention_heatmap(attention_weights, sequence, tokenizer)

if __name__ == "__main__":
    sample_text = "I love this product, it works great and I would recommend it to everyone."
    run_attention_visualization(
        data_path='../../data/processed/cleaned_data.csv',
        model_path='../../models/lstm_model.h5',
        tokenizer_path='../../models/tokenizer_lstm.pkl',
        text_sample=sample_text
    )
