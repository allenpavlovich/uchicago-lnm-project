# src/models/train_lstm.py

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import joblib

def load_data(file_path):
    """Load the cleaned data from the specified file."""
    return pd.read_csv(file_path)

def tokenize_and_pad(df, max_num_words=10000, max_sequence_length=150):
    """Tokenize the text data and pad sequences."""
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(df['clean_text'])
    sequences = tokenizer.texts_to_sequences(df['clean_text'])
    X = pad_sequences(sequences, maxlen=max_sequence_length)
    y = df['label'].values
    return X, y, tokenizer

def build_lstm_model(embedding_dim=128, max_sequence_length=150):
    """Build the LSTM model."""
    model = Sequential([
        Embedding(input_dim=10000, output_dim=embedding_dim, input_length=max_sequence_length),
        LSTM(128, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_lstm_model(X, y, model, batch_size=256, epochs=50):
    """Train the LSTM model with early stopping."""
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[early_stopping])
    return model, history

def evaluate_model(model, X, y):
    """Evaluate the model's performance on the given dataset."""
    y_pred = (model.predict(X) > 0.5).astype("int32")
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)
    return accuracy, report

def save_model_and_tokenizer(model, tokenizer, model_path, tokenizer_path):
    """Save the trained model and tokenizer to disk."""
    model.save(model_path)
    joblib.dump(tokenizer, tokenizer_path)

def run_training_pipeline(data_path, model_path, tokenizer_path):
    """Full pipeline to load data, train the model, evaluate it, and save artifacts."""
    df = load_data(data_path)
    df = df.dropna(subset=['clean_text'])
    X, y, tokenizer = tokenize_and_pad(df)
    model = build_lstm_model()
    model, history = train_lstm_model(X, y, model)
    accuracy, report = evaluate_model(model, X, y)
    print(f"Model Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")
    save_model_and_tokenizer(model, tokenizer, model_path, tokenizer_path)

if __name__ == "__main__":
    run_training_pipeline(
        data_path='../../data/processed/cleaned_data.csv',
        model_path='../../models/lstm_model.h5',
        tokenizer_path='../../models/tokenizer_lstm.pkl'
    )
