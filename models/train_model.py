import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle
import re

# 💡 Preprocessing function for corpus cleaning
def clean_text(text):
    """Lowercase, remove special characters, and extra spaces."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove multiple spaces
    return text

# ✅ Load and clean corpus
with open("corpus.txt", "r", encoding="utf-8") as f:
    corpus = f.read().splitlines()

# Clean and filter empty lines
corpus = [clean_text(line) for line in corpus if line.strip()]

# ✅ Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# ✅ Prepare data for training
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

# ✅ Pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# ✅ Features and labels
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# ✅ Build Improved LSTM model with Bidirectional layers
model = Sequential()
model.add(Embedding(total_words, 300, input_length=max_sequence_len - 1))  # Increased embedding dimension
model.add(Bidirectional(LSTM(256, return_sequences=True)))  # Bidirectional for better context
model.add(Dropout(0.3))  # Increased dropout
model.add(Bidirectional(LSTM(256)))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(total_words, activation='softmax'))

# ✅ Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# ✅ Callbacks for better convergence
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001)

# ✅ Train the model
model.fit(X, y, epochs=100, batch_size=64, verbose=1, callbacks=[early_stopping, reduce_lr])

# ✅ Ensure models directory exists
os.makedirs("models", exist_ok=True)

# ✅ Save the model and tokenizer
model.save("models/lstm_model.h5")

# ✅ Save the tokenizer safely
try:
    with open("models/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print("✅ Model and tokenizer saved successfully.")
except Exception as e:
    print(f"❌ Error saving tokenizer: {e}")
