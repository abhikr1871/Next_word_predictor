import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import re
from sklearn.model_selection import train_test_split  # Import train_test_split

#  Preprocessing function for corpus cleaning
def clean_text(text):
    """Lowercase, remove special characters, and extra spaces."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove multiple spaces
    return text

#  Load and clean corpus
with open("corpus.txt", "r", encoding="utf-8") as f:
    corpus = f.read().split('.')

# Clean and filter empty lines
corpus = [clean_text(line) for line in corpus if line.strip()]

#  Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# Prepare data for training
input_sequences = []
sequences = tokenizer.texts_to_sequences(corpus)
for token_list in sequences:
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Features (X) and labels (y)
X = input_sequences[:, :-1]  # All elements except last one
y = input_sequences[:, -1]   # Last element as label

# Convert labels to one-hot encoding
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Build Improved LSTM model with Bidirectional layers
model = Sequential([
    Embedding(total_words, 300, input_length=max_sequence_len - 1),  # Increased embedding dimension
    Bidirectional(LSTM(256, return_sequences=True)),  # Bidirectional for better context
    Dropout(0.3),  # Increased dropout
    Bidirectional(LSTM(256)),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(total_words, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#  Train the model
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    verbose=1,
    validation_data=(X_test, y_test),  # Track validation accuracy during training
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),  # Monitor validation loss
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
    ]
)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {test_accuracy:.4f}")


if test_accuracy= 0.80:
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Save the model
    model.save("lstm_model.keras")

    # Save the tokenizer
    try:
        with open("tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        print("Model and tokenizer saved successfully.")
    except Exception as e:
        print(f"Error saving tokenizer: {e}")
else:
    print("Model accuracy is below 80%, retraining is recommended.")
