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
from sklearn.model_selection import train_test_split

#  Preprocessing function for corpus cleaning
def clean_text(text):
    """Lowercase, remove special characters, and extra spaces."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove multiple spaces
    return text

#  Load and clean corpus
with open("corpus.txt", "r", encoding="utf-8") as f:
    corpus = f.read().splitlines()

# Clean and filter empty lines
corpus = [clean_text(line) for line in corpus if line.strip()]

#  Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

#  Prepare data for training
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

#  Pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

#  Features and labels
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

#  Split data into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#  Build Improved LSTM model with Bidirectional layers
model = Sequential([
    Embedding(total_words, 300, input_length=max_sequence_len - 1),
    Bidirectional(LSTM(256, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(256)),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(total_words, activation='softmax')
])

#  Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#  Callbacks for better convergence
early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)

#  Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1, 
                    validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr])

#  Ensure accuracy is at least 80%
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

if test_acc >= 0.80:
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Save the model
    model.save("models/lstm_model.keras")

    # Save the tokenizer
    try:
        with open("models/tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        print("Model and tokenizer saved successfully.")
    except Exception as e:
        print(f"Error saving tokenizer: {e}")
else:
    print("Model accuracy is below 80%, retraining is recommended.")
