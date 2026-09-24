import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load and clean text
with open("corpus.txt", "r", encoding="utf8") as file:
    lines = file.readlines()

data = ' '.join(lines)
data = data.replace('\n', '').replace('\r', '').replace('\ufeff', '').replace('“','').replace('”','')
data = data.split()
data = ' '.join(data)

# Tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

# Save tokenizer
with open('new_fine.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Create sequences
sequence_data = tokenizer.texts_to_sequences([data])[0]
vocab_size = len(tokenizer.word_index) + 1
sequences = []

for i in range(3, len(sequence_data)):
    words = sequence_data[i-3:i+1]
    sequences.append(words)

print("The length of sequences is:", len(sequences))
sequences = np.array(sequences)

# Split into input (X) and target (y)
X = sequences[:, 0:3]
y = sequences[:, 3]
y = to_categorical(y, num_classes=vocab_size)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model architecture
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=3))
model.add(LSTM(1000, return_sequences=True))
model.add(LSTM(1000))
model.add(Dense(1000, activation="relu"))
model.add(Dense(vocab_size, activation="softmax"))

model.build(input_shape=(None, 3))
model.summary()

# Checkpoint callback
checkpoint = ModelCheckpoint("new_fine.h5", monitor='loss', verbose=1, save_best_only=True)

# Compile and train
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
model.fit(X_train, y_train, epochs=20, batch_size=64, callbacks=[checkpoint])

# Evaluate on test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")
