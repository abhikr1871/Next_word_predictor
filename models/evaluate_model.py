import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model and tokenizer
model_path = "models/lstm_model.keras"  # Updated to the recommended Keras format
tokenizer_path = "models/tokenizer.pkl"

model = load_model(model_path)

with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

# Function to predict the next word
def predict_next_word(seed_text, max_len):
    """Predict the next word for the given seed text."""
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_len - 1, padding='pre')
    
    predicted_index = np.argmax(model.predict(token_list, verbose=0))
    return next((word for word, index in tokenizer.word_index.items() if index == predicted_index), "")

# Test cases
test_sentences = [
    "I love",
    "Artificial Intelligence is",
    "Machine learning is"
]

print("\nModel Predictions:")
for sentence in test_sentences:
    print(f"{sentence} → {predict_next_word(sentence, model.input_shape[1])}")
