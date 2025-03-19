import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model and tokenizer
model = load_model("models/lstm_model.h5")
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Function to predict the next word
def predict_next_word(seed_text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    
    predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
    
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            return word
    return ""

# Test the model
test_sentences = [
    "I love",
    "Artificial Intelligence is",
    "Machine learning is"
]

print("\nModel Evaluation:")
for sentence in test_sentences:
    next_word = predict_next_word(sentence, model.input_shape[1])
    print(f"{sentence} → {next_word}")
