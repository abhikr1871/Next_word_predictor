from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from config import MODEL_PATH, TOKENIZER_PATH

app = Flask(__name__)

# Load model and tokenizer
model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# API route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    seed_text = data['text']
    
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=model.input_shape[1] - 1, padding='pre')
    
    predicted = np.argmax(model.predict(token_list), axis=-1)
    
    next_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            next_word = word
            break
    
    return jsonify({"next_word": next_word})

if __name__ == '__main__':
    app.run(debug=True)
