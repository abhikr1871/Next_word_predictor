# Configuration file for backend

# Default model currently used for prediction.
# BERT can be added later in MODEL_MAP without breaking the app.
DEFAULT_MODEL = "new_fine"

MODEL_MAP = {
    "new_fine": {
        "model": "../models/new_fine.h5",
        "tokenizer": "../models/new_fine.pkl",
    },
    "new_lstm": {
        "model": "../models/new_lstm.h5",
        "tokenizer": "../models/new_lstm.pkl",
    },
    # Future BERT model can be added here when ready:
    # "bert": {"model": "../models/bert_model.h5", "tokenizer": "../models/bert_tokenizer.pkl"}
}

MODEL_CHOICES = list(MODEL_MAP.keys())

# Flask server settings
HOST = "127.0.0.1"
PORT = 5000
