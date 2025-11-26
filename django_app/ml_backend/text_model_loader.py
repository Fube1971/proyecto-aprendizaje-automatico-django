from pathlib import Path
import json
import pickle
from typing import Tuple
import numpy as np

# BASE_DIR = raíz de django_app
BASE_DIR = Path(__file__).resolve().parent.parent
MODELOS_DIR = BASE_DIR / "modelos"

MODEL_PATH = MODELOS_DIR / "lstm_steam.h5"
TOKENIZER_PATH = MODELOS_DIR / "tokenizer_steam.pkl"
LABEL_MAP_PATH = MODELOS_DIR / "label_map_steam.json"

# Lazy loading
_text_model = None
_tokenizer = None
_label_map = None

MAX_SEQUENCE_LENGTH = 200


def get_model():
    """Carga el modelo LSTM solo la primera vez."""
    global _text_model
    if _text_model is None:
        from tensorflow.keras.models import load_model  # import interno
        _text_model = load_model(MODEL_PATH)
    return _text_model


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        with open(TOKENIZER_PATH, "rb") as f:
            _tokenizer = pickle.load(f)
    return _tokenizer


def get_label_map():
    global _label_map
    if _label_map is None:
        with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            _label_map = json.load(f)
    return _label_map


def predict_review(text: str) -> Tuple[int, str, float]:
    model = get_model()
    tokenizer = get_tokenizer()
    label_map = get_label_map()

    # 1. Texto → secuencia
    seq = tokenizer.texts_to_sequences([text])
    pad = __pad_sequences(seq)

    # 2. Predicción
    prob = float(model.predict(pad)[0][0])

    # 3. Clasificación
    label_id = 1 if prob >= 0.5 else 0
    label_str = label_map.get(str(label_id), "desconocido")

    return label_id, label_str, prob * 100


def __pad_sequences(seq):
    """Import interno para evitar cargar Keras al inicio."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    return pad_sequences(
        seq,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )
