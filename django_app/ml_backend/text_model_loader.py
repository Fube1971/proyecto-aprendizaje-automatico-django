from pathlib import Path
import json
import re
import pickle
from typing import Tuple

import numpy as np
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Longitud máxima usada en el notebook de texto.
# Se debe colocar el mismo valor que se usó allí (por ejemplo 200).
MAX_SEQUENCE_LENGTH = 200

MODELS_DIR = Path(settings.BASE_DIR).parent / "modelos"

_TEXT_MODEL = None
_TOKENIZER = None
_LABEL_MAP = None


def clean_text(text: str) -> str:
    """
    Aplica una limpieza básica de texto similar a la usada en el notebook:
    - Conversión a minúsculas,
    - Eliminación de caracteres no alfanuméricos básicos,
    - Colapso de espacios.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_text_assets():
    """
    Carga perezosamente el modelo LSTM, el tokenizer y el mapa de etiquetas.
    """
    global _TEXT_MODEL, _TOKENIZER, _LABEL_MAP

    if _TEXT_MODEL is None:
        model_path = MODELS_DIR / "lstm_steam.h5"
        tokenizer_path = MODELS_DIR / "tokenizer_steam.pkl"
        label_map_path = MODELS_DIR / "label_map_steam.json"

        _TEXT_MODEL = load_model(model_path)

        with open(tokenizer_path, "rb") as f:
            _TOKENIZER = pickle.load(f)

        with open(label_map_path, "r", encoding="utf-8") as f:
            _LABEL_MAP = json.load(f)

    return _TEXT_MODEL, _TOKENIZER, _LABEL_MAP


def predict_review(text: str) -> Tuple[str, str, float]:
    """
    Realiza una predicción de sentimiento sobre una reseña de juego.

    Retorna:
        - id de etiqueta ("0" o "1"),
        - descripción de la etiqueta según el label_map,
        - probabilidad estimada de ser 'positivo' (etiqueta 1).
    """
    model, tokenizer, label_map = load_text_assets()

    clean = clean_text(text)
    seq = tokenizer.texts_to_sequences([clean])
    pad = pad_sequences(
        seq,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )

    prob = float(model.predict(pad)[0][0])
    label_id = "1" if prob >= 0.5 else "0"
    label_str = label_map.get(label_id, "desconocido")

    return label_id, label_str, prob
