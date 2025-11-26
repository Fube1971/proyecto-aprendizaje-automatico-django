from pathlib import Path
import json
import pickle
from typing import Tuple

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# BASE_DIR = raíz de django_app
BASE_DIR = Path(__file__).resolve().parent.parent
MODELOS_DIR = BASE_DIR / "modelos"

MODEL_PATH = MODELOS_DIR / "lstm_steam.h5"
TOKENIZER_PATH = MODELOS_DIR / "tokenizer_steam.pkl"
LABEL_MAP_PATH = MODELOS_DIR / "label_map_steam.json"

# --- Carga en memoria una sola vez --- #
text_model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)

# Debe coincidir con el notebook
MAX_SEQUENCE_LENGTH = 200


def predict_review(text: str) -> Tuple[int, str, float]:
    """
    Recibe un texto en inglés y devuelve:
    - label_id: 0 (no recomendado) o 1 (recomendado)
    - label_str: descripción en español según label_map
    - prob_percent: probabilidad (0–100) de que sea RECOMENDADO
    """

    # 1) Texto -> secuencia
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(
        seq,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )

    # 2) Modelo -> prob (sigmoide, 0–1)
    prob = float(text_model.predict(pad)[0][0])

    # 3) Umbral EXACTAMENTE como en el notebook
    label_id = 1 if prob >= 0.5 else 0

    # 4) Traducimos a string usando el mismo label_map del notebook
    label_str = label_map.get(str(label_id), "desconocido")

    # 5) Convertimos a porcentaje para la UI
    prob_percent = prob * 100.0

    return label_id, label_str, prob_percent
