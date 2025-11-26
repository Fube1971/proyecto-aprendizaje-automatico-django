from pathlib import Path
import json
import pickle
from typing import Tuple
import numpy as np
import tensorflow.lite as tflite  # ✔ TFLite (liviano, seguro para Render)

# BASE_DIR = raíz de django_app
BASE_DIR = Path(__file__).resolve().parent.parent
MODELOS_DIR = BASE_DIR / "modelos"

MODEL_PATH = MODELOS_DIR / "lstm_steam.tflite"
TOKENIZER_PATH = MODELOS_DIR / "tokenizer_steam.pkl"
LABEL_MAP_PATH = MODELOS_DIR / "label_map_steam.json"

# --- Lazy Loading --- #
_INTERPRETER = None
_INPUT_DETAILS = None
_OUTPUT_DETAILS = None
_TOKENIZER = None
_LABEL_MAP = None

MAX_SEQUENCE_LENGTH = 200


def load_tflite_lstm():
    """Carga el modelo LSTM TFLite solo una vez."""
    global _INTERPRETER, _INPUT_DETAILS, _OUTPUT_DETAILS

    if _INTERPRETER is None:
        interpreter = tflite.Interpreter(model_path=str(MODEL_PATH))
        interpreter.allocate_tensors()

        _INTERPRETER = interpreter
        _INPUT_DETAILS = interpreter.get_input_details()
        _OUTPUT_DETAILS = interpreter.get_output_details()

    return _INTERPRETER, _INPUT_DETAILS, _OUTPUT_DETAILS


def load_tokenizer():
    """Carga tokenizer solo una vez."""
    global _TOKENIZER
    if _TOKENIZER is None:
        with open(TOKENIZER_PATH, "rb") as f:
            _TOKENIZER = pickle.load(f)
    return _TOKENIZER


def load_label_map():
    """Carga el diccionario id → etiqueta."""
    global _LABEL_MAP
    if _LABEL_MAP is None:
        with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            _LABEL_MAP = json.load(f)
    return _LABEL_MAP


def pad_sequence(text: str):
    """Convierte texto a secuencia padded (igual que en el .h5)."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences  # ✔ import interno
    tokenizer = load_tokenizer()
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(
        seq,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
        dtype="float32"
    )
    return padded


def predict_review(text: str) -> Tuple[int, str, float]:
    """
    Retorna:
      - label_id (0 ó 1)
      - label_name (string)
      - probabilidad % de review positiva
    """

    interpreter, input_details, output_details = load_tflite_lstm()
    label_map = load_label_map()

    # Preprocesar texto
    padded = pad_sequence(text).astype(np.float32)

    # Enviar datos al modelo
    interpreter.set_tensor(input_details[0]["index"], padded)
    interpreter.invoke()

    # Obtener predicción del modelo
    prob = float(interpreter.get_tensor(output_details[0]["index"])[0][0])

    # Clasificación (igual que en el notebook)
    label_id = 1 if prob >= 0.5 else 0
    label_str = label_map.get(str(label_id), "desconocido")

    return label_id, label_str, prob * 100
