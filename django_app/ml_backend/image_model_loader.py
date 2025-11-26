from pathlib import Path
import json
from typing import Tuple, List

import numpy as np
from PIL import Image
from django.conf import settings
import tensorflow.lite as tflite  # ✔ TFLite correcto

# Tamaño de imagen usado en el entrenamiento
IMG_SIZE = 128

# Carpeta donde están tus modelos
MODELS_DIR = Path(settings.BASE_DIR).parent / "modelos"

# Variables globales (lazy loading)
_INTERPRETER = None
_INPUT_DETAILS = None
_OUTPUT_DETAILS = None
_CLASS_NAMES: List[str] = []


def load_tflite_assets():
    """
    Carga perezosamente el modelo TFLite y los nombres de clase.
    Ahora Render NO se rompe porque no usa TensorFlow pesado.
    """
    global _INTERPRETER, _INPUT_DETAILS, _OUTPUT_DETAILS, _CLASS_NAMES

    if _INTERPRETER is None:
        model_path = MODELS_DIR / "cnn_animales.tflite"
        class_names_path = MODELS_DIR / "class_names_animales.json"

        # ✔ Cargar modelo TFLite (MUY liviano)
        interpreter = tflite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()

        _INTERPRETER = interpreter
        _INPUT_DETAILS = interpreter.get_input_details()
        _OUTPUT_DETAILS = interpreter.get_output_details()

        # Cargar nombres de clases
        with open(class_names_path, "r", encoding="utf-8") as f:
            _CLASS_NAMES = json.load(f)

    return _INTERPRETER, _INPUT_DETAILS, _OUTPUT_DETAILS, _CLASS_NAMES


def preprocess_image(file_obj) -> np.ndarray:
    """
    Preprocesamiento de imagen:
    RGB, resize, normalización y batch dimension.
    """
    img = Image.open(file_obj).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_animal_image(file_obj) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    Realiza predicción usando el modelo TFLite.
    """
    interpreter, input_details, output_details, class_names = load_tflite_assets()
    img_array = preprocess_image(file_obj)

    # ✔ Enviar al modelo TFLite
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    # ✔ Obtener predicción
    probs = interpreter.get_tensor(output_details[0]['index'])[0]
    probs = probs.astype(float)

    # Top 1
    top1_idx = int(np.argmax(probs))
    top1_class = class_names[top1_idx]
    top1_prob = float(probs[top1_idx])

    # Top 3
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [(class_names[i], float(probs[i])) for i in top3_idx]

    return top1_class, top1_prob, top3
