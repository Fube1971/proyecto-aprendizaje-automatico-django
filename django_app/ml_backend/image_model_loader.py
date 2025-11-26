from pathlib import Path
import json
from typing import Tuple, List

import numpy as np
from PIL import Image
from django.conf import settings

# ⚠️ IMPORTANTE:
# NO importamos TensorFlow aquí, porque Render explota la RAM.
# Lo importamos dentro de la función cuando realmente lo necesitemos.

# Tamaño de imagen usado en el entrenamiento
IMG_SIZE = 128

# Carpeta donde están tus modelos
MODELS_DIR = Path(settings.BASE_DIR).parent / "modelos"

# Variables globales de carga diferida
_CNN_MODEL = None
_CLASS_NAMES: List[str] = []


def load_image_assets():
    """
    Carga perezosamente el modelo y los nombres de clase.
    Render no puede cargar TensorFlow al iniciar, por eso la carga se hace aquí,
    y únicamente la primera vez.
    """
    global _CNN_MODEL, _CLASS_NAMES

    if _CNN_MODEL is None:
        model_path = MODELS_DIR / "cnn_animales.h5"
        class_names_path = MODELS_DIR / "class_names_animales.json"

        # ⚠️ IMPORTAR TENSORFLOW AQUÍ (lazy import)
        from tensorflow.keras.models import load_model

        # Cargar modelo
        _CNN_MODEL = load_model(model_path)

        # Cargar nombres de clases
        with open(class_names_path, "r", encoding="utf-8") as f:
            _CLASS_NAMES = json.load(f)

    return _CNN_MODEL, _CLASS_NAMES


def preprocess_image(file_obj) -> np.ndarray:
    """
    Preprocesa la imagen igual que en el entrenamiento:
    RGB, resize, normalización y batch dimension.
    """
    img = Image.open(file_obj).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_animal_image(file_obj) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    Realiza una predicción y devuelve:
        top1_class, top1_prob, top3
    """
    model, class_names = load_image_assets()
    x = preprocess_image(file_obj)

    # Predicción
    probs = model.predict(x)[0]  # vector de probabilidades
    probs = probs.astype(float)

    # Top 1
    top1_idx = int(np.argmax(probs))
    top1_class = class_names[top1_idx]
    top1_prob = float(probs[top1_idx])

    # Top 3
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [(class_names[i], float(probs[i])) for i in top3_idx]

    return top1_class, top1_prob, top3
