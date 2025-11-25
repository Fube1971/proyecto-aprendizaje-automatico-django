from pathlib import Path
import json
from typing import Tuple, List

import numpy as np
from PIL import Image

from django.conf import settings
from tensorflow.keras.models import load_model


# Tamaño de imagen que se usó en el notebook de la CNN.
# Si en el notebook se usó otro (por ejemplo 224), se debe ajustar este valor.
IMG_SIZE = 128

# Ruta a la carpeta de modelos; se asume que 'modelos/' está
# un nivel por encima de 'django_app/'.
MODELS_DIR = Path(settings.BASE_DIR).parent / "modelos"

_CNN_MODEL = None
_CLASS_NAMES: List[str] = []


def load_image_assets():
    """
    Carga perezosamente el modelo de la CNN de animales y los nombres de clase.
    Esta función solo se ejecuta la primera vez; después reutiliza los objetos.
    """
    global _CNN_MODEL, _CLASS_NAMES

    if _CNN_MODEL is None:
        model_path = MODELS_DIR / "cnn_animales.h5"
        class_names_path = MODELS_DIR / "class_names_animales.json"

        # Se carga el modelo Keras entrenado previamente.
        _CNN_MODEL = load_model(model_path)

        # Se cargan los nombres de las clases en el mismo orden que en el entrenamiento.
        with open(class_names_path, "r", encoding="utf-8") as f:
            _CLASS_NAMES = json.load(f)

    return _CNN_MODEL, _CLASS_NAMES


def preprocess_image(file_obj) -> np.ndarray:
    """
    Aplica el mismo preprocesamiento usado en el notebook de la CNN:
    - Se abre la imagen,
    - Se convierte a RGB,
    - Se redimensiona a IMG_SIZE x IMG_SIZE,
    - Se normaliza al rango [0, 1],
    - Se añade un eje batch.

    Parámetros
    ----------
    file_obj : archivo subido desde el formulario (InMemoryUploadedFile).

    Retorna
    -------
    np.ndarray
        Arreglo de forma (1, IMG_SIZE, IMG_SIZE, 3) listo para pasar al modelo.
    """
    img = Image.open(file_obj).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_animal_image(file_obj) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    Realiza una predicción con la CNN de animales a partir de una imagen subida.

    Retorna:
        - nombre de la clase top-1,
        - probabilidad asociada,
        - lista con el top-3 [(clase, prob), ...].
    """
    model, class_names = load_image_assets()
    x = preprocess_image(file_obj)

    # Predicción de probabilidades para cada clase.
    probs = model.predict(x)[0]  # vector de longitud n_clases
    probs = probs.astype(float)

    # Top-1
    top1_idx = int(np.argmax(probs))
    top1_class = class_names[top1_idx]
    top1_prob = float(probs[top1_idx])

    # Top-3
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [(class_names[i], float(probs[i])) for i in top3_idx]

    return top1_class, top1_prob, top3
