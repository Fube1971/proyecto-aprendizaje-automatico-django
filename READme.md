# 🐾 ML Animal World – Django ML App

Aplicación web en **Django** que muestra dos modelos de aprendizaje automático:

1. 🐶🐱🕊️ **Clasificador de animales (dog / cat / bird)**  
   Red neuronal convolucional (CNN) que clasifica imágenes en tres clases:  
   **perro**, **gato** y **pájaro**.

2. 🎮 **Análisis de sentimiento de reseñas estilo Steam**  
   Modelo de texto basado en **LSTM** que clasifica reseñas en:  
   **“positivo (recomendado)”** o **“negativo (no recomendado)”**.

Incluye además una página de documentación donde se explica **cómo se entrenaron los modelos**, datos usados y decisiones de diseño.

---

## 🚀 1. Requisitos

- **Python** 3.10+ (probado con 3.13 en Windows).
- `pip` actualizado.
- Recomendado: entorno virtual (`venv`).

Las dependencias se instalan desde `requirements.txt`.

---

## 📁 2. Estructura del proyecto (simplificada)

```text
proyecto_final_procesamiento/
├─ django_app/
│  ├─ manage.py
│  ├─ config/                 # Proyecto Django (settings, urls, wsgi)
│  ├─ app_ml/                 # App principal
│  │  ├─ views.py             # Vistas (animal, texto, about)
│  │  ├─ urls.py              # Rutas de la app
│  │  └─ templates/app_ml/
│  │     ├─ base.html
│  │     ├─ home.html
│  │     ├─ predict_image.html   # Clasificador dog/cat/bird
│  │     ├─ predict_text.html    # Reseñas estilo Steam
│  │     └─ about_models.html    # Explicación de los modelos
│  ├─ ml_backend/
│  │  ├─ image_model_loader.py   # Carga modelo de imágenes
│  │  └─ text_model_loader.py    # Carga modelo de texto
│  └─ modelos/
│     ├─ cnn_animales.h5
│     ├─ class_names_animales.json
│     └─ steam_lstm.h5 (según el nombre que hayas exportado)
└─ notebooks/
   ├─ 00_preparar_animalitos.ipynb
   ├─ 01_cnn_animales_imagenes.ipynb
   └─ 02_rnn_lstm_steam.ipynb
````

---

## 🧩 3. Instalación y ejecución local

### 3.1. Clonar el repositorio

```bash
git clone <https://github.com/Fube1971/proyecto-aprendizaje-automatico-django>.git
cd proyecto_final_procesamiento
```

### 3.2. Crear y activar entorno virtual

#### Windows (PowerShell)

```bash
python -m venv .venv
.\.venv\Scripts\Activate
```

#### Windows (Git Bash)

```bash
python -m venv .venv
source .venv/Scripts/activate
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3.3. Instalar dependencias

```bash
pip install -r requirements.txt
```

> Asegúrate de ejecutar esto donde esté `requirements.txt`.
> Si está dentro de `django_app/`, haz antes `cd django_app`.

### 3.4. Migraciones de Django

```bash
cd django_app
python manage.py migrate
```

### 3.5. Ejecutar servidor de desarrollo

```bash
python manage.py runserver
```

Abrir en el navegador:

```text
http://127.0.0.1:8000/
```

---

## 🖥️ 4. Uso de la aplicación

### 🏠 Home (`/`)

Página de inicio con explicación general y accesos directos a:

* Animal Classifier
* Steam Review Quest
* Meet the Models

### 🐶🐱🕊️ Animal Classifier (`/imagen/`)

* Sube una imagen donde aparezca **un perro, un gato o un pájaro**.
* El modelo devuelve:

  * Clase predicha.
  * Confianza (%).
  * Top-3 de clases con sus probabilidades.

> ⚠️ El modelo **solo** está entrenado para dog/cat/bird; otros animales no son fiables.

### 🎮 Steam Review Quest (`/texto/`)

* Escribe una reseña de videojuego en **inglés** o usa un ejemplo aleatorio.
* El modelo devuelve:

  * **Recommended** / **Not recommended**.
  * Confianza (%).
  * Etiqueta original del modelo: `positivo (recomendado)` o `negativo (no recomendado)`.

### 🧠 Meet the Models (`/modelos/`)

Página donde se explica:

* Cómo se construyó el dataset de animales.
* Cómo se entrenó la CNN.
* Cómo se procesan las reseñas de Steam y se entrenó la LSTM.
* Resultados (accuracy y comportamiento en test).

---

## 🧠 5. Detalle de los modelos de ML

### 5.1. Clasificador de imágenes (dog / cat / bird)

**Notebook:** `01_cnn_animales_imagenes.ipynb`
**Dataset preparado en:** `00_preparar_animalitos.ipynb`

#### 🐾 Dataset

* Origen: datasets de **perros/gatos** y de **pájaros** (Kaggle).
* Se construyó un dataset balanceado con **3 clases**:

  * `perro`, `gato`, `pájaro`.
* Cada clase se equilibró a la misma cantidad de imágenes y se dividió en:

  * **70%** entrenamiento
  * **15%** validación
  * **15%** prueba

Según la salida del notebook de preparación:

* Perro  → train: 3507, val: 751, test: 753
* Gato   → train: 3507, val: 751, test: 753
* Pájaro → train: 3507, val: 751, test: 753

#### 🧼 Preprocesamiento

* Imágenes reescaladas a **128×128 píxeles**.
* Normalización al rango **[0, 1]** con `rescale=1.0/255.0`.
* Para entrenamiento se aplicó **data augmentation** con `ImageDataGenerator`:

  * `rotation_range=20`
  * `width_shift_range=0.1`
  * `height_shift_range=0.1`
  * `zoom_range=0.1`
  * `horizontal_flip=True`
* Validación y test solo usan normalización (sin augmentación).

#### 🧱 Arquitecturas evaluadas

Se definieron dos modelos CNN:

1. **Modelo 1 – CNN sencilla**

   * 2 bloques Conv2D + MaxPooling:

     * Conv2D(32, 3×3, ReLU) → MaxPool(2×2)
     * Conv2D(64, 3×3, ReLU) → MaxPool(2×2)
   * `Flatten`
   * `Dense(128, ReLU)`
   * `Dropout(0.5)`
   * `Dense(3, softmax)`

2. **Modelo 2 – CNN más profunda**

   * 3 bloques convolucionales con más filtros:

     * Bloque 1: 2×Conv2D(32, 3×3, ReLU) → MaxPool(2×2)
     * Bloque 2: 2×Conv2D(64, 3×3, ReLU) → MaxPool(2×2)
     * Bloque 3: Conv2D(128, 3×3, ReLU) → MaxPool(2×2)
   * `Flatten`
   * `Dense(256, ReLU)`
   * `Dropout(0.5)`
   * `Dense(3, softmax)`

#### ⚙️ Búsqueda de hiperparámetros

Para cada arquitectura se probaron varias configuraciones:

* **Modelo 1**

  * `1A_Adam_CCE`

    * Optimizador: Adam, `lr=1e-3`
    * Pérdida: `CategoricalCrossentropy`
  * `1B_SGD_CCE`

    * Optimizador: SGD, `lr=1e-2`, `momentum=0.9`
    * Pérdida: `CategoricalCrossentropy`

* **Modelo 2**

  * `2A_AdamLowLR_CCE`

    * Optimizador: Adam, `lr=5e-4`
    * Pérdida: `CategoricalCrossentropy`
  * `2B_AdamLowLR_CCE_LS`

    * Optimizador: Adam, `lr=5e-4`
    * Pérdida: `CategoricalCrossentropy(label_smoothing=0.1)`

Se entrenaron usando la función `train_and_plot`, que genera curvas de loss/accuracy en train y val.
Revisando esas curvas y los resultados en validación se eligió:

* **Modelo final CNN:**
  **Modelo 2** + configuración **`2A_AdamLowLR_CCE`**
  con **15 épocas** (`BEST_EPOCHS = 15`).

#### 📊 Resultados del modelo final (CNN)

En el conjunto de **test**:

* **Test Loss:** 0.3721
* **Test Accuracy:** 0.8490 (≈ 85%)

Reporte de clasificación (resumen):

* Gato   → precision ≈ 0.84, recall ≈ 0.77
* Pájaro → precision ≈ 0.90, recall ≈ 0.97
* Perro  → precision ≈ 0.80, recall ≈ 0.81

Esto justifica el uso de la CNN profunda (Modelo 2) y de **Adam con `lr=5e-4`**, que ofreció:

* Mejor accuracy en validación y prueba.
* Curvas de entrenamiento más estables.
* Menor sobreajuste que configuraciones con learning rate más alto.

---

### 5.2. Clasificador de reseñas de Steam (LSTM)

**Notebook:** `02_rnn_lstm_steam.ipynb`
**Dataset:** Kaggle – *Sentiment Analysis for Steam Reviews* (`train.csv`)

#### 🧾 Dataset y etiquetas

* Columna de texto: `user_review`
* Columna de etiqueta: `user_suggestion`

  * 1 → positivo (recomendado)
  * 0 → negativo (no recomendado)

Se eliminan filas con `NaN` y se trabaja solo con estas dos columnas.

#### 🧼 Limpieza de texto

Función `clean_text`:

* Pasa a minúsculas.
* Elimina URLs.
* Elimina caracteres no alfabéticos (manteniendo puntuación básica).
* Colapsa espacios múltiples.

El texto limpio se guarda en la columna `clean_text`.

#### ✂️ División Train / Val / Test

Usando `train_test_split` estratificado:

* Primero:

  * 15% para **test**
  * 85% para **train+val**
* Luego se divide train+val de forma que el resultado final sea:

  * **70%** train
  * **15%** val
  * **15%** test

#### 🔡 Tokenización y secuencias

* `MAX_NUM_WORDS = 30000` (tamaño máximo de vocabulario).
* `MAX_SEQUENCE_LENGTH = 200` tokens por reseña.
* `Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")`.
* Padding con `pad_sequences(..., maxlen=200, padding="post", truncating="post")`.

#### 🧱 Modelos evaluados

1. **RNN simple**

   * Embedding(128)
   * SimpleRNN(64)
   * Dense(1, sigmoide)

2. **LSTM 1**

   * Embedding(64)
   * LSTM(64)
   * Dense(1, sigmoide)

3. **LSTM 2 (modelo final)**

   * Embedding(128)
   * LSTM(128) con:

     * `dropout = 0.3`
     * `recurrent_dropout = 0.3`
   * Dense(1, sigmoide)

#### ⚙️ Entrenamiento y búsqueda de hiperparámetros

Función genérica `train_text_model`:

* Pérdida: `BinaryCrossentropy(label_smoothing=0.05)`
* Optimizador: `Adam(learning_rate=lr)`
* `batch_size=64`
* Métrica: `accuracy`
* Opción de `EarlyStopping` (usado según configuración).

Configuraciones probadas (resumen):

* **RNN A:** emb=128, units=64, `lr=1e-3`, 10 épocas.
* **RNN B:** emb=64, units=64, `lr=5e-4`, 10 épocas.
* **LSTM 1:** emb=64, units=64, `lr=1e-3`, 5 épocas.
* **LSTM 2:** emb=128, units=128, `lr=1e-4`, 10 épocas.

Se comparan las curvas de entrenamiento, `val_accuracy` y resultados en test.

#### 📊 Resultados y modelo elegido

Del notebook:

* **Mejor `val_accuracy`**:

  * LSTM 1: 0.5931
  * LSTM 2: 0.7760

* **Rendimiento en test:**

  * LSTM 1 → Test Accuracy: 0.5749
  * LSTM 2 → Test Accuracy: 0.7653

El mejor modelo fue:

* **Modelo final de texto:**
  **LSTM 2** – `Embedding(128) + LSTM(128, dropout=0.3, recurrent_dropout=0.3)`
  entrenada con:

  * `lr = 1e-4`
  * `epochs = 10`
  * pérdida: BinaryCrossentropy con *label smoothing* 0.05

Por eso, en la aplicación web se carga este modelo LSTM final:
es más robusto que la RNN simple y que la LSTM pequeña, y alcanza ~**76% de accuracy en test** con un balance razonable entre capacidad y riesgo de sobreajuste.

---

## 🧪 6. Pruebas rápidas

* Probar varias imágenes de **perros, gatos y pájaros** para comprobar que la CNN responde razonablemente.
* Probar reseñas:

  * Extremadamente positivas (“best game ever, amazing story, super polished gameplay…”).
  * Extremadamente negativas (“full of bugs, boring, repetitive, crashes every 5 minutes…”).

Comparar:

* El mensaje **Recommended / Not recommended**.
* El campo **Raw label** que viene directo del modelo (`positivo (recomendado)` / `negativo (no recomendado)`).

---

## 🐛 7. Problemas frecuentes

### `NoReverseMatch` con `home`, `predict_image`, etc.

Comprobar que:

```python
# config/urls.py
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include(("app_ml.urls", "app_ml"), namespace="app_ml")),
]
```

y en `app_ml/urls.py`:

```python
app_name = "app_ml"

urlpatterns = [
    path("", views.home, name="home"),
    path("imagen/", views.predict_image_view, name="predict_image"),
    path("texto/", views.predict_text_view, name="predict_text"),
    path("modelos/", views.about_models, name="about_models"),
]
```

Los templates deben usar `{% url 'app_ml:home' %}`, etc.

---

## 🌐 8. Deploy en Render.com

La aplicación está desplegada públicamente en:

👉 **Demo online:** https://ml-animal-world.onrender.com  

> Render puede “dormir” el servicio si no tiene tráfico.  
> La primera carga puede tardar unos segundos mientras despierta.

### 8.1. Cómo volver a desplegar (fork / clon nuevo)

Si alguien quiere desplegar este proyecto en su propia cuenta de Render:

1. Hacer **fork** del repo o clonarlo en su GitHub.
2. En Render → **New → Web Service** → conectar con su repo.

Configurar:

- **Build Command**

  ```bash
  pip install -r requirements.txt && cd django_app && python manage.py migrate && python manage.py collectstatic --noinput


* **Start Command**

  ```bash
  cd django_app && gunicorn config.wsgi:application --bind 0.0.0.0:$PORT
  ```

### 8.2. Variables de entorno necesarias

En la sección **Environment** del servicio en Render:

* `PYTHON_VERSION` = `3.10.14`
* `DJANGO_SETTINGS_MODULE` = `config.settings`
* `SECRET_KEY` = cadena larga y aleatoria (no se sube al repo)
* `DEBUG` = `False`
* `WEB_CONCURRENCY` = `1`  ← importante para que solo arranque un worker y no se quede sin memoria.

### 8.3. Notas sobre TensorFlow

Para que el despliegue quepa en el plan gratuito de Render se usa:

```txt
tensorflow-cpu>=2.10.0,<3.0
```

en `requirements.txt`, que es la versión solo CPU de TensorFlow (más ligera que la versión con GPU).
Los modelos (`cnn_animales.h5` y `lstm_steam.h5`) se cargan desde `django_app/modelos/` en el arranque.



---

## 📄 9. Licencia

Proyecto desarrollado como trabajo académico de **Aprendizaje Automático** de la Universidad Militar Nueva Granada - por los estudiandtes, Daniela Fuentes, Juan Jose Gutierres y Maria Natalia caro.
