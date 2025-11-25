from django.shortcuts import render
from django.http import HttpRequest, HttpResponse

from ml_backend.image_model_loader import predict_animal_image
from ml_backend.text_model_loader import predict_review


def home(request: HttpRequest) -> HttpResponse:
    """
    Vista principal de la aplicación.
    Presenta un menú sencillo con enlaces a las demás funcionalidades.
    """
    return render(request, "app_ml/home.html")


def predict_image_view(request: HttpRequest) -> HttpResponse:
    """
    Vista para la clasificación de imágenes.
    Permite subir una imagen de animal (perro/gato/pájaro) y muestra la predicción
    de la CNN entrenada, junto con la probabilidad y el top-3 de clases.
    """
    context = {
        "prediction": None,
        "top3": None,
        "error": None,
    }

    if request.method == "POST":
        file_obj = request.FILES.get("image")

        if not file_obj:
            context["error"] = "Por favor, selecciona una imagen."
        else:
            try:
                clase, prob, top3 = predict_animal_image(file_obj)
                context["prediction"] = {
                    "class": clase,
                    "prob": prob,
                }
                context["top3"] = top3
            except Exception as e:
                context["error"] = f"Error al procesar la imagen: {e}"

    return render(request, "app_ml/predict_image.html", context)


def predict_text_view(request: HttpRequest) -> HttpResponse:
    """
    Vista para la clasificación de texto.
    Permite ingresar una reseña de videojuego y muestra la predicción de
    sentimiento según la LSTM entrenada, junto con la probabilidad.
    """
    context = {
        "text": "",
        "prediction": None,
        "error": None,
    }

    if request.method == "POST":
        text = request.POST.get("review", "").strip()
        context["text"] = text

        if not text:
            context["error"] = "Por favor, escribe una reseña."
        else:
            try:
                label_id, label_str, prob = predict_review(text)
                context["prediction"] = {
                    "label_id": label_id,
                    "label_str": label_str,
                    "prob": prob,
                }
            except Exception as e:
                context["error"] = f"Error al procesar el texto: {e}"

    return render(request, "app_ml/predict_text.html", context)


def about_models(request: HttpRequest) -> HttpResponse:
    """
    Vista informativa que describe brevemente los modelos, los datos usados
    y algunas de sus métricas y limitaciones.
    El contenido principal se coloca en la plantilla HTML.
    """
    return render(request, "app_ml/about_models.html")
