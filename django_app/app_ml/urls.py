from django.urls import path
from . import views

app_name = "app_ml"

urlpatterns = [
    path("", views.home, name="home"),
    path("imagen/", views.predict_image_view, name="predict_image"),
    path("texto/", views.predict_text_view, name="predict_text"),
    path("modelos/", views.about_models, name="about_models"),
]
