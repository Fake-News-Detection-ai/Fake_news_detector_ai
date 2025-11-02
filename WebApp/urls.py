from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='Home'),
    path('predict/', views.predict, name='predict'),
    path('api/predict/', views.api_predict, name='api_predict'),
]
