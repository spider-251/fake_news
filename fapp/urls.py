from django.urls import path

from . import views

app_name = 'fapp'

urlpatterns = [
    path('', views.homepage, name='homepage'),
]