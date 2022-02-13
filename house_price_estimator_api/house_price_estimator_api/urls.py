from django.contrib import admin
from django.urls import path
from predictor_api import views
urlpatterns = [
    path('admin/', admin.site.urls),
    path('predict/', views.call_model.as_view())
]
