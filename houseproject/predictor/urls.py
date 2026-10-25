from django.urls import path
from django.contrib.auth import views as auth_views
from . import views
from .forms import LoginForm

urlpatterns = [
    path("", views.home, name="home"),
    path("predict/", views.predict_price, name="predict"),
    path("history/", views.history, name="history"),
    path("history/delete/<int:pk>/", views.delete_prediction, name="delete_prediction"),
    path("insights/", views.insights, name="insights"),
    path("register/", views.register, name="register"),
    path("login/", auth_views.LoginView.as_view(
        template_name="login.html", authentication_form=LoginForm,
        redirect_authenticated_user=True), name="login"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
]