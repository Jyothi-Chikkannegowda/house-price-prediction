from django.contrib import admin
from .models import Prediction

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ("user", "location", "sqft", "bhk", "bath", "price_lakhs", "created_at")
    list_filter = ("location", "user")