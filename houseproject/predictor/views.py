import os
import json
import joblib
import numpy as np
import pandas as pd
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST

from .forms import RegisterForm
from .models import Prediction

base = os.path.join(os.path.dirname(__file__), "ml_model")
model = joblib.load(os.path.join(base, "house_model.pkl"))
locations = joblib.load(os.path.join(base, "locations.pkl"))

try:
    with open(os.path.join(base, "model_info.json"), encoding="utf-8") as f:
        model_info = json.load(f)
except FileNotFoundError:
    model_info = {}

POPULAR = ["Whitefield", "Electronic City", "Sarjapur Road", "Hebbal",
           "Marathahalli", "Koramangala", "Indira Nagar", "Yelahanka"]


def fmt(lakhs):
    if lakhs >= 100:
        return f"₹{lakhs / 100:.2f} Cr"
    return f"₹{lakhs:.2f} Lakhs"


def predict_many(df):
    return np.expm1(model.predict(df))

def home(request):
    features = [
        ("🎯", "Accurate predictions", "Tuned gradient boosting model trained on log-transformed prices."),
        ("📈", "Trend charts", "See how the price changes with area and across popular locations."),
        ("🔐", "User accounts", "Register, log in and keep your own prediction history."),
        ("🌙", "Dark mode", "Switch between light and dark themes, saved in the browser."),
    ]
    stack = ["Python", "Django", "Scikit-learn", "Pandas", "NumPy", "GridSearchCV",
             "Bootstrap 5", "Chart.js", "SQLite", "HTML/CSS/JS"]
    return render(request, "home.html", {
        "info": model_info, "n_locations": len(locations),
        "features": features, "stack": stack,
    })

def register(request):
    if request.user.is_authenticated:
        return redirect("predict")
    form = RegisterForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = form.save()
        login(request, user)
        return redirect("predict")
    return render(request, "register.html", {"form": form})


@login_required
def predict_price(request):
    ctx = {"locations": locations, "selected": {}, "error": None}

    if request.method == "POST":
        try:
            location = request.POST["location"]
            sqft = float(request.POST["sqft"])
            bath = int(request.POST["bath"])
            bhk = int(request.POST["bhk"])

            if location not in locations or not (300 <= sqft <= 10000) \
                    or not (1 <= bath <= 10) or not (1 <= bhk <= 10):
                raise ValueError("out of range")

            ctx["selected"] = {"location": location, "sqft": int(sqft), "bath": bath, "bhk": bhk}

            one = pd.DataFrame({"location": [location], "total_sqft": [sqft],
                                "bath": [bath], "bhk": [bhk]})
            price = float(predict_many(one)[0])

            areas = sorted({int(sqft * f) for f in (0.6, 0.8, 1.0, 1.2, 1.4)})
            trend_df = pd.DataFrame({"location": [location] * len(areas), "total_sqft": areas,
                                     "bath": [bath] * len(areas), "bhk": [bhk] * len(areas)})
            trend_vals = predict_many(trend_df)

            comp = [l for l in POPULAR if l in locations]
            if location not in comp:
                comp.append(location)
            comp_df = pd.DataFrame({"location": comp, "total_sqft": [sqft] * len(comp),
                                    "bath": [bath] * len(comp), "bhk": [bhk] * len(comp)})
            comp_vals = predict_many(comp_df)

            ctx.update({
                "price": round(price, 2),
                "price_text": fmt(price),
                "low_text": fmt(price * 0.85),
                "high_text": fmt(price * 1.15),
                "per_sqft": f"{price * 100000 / sqft:,.0f}",
                "chart_data": {
                    "price": round(price, 2),
                    "trend_labels": [f"{a} sq ft" for a in areas],
                    "trend_values": [round(float(v), 2) for v in trend_vals],
                    "loc_labels": comp,
                    "loc_values": [round(float(v), 2) for v in comp_vals],
                    "selected_location": location,
                },
            })
        except Exception:
            ctx["error"] = "Please enter valid values."
        else:
            Prediction.objects.create(user=request.user, location=location, sqft=sqft,
                                      bhk=bhk, bath=bath, price_lakhs=round(price, 2))

    ctx["history"] = Prediction.objects.filter(user=request.user)[:5]
    return render(request, "index.html", ctx)


@login_required
def history(request):
    items = Prediction.objects.filter(user=request.user)
    return render(request, "history.html", {"items": items})


@login_required
@require_POST
def delete_prediction(request, pk):
    get_object_or_404(Prediction, pk=pk, user=request.user).delete()
    return redirect("history")


@login_required
def insights(request):
    return render(request, "insights.html", {"info": model_info})