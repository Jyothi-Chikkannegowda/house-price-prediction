import os, json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "houseproject", "predictor", "ml_model")
os.makedirs(OUT, exist_ok=True)

# 1. Load
df = pd.read_csv(os.path.join(BASE, "house_data.csv"))
df = df[["location", "size", "total_sqft", "bath", "price"]].dropna()
df["bhk"] = df["size"].str.split(" ").str[0].astype(int)

def to_number(x):
    try:
        return float(x)
    except ValueError:
        parts = str(x).split("-")
        if len(parts) == 2:
            try:
                return (float(parts[0]) + float(parts[1])) / 2
            except ValueError:
                return None
    return None

df["total_sqft"] = df["total_sqft"].apply(to_number)
df = df.dropna()

# 2. Locations
df["location"] = df["location"].str.strip()
counts = df["location"].value_counts()
df["location"] = df["location"].replace(list(counts[counts <= 10].index), "other")

# 3. Outliers
df = df[df["total_sqft"] / df["bhk"] >= 300]
df = df[df["bath"] <= df["bhk"] + 2]
df["price_per_sqft"] = df["price"] * 100000 / df["total_sqft"]

def remove_pps_outliers(data):
    keep = []
    for loc, g in data.groupby("location"):
        m, s = g["price_per_sqft"].mean(), g["price_per_sqft"].std()
        if pd.isna(s):
            keep.append(g)
        else:
            keep.append(g[(g["price_per_sqft"] > m - s) & (g["price_per_sqft"] <= m + s)])
    return pd.concat(keep, ignore_index=True)

df = remove_pps_outliers(df)
print("After cleaning:", df.shape)

# 4. Split
X = df[["location", "total_sqft", "bath", "bhk"]]
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train_log = np.log1p(y_train)

# 5. Pipelines + hyperparameter grids
def make_pipeline(reg):
    prep = ColumnTransformer(
        [("loc", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["location"])],
        remainder="passthrough"
    )
    return Pipeline([("prep", prep), ("reg", reg)])

searches = {
    "Ridge Regression": (make_pipeline(Ridge()),
        {"reg__alpha": [0.1, 1.0, 10.0]}),
    "Random Forest": (make_pipeline(RandomForestRegressor(random_state=42, n_jobs=-1)),
        {"reg__n_estimators": [200, 300], "reg__max_depth": [None, 20], "reg__min_samples_leaf": [1, 2]}),
    "Gradient Boosting": (make_pipeline(GradientBoostingRegressor(random_state=42)),
        {"reg__n_estimators": [200, 300], "reg__max_depth": [3, 5], "reg__learning_rate": [0.1]}),
    "Hist Gradient Boosting": (make_pipeline(HistGradientBoostingRegressor(random_state=42)),
        {"reg__max_iter": [200, 400], "reg__learning_rate": [0.05, 0.1], "reg__max_leaf_nodes": [15, 31]}),
}

# 6. Tune, evaluate, compare
comparison = []
best = {"r2": -1}

for name, (pipe, grid) in searches.items():
    print(f"Tuning {name} ...")
    gs = GridSearchCV(pipe, grid, cv=3, scoring="r2")
    gs.fit(X_train, y_train_log)
    est = gs.best_estimator_
    pred = np.expm1(est.predict(X_test))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    params = {k.replace("reg__", ""): v for k, v in gs.best_params_.items()}
    print(f"   Test MAE: {mae:.2f}   Test R2: {r2:.3f}   Best params: {params}")
    comparison.append({"model": name, "cv_r2": round(gs.best_score_, 3),
                       "r2": round(r2, 3), "mae": round(mae, 2), "params": params})
    if r2 > best["r2"]:
        best = {"name": name, "model": est, "r2": r2, "mae": mae, "params": params}

print("\nBest model:", best["name"], "R2 =", round(best["r2"], 3))

# 7. Feature importance (permutation importance on raw features)
imp = permutation_importance(best["model"], X_test, np.log1p(y_test),
                             n_repeats=5, random_state=42, scoring="r2")
vals = np.clip(imp.importances_mean, 0, None)
vals = vals / vals.sum() * 100
names = {"location": "Location", "total_sqft": "Area (sq ft)", "bath": "Bathrooms", "bhk": "BHK"}
importance = {names[c]: round(float(v), 1) for c, v in zip(X_test.columns, vals)}
importance = dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))
print("Feature importance (%):", importance)

# 8. Save everything
joblib.dump(best["model"], os.path.join(OUT, "house_model.pkl"))
joblib.dump(sorted(df["location"].unique().tolist()), os.path.join(OUT, "locations.pkl"))
info = {
    "best_model": best["name"], "r2": round(best["r2"], 3), "mae": round(best["mae"], 2),
    "best_params": best["params"], "comparison": comparison, "importance": importance,
    "train_rows": len(X_train), "test_rows": len(X_test),
}
with open(os.path.join(OUT, "model_info.json"), "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2, default=str)
pd.DataFrame(comparison).to_csv(os.path.join(BASE, "model_comparison.csv"), index=False)
print("Saved to:", OUT)