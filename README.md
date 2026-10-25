# 🏠 Bengaluru House Price Predictor

A full-stack machine learning web app that predicts house prices in Bengaluru from location, area, BHK and bathrooms. Built with Django and Scikit-learn.

## Features

- Instant price estimate with a low-to-high range and price per sq ft
- Charts: price vs area, and the same house across popular locations
- User registration, login and saved prediction history (database)
- Model Insights page: model comparison, tuned hyperparameters, feature importance
- Responsive design with Bootstrap

## Machine Learning

- **Data:** Bengaluru house price dataset (13,320 listings from Kaggle)
- **Cleaning:** converted sq ft ranges, extracted BHK from text, grouped rare locations, removed price-per-sq-ft outliers per location
- **Target:** log-transformed price to handle skewness
- **Models compared:** Ridge Regression, Random Forest, Gradient Boosting, Hist Gradient Boosting, tuned with GridSearchCV (3-fold CV)
- **Best model:** Gradient Boosting
- **Test R²:** 0.731 | **MAE:** 17.54 lakhs

## Tech Stack

Python, Django, Scikit-learn, Pandas, NumPy, SQLite, Bootstrap 5, Chart.js

## Run Locally

```
git clone https://github.com/Jyothi-Chikkannegowda/house-price-prediction.git
cd house-price-prediction
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
cd houseproject
python manage.py migrate
python manage.py runserver
```

Open http://127.0.0.1:8000/

## Project Structure

```
house-price-prediction/
├── train_model.py        # cleaning, training, tuning, saving model
├── house_data.csv        # dataset
├── requirements.txt
└── houseproject/
    ├── manage.py
    └── predictor/        # Django app (views, models, templates, ml_model)
```

## Author

**Jyothi C** · [LinkedIn](https://www.linkedin.com/in/jyothi-c-568036369/)