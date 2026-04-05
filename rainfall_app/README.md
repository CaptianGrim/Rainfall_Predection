# RainCast – ML Rainfall Prediction Web App

## Project Structure

```
rainfall_app/
├── app.py                  # Flask application (routes, DB models)
├── requirements.txt
├── ml/
│   ├── __init__.py
│   └── predictor.py        # Data fetch + KNN / RF / DT training & evaluation
├── templates/
│   ├── base.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html      # Map + prediction UI
│   └── history.html
└── static/
    └── css/
        └── style.css
```

## Setup & Run

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py
```

Open http://localhost:5000 in your browser.

## How It Works

1. **Register / Login** – standard session-based auth via Flask + SQLite.
2. **Select Location** – click anywhere on the Leaflet map or search by name.
3. **Run Prediction** – backend calls Open-Meteo's free historical archive API
   to fetch ~90 days of daily weather data for the chosen lat/lon.
4. **ML Pipeline**:
   - Features: temp max/min/mean, precipitation, rain sum, wind speed,
     wind gusts, humidity, dew point, pressure, cloud cover, sunshine duration.
   - Label: `rain_tomorrow` (1 if next day precipitation > 1 mm).
   - Three models trained: KNN (k=5), Random Forest (100 trees), Decision Tree (max depth 6).
   - Best model selected by highest F1 score.
   - The best model predicts tomorrow's rain for the last available day.
5. **Results** show prediction, confidence, and a side-by-side model comparison
   (accuracy, precision, recall, F1).
6. Every prediction is stored and viewable in **History**.

## Data Source

- **Open-Meteo Historical Archive** (https://archive-api.open-meteo.com)
  – completely free, no API key required.

## Extending the App

- Swap historical data with a live forecast API (e.g. Open-Meteo forecast endpoint)
  for a true "real-time" feed.
- Add more models (XGBoost, LSTM).
- Add charts (Chart.js) for feature importance from Random Forest.
- Deploy with Gunicorn + Nginx on any Linux VPS.
