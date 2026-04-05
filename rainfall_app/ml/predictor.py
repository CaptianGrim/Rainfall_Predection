"""
ml/predictor.py
Fetches historical weather from Open-Meteo (free, no API key) and trains
KNN, Random Forest, and Decision Tree models to predict tomorrow's rain.
"""

import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

FEATURES = [
    'temp_max', 'temp_min', 'temp_mean',
    'precipitation', 'rain_sum',
    'windspeed_max', 'windgusts_max',
    'humidity_mean', 'dewpoint_mean',
    'pressure_mean', 'cloudcover_mean',
    'sunshine_duration',
]

# ── Connectivity check ────────────────────────────────────────────────────────

def check_api_reachable() -> tuple[bool, str]:
    """Quick probe to give a helpful error reason."""
    try:
        requests.get("https://archive-api.open-meteo.com", timeout=5)
        return True, ""
    except requests.exceptions.ProxyError:
        return False, "proxy"
    except requests.exceptions.SSLError:
        return False, "ssl"
    except requests.exceptions.ConnectionError:
        return False, "connection"
    except requests.exceptions.Timeout:
        return False, "timeout"
    except Exception as e:
        return False, str(e)[:60]

# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_weather(lat: float, lon: float, days: int = 90) -> pd.DataFrame | None:
    end   = datetime.utcnow().date() - timedelta(days=1)
    start = end - timedelta(days=days)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": str(start),
        "end_date":   str(end),
        "daily": [
            "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
            "precipitation_sum", "rain_sum",
            "windspeed_10m_max", "windgusts_10m_max",
            "dewpoint_2m_mean", "pressure_msl_mean", "cloudcover_mean",
            "sunshine_duration",
        ],
        "hourly":   ["relativehumidity_2m"],
        "timezone": "auto",
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        print(f"[fetch_weather] API error: {type(exc).__name__}: {exc}")
        return None

    daily  = data.get('daily',  {})
    hourly = data.get('hourly', {})

    hum_df = pd.DataFrame({
        'time': hourly.get('time', []),
        'hum':  hourly.get('relativehumidity_2m', [])
    })
    hum_df['date'] = pd.to_datetime(hum_df['time']).dt.date.astype(str)
    hum_daily = hum_df.groupby('date')['hum'].mean().reset_index()
    hum_daily.columns = ['time', 'humidity_mean']

    df = pd.DataFrame({
        'time':             daily.get('time', []),
        'temp_max':         daily.get('temperature_2m_max',  []),
        'temp_min':         daily.get('temperature_2m_min',  []),
        'temp_mean':        daily.get('temperature_2m_mean', []),
        'precipitation':    daily.get('precipitation_sum',   []),
        'rain_sum':         daily.get('rain_sum',            []),
        'windspeed_max':    daily.get('windspeed_10m_max',   []),
        'windgusts_max':    daily.get('windgusts_10m_max',   []),
        'dewpoint_mean':    daily.get('dewpoint_2m_mean',    []),
        'pressure_mean':    daily.get('pressure_msl_mean',   []),
        'cloudcover_mean':  daily.get('cloudcover_mean',     []),
        'sunshine_duration':daily.get('sunshine_duration',   []),
    })

    df = df.merge(hum_daily, on='time', how='left')
    df['rain_tomorrow'] = (df['precipitation'].shift(-1).fillna(0) > 1.0).astype(int)
    df = df.dropna(subset=['temp_max', 'precipitation']).iloc[:-1]
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())

    return df if len(df) >= 20 else None


def _synthetic_fallback(lat: float, lon: float, days: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(seed=int(abs(lat * 100 + lon * 10)) % 9999)
    n = days
    base_temp = 30 - abs(lat) * 0.4
    rain_prob = 0.45 if abs(lat) < 25 else 0.35

    temp_mean    = base_temp + rng.normal(0, 4, n)
    temp_max     = temp_mean + rng.uniform(3, 8, n)
    temp_min     = temp_mean - rng.uniform(3, 8, n)
    rain_day     = rng.random(n) < rain_prob
    precipitation= np.where(rain_day, rng.exponential(8, n), rng.uniform(0, 0.5, n))
    rain_sum     = precipitation * rng.uniform(0.7, 1.0, n)
    windspeed    = rng.uniform(5, 40, n)
    windgusts    = windspeed + rng.uniform(5, 20, n)
    humidity     = np.where(rain_day, rng.uniform(70, 95, n), rng.uniform(35, 70, n))
    dewpoint     = temp_mean - rng.uniform(2, 10, n)
    pressure     = 1013 + rng.normal(0, 8, n)
    cloudcover   = np.where(rain_day, rng.uniform(60, 100, n), rng.uniform(5, 60, n))
    sunshine     = np.where(rain_day, rng.uniform(0, 14400, n), rng.uniform(14400, 43200, n))

    df = pd.DataFrame({
        'temp_max': temp_max, 'temp_min': temp_min, 'temp_mean': temp_mean,
        'precipitation': precipitation, 'rain_sum': rain_sum,
        'windspeed_max': windspeed, 'windgusts_max': windgusts,
        'humidity_mean': humidity, 'dewpoint_mean': dewpoint,
        'pressure_mean': pressure, 'cloudcover_mean': cloudcover,
        'sunshine_duration': sunshine,
    })
    df['rain_tomorrow'] = (np.roll(precipitation, -1) > 1.0).astype(int)
    return df.iloc[:-1]

# ── Evaluation ────────────────────────────────────────────────────────────────

def _evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        'accuracy':  round(float(accuracy_score(y_test, y_pred)), 4),
        'precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        'recall':    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        'f1':        round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
    }

# ── Main predictor ────────────────────────────────────────────────────────────

class RainfallPredictor:

    def run(self, lat: float, lon: float) -> dict:
        df = fetch_weather(lat, lon)
        using_synthetic = False
        api_fail_reason = None

        if df is None:
            reachable, reason = check_api_reachable()
            api_fail_reason = reason if not reachable else "bad_data"
            df = _synthetic_fallback(lat, lon)
            using_synthetic = True

        X = df[FEATURES].values
        y = df['rain_tomorrow'].values

        stratify = y if (y.sum() > 5 and (len(y) - y.sum()) > 5) else None
        test_size = max(0.2, min(0.3, 15 / len(df)))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        models = {
            'KNN':           KNeighborsClassifier(n_neighbors=5),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
        }

        results = {}
        trained = {}
        for name, clf in models.items():
            Xtr = X_train_s if name == 'KNN' else X_train
            Xte = X_test_s  if name == 'KNN' else X_test
            clf.fit(Xtr, y_train)
            results[name] = _evaluate(clf, Xte, y_test)
            trained[name] = clf

        best_model = max(results, key=lambda k: results[k]['f1'])

        last_row   = df[FEATURES].iloc[-1].values.reshape(1, -1)
        last_row_s = scaler.transform(last_row)

        clf_best = trained[best_model]
        input_X  = last_row_s if best_model == 'KNN' else last_row
        pred_val = int(clf_best.predict(input_X)[0])
        proba    = clf_best.predict_proba(input_X)[0]

        importances = None
        if best_model in ('Random Forest', 'Decision Tree'):
            imp = clf_best.feature_importances_
            importances = dict(zip(FEATURES, [round(float(v), 4) for v in imp]))

        return {
            'prediction':     'Rain' if pred_val == 1 else 'No Rain',
            'confidence':     round(float(max(proba)) * 100, 1),
            'best_model':     best_model,
            'model_results':  results,
            'data_points':    len(df),
            'importances':    importances,
            'synthetic_data': using_synthetic,
            'api_fail_reason': api_fail_reason,
        }
