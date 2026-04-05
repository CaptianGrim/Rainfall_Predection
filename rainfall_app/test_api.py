"""
Run this on YOUR machine to check if Open-Meteo is reachable:
    python test_api.py
"""
import requests

print("Testing Open-Meteo API connection...")
try:
    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": 9.93,
            "longitude": 76.26,
            "start_date": "2025-01-01",
            "end_date": "2025-01-10",
            "daily": ["precipitation_sum", "temperature_2m_max"],
            "timezone": "auto",
        },
        timeout=15,
    )
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print("SUCCESS! Sample data received:")
        print("  Dates:", data["daily"]["time"][:3])
        print("  Precip:", data["daily"]["precipitation_sum"][:3])
        print()
        print("The Open-Meteo API is reachable. Your app will use REAL weather data.")
    else:
        print("Error response:", r.text[:200])
except Exception as e:
    print(f"FAILED: {e}")
    print()
    print("Possible reasons:")
    print("  1. No internet connection")
    print("  2. Firewall / proxy blocking HTTPS")
    print("  3. Open-Meteo is temporarily down")
    print()
    print("Check: can you open https://archive-api.open-meteo.com in your browser?")
