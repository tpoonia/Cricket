"""#3: match-day weather from the Open-Meteo historical archive (CC BY 4.0 — credit it).

For each match we fetch hourly cloud cover (low/mid/high), precipitation, relative humidity,
dew point and wind, and summarise afternoon (12:00-17:00) and evening (17:00-22:00) means.
Every call is cached to cache/weather/<hash>.json so re-runs cost nothing (no API key needed).
"""
from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

CACHE_DIR = Path("cache/weather")
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY = [
    "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "precipitation", "relative_humidity_2m", "dew_point_2m", "wind_speed_10m",
]
AFTERNOON = range(12, 17)
EVENING = range(17, 22)

ATTRIBUTION = "Weather data by Open-Meteo.com (CC BY 4.0)."


def _key(lat: float, lon: float, date: str) -> str:
    return hashlib.sha256(f"{lat:.4f},{lon:.4f},{date}".encode()).hexdigest()[:16]


def _fetch(lat: float, lon: float, date: str) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{_key(lat, lon, date)}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": date, "end_date": date,
        "hourly": ",".join(HOURLY), "timezone": "auto",
    }
    url = ARCHIVE_URL + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=30) as r:  # noqa: S310 (open-data host)
        data = json.load(r)
    cache.write_text(json.dumps(data))
    return data


def _window_mean(hourly: dict, var: str, hours: range) -> float | None:
    times = hourly.get("time", [])
    vals = hourly.get(var, [])
    picked = [v for t, v in zip(times, vals) if int(t[11:13]) in hours and v is not None]
    return round(sum(picked) / len(picked), 2) if picked else None


def match_weather(lat: float, lon: float, date: str) -> dict:
    """Return a compact weather summary dict for one match day (cached)."""
    data = _fetch(lat, lon, date)
    h = data.get("hourly", {})
    out = {"date": date, "lat": lat, "lon": lon}
    for var, short in [
        ("cloud_cover_low", "cloud_low"), ("cloud_cover_mid", "cloud_mid"),
        ("cloud_cover_high", "cloud_high"), ("precipitation", "precip"),
        ("relative_humidity_2m", "humidity"), ("dew_point_2m", "dew_point"),
        ("wind_speed_10m", "wind"),
    ]:
        out[f"aft_{short}"] = _window_mean(h, var, AFTERNOON)
        out[f"eve_{short}"] = _window_mean(h, var, EVENING)
    return out


def weather_for_match(match_row, coords: dict) -> dict:
    date = str(match_row["date"])[:10]
    return match_weather(coords["lat"], coords["lon"], date)
