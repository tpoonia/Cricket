"""#2: venue -> lat/lon/country.

Resolution order: manual CSV override, then cached geocode, then the Open-Meteo geocoding API
(free, CC-BY; not scraping — DECISIONS D15). Results are cached to data/manual/venue_coords.csv
so re-runs need no network.
"""
from __future__ import annotations

import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

COORDS_CSV = Path("data/manual/venue_coords.csv")
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"


def _load() -> pd.DataFrame:
    if COORDS_CSV.exists():
        return pd.read_csv(COORDS_CSV, dtype={"venue": str})
    return pd.DataFrame(columns=["venue", "city", "lat", "lon", "country", "source"])


def _save(df: pd.DataFrame) -> None:
    COORDS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(COORDS_CSV, index=False)


def _geocode(query: str) -> dict | None:
    url = GEOCODE_URL + "?" + urllib.parse.urlencode({"name": query, "count": 1})
    try:
        with urllib.request.urlopen(url, timeout=20) as r:  # noqa: S310 (open-data host)
            import json

            data = json.load(r)
    except Exception:
        return None
    res = data.get("results")
    if not res:
        return None
    r0 = res[0]
    return {"lat": r0["latitude"], "lon": r0["longitude"], "country": r0.get("country")}


def resolve(venue: str, city: str | None = None, allow_network: bool = True) -> dict | None:
    """Return {lat, lon, country} for a venue, using cache/manual first."""
    df = _load()
    hit = df[df["venue"] == venue]
    if len(hit) and pd.notna(hit.iloc[0]["lat"]):
        row = hit.iloc[0]
        return {"lat": float(row["lat"]), "lon": float(row["lon"]), "country": row.get("country")}
    if not allow_network:
        return None
    # try city, then the city part of the venue name, then the venue name
    queries = []
    if city and isinstance(city, str):
        queries.append(city)
    if "," in venue:
        queries.append(venue.split(",")[-1].strip())
    queries.append(venue)
    geo = next((g for q in queries if (g := _geocode(q))), None)
    if geo is None:
        return None
    new = {"venue": venue, "city": city, "lat": geo["lat"], "lon": geo["lon"],
           "country": geo["country"], "source": "open-meteo-geocode"}
    new_df = pd.DataFrame([new])
    df = new_df if df.empty else pd.concat([df, new_df], ignore_index=True)
    df = df.drop_duplicates("venue", keep="last")
    _save(df)
    return {"lat": geo["lat"], "lon": geo["lon"], "country": geo["country"]}


def resolve_all(venue_city_pairs, pause: float = 0.2) -> pd.DataFrame:
    """Batch-resolve (venue, city) pairs, caching to CSV. Returns the coords table."""
    for venue, city in venue_city_pairs:
        if resolve(venue, city) is None:
            # record a blank so we don't retry forever; can be hand-filled later
            df = _load()
            if venue not in set(df["venue"]):
                df = pd.concat(
                    [df, pd.DataFrame([{"venue": venue, "city": city, "lat": None,
                                        "lon": None, "country": None, "source": "unresolved"}])],
                    ignore_index=True,
                )
                _save(df)
        time.sleep(pause)
    return _load()


def country_map() -> dict:
    df = _load()
    return dict(zip(df["venue"], df["country"]))
