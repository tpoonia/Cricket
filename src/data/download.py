"""F1: download Cricsheet men's T20I data + register into data/raw/.

Idempotent: skips files already present (unless --force). Records date + licence in docs/DATA.md
(the licence note is maintained by hand there; this script prints a reminder).

Usage:
    python -m src.data.download            # download if missing, then unzip
    python -m src.data.download --force     # re-download
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import urllib.request
import zipfile
from pathlib import Path

RAW = Path("data/raw")
ZIP_URL = "https://cricsheet.org/downloads/t20s_male_json.zip"
PEOPLE_URL = "https://cricsheet.org/register/people.csv"
ZIP_PATH = RAW / "t20s_male_json.zip"
PEOPLE_PATH = RAW / "people.csv"
JSON_DIR = RAW / "t20s_json"


def _fetch(url: str, dest: Path, force: bool) -> None:
    if dest.exists() and not force:
        print(f"skip (exists): {dest}")
        return
    print(f"downloading {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)  # noqa: S310 (trusted open-data host)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    _fetch(ZIP_URL, ZIP_PATH, args.force)
    _fetch(PEOPLE_URL, PEOPLE_PATH, args.force)

    JSON_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(JSON_DIR)
    n = len(list(JSON_DIR.glob("*.json")))
    print(f"extracted {n} match json files to {JSON_DIR}")
    print(f"download date: {dt.date.today().isoformat()}  licence: Cricsheet ODC-By (see docs/DATA.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
