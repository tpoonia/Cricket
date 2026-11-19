"""10-call pilot per real VLM reader (MASTER_PROMPT Hard rule #5).

For each backend (claude = Haiku 5.5 main, ollama = Qwen3-VL:8b local second) it runs 10 real
image->label calls on a spread of Set-S images, and records: valid-JSON rate, per-call latency,
and real USD cost (from the shared budget ledger). Results -> results/pilot.json.

    python scripts/pilot_vlm.py                 # both readers
    python scripts/pilot_vlm.py claude          # one reader
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _pilot_images(n: int = 10) -> list[dict]:
    import pandas as pd

    mf = pd.read_csv(ROOT / "data/synthetic/manifest.csv")
    seen, out = set(), []
    # one per (kind,class) first, for spread, then fill to n
    for _, r in mf.iterrows():
        key = (r["kind"], r["class"])
        if r["kind"] in ("pitch", "sky") and key not in seen:
            seen.add(key)
            out.append(r)
    for _, r in mf.iterrows():
        if len(out) >= n:
            break
        if r["id"] not in {x["id"] for x in out}:
            out.append(r)
    return [{"id": r["id"], "image_path": str(ROOT / "data/synthetic/images" / r["file"]),
             "has_image": True, "pitch_class": r["class"] if r["kind"] == "pitch" else "flat",
             "sky_class": r["class"] if r["kind"] == "sky" else "clear",
             "intended": r["class"], "kind": r["kind"]} for r in out[:n]]


def run_pilot(backend: str) -> dict:
    from src.vlm.client import VLMClient, _cost_so_far

    client = VLMClient(backend=backend)
    scns = _pilot_images(10)
    before = _cost_so_far()
    ok, lat, rows, err = 0, [], [], None
    for scn in scns:
        t = time.time()
        try:
            resp = client.run(scn, mode="label", k=1)[0]
            dt = time.time() - t
            lat.append(dt)
            valid = bool(resp.get("schema_ok"))
            ok += int(valid)
            rows.append({"id": scn["id"], "intended": scn["intended"], "kind": scn["kind"],
                         "pred_pitch": resp.get("pitch"), "pred_sky": resp.get("sky"),
                         "schema_ok": valid, "sec": round(dt, 2)})
        except Exception as e:  # noqa: BLE001 - pilot must report, not crash
            err = f"{type(e).__name__}: {e}"
            rows.append({"id": scn["id"], "error": err})
            break
    cost = round(_cost_so_far() - before, 6)
    return {"backend": backend, "model": client.model, "n": len(scns),
            "valid_json": ok, "valid_rate": round(ok / max(1, len(lat)), 3) if lat else 0.0,
            "mean_sec": round(sum(lat) / len(lat), 2) if lat else None,
            "max_sec": round(max(lat), 2) if lat else None,
            "usd_cost": cost, "error": err, "rows": rows}


def main(argv=None):
    _load_env()
    backends = argv or (sys.argv[1:] or ["claude", "ollama"])
    out = {}
    for b in backends:
        print(f"\n=== pilot: {b} ===", flush=True)
        res = run_pilot(b)
        out[b] = res
        print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=2), flush=True)
        for r in res["rows"]:
            print("  ", r, flush=True)
    dst = ROOT / "results/pilot.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dst}")
    return out


if __name__ == "__main__":
    main()
