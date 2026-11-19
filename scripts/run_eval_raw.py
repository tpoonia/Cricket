"""Evaluation-only raw run (no figures/metrics/stats/paper). Saves raw model outputs.

Readers (see docs/DECISIONS.md): Claude Haiku 5.5 is the MAIN reader (Anthropic credit restored,
verified by the pilot), Qwen3-VL:8b via Ollama is the local SECOND reader. Gemini image-gen stays
blocked (free tier), so Set-S images are the real SDXL-Turbo set (D22).

Outputs (all raw):
  data/results/raw/set_s_results.json        Set S planted decision, all arms (VLM reader=Haiku)
  data/results/raw/set_r_results.json        Set R decision, all arms (VLM reader=Haiku)
  data/results/raw/qwen_results.json         Qwen (local) on 50 seeded Set-S scenarios (label+rules raw)
  data/results/raw/extractions/Ours1_label.json, Ours2_rules.json   per-scenario extracted rules (Qwen)
  results/pilot_haiku.json                   Haiku 10-call pilot (JSON + cost)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RAW = ROOT / "data/results/raw"
EXTR = RAW / "extractions"


def _load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    print(f"  saved {path}  ({path.stat().st_size} bytes)", flush=True)


def step1_haiku_pilot():
    from scripts.pilot_vlm import run_pilot
    print("\n[1] Haiku 10-call pilot (verify JSON + cost)", flush=True)
    try:
        res = run_pilot("claude")
    except Exception as e:  # noqa: BLE001
        res = {"backend": "claude", "error": f"{type(e).__name__}: {e}"}
    _dump(res, ROOT / "results/pilot_haiku.json")
    print(f"  valid_json={res.get('valid_json')}/{res.get('n')}  "
          f"cost=${res.get('usd_cost', 0.0)}  error={res.get('error')}", flush=True)


def step2_set_s():
    from src.evaluation.vision_constraints import run_set_s_decision
    print("\n[2] Set S planted decision, all arms (VLM reader = Haiku 5.5, k=1)", flush=True)
    t = time.time()
    df = run_set_s_decision(n_carriers=120, n_sims=4000, backend="claude", k=1)
    print(f"  rows={len(df)} arms={sorted(df['arm'].unique())} ({time.time()-t:.0f}s)", flush=True)
    _dump(df.to_dict(orient="records"), RAW / "set_s_results.json")


def step3_set_r():
    from src.evaluation.vision_constraints import run_setr
    print("\n[3] Set R decision, all arms (VLM reader = Haiku 5.5 on all matches, k=1)", flush=True)
    t = time.time()
    df = run_setr(n_sims=4000, n_matches=250, backend="claude", k=1, vlm_max=None)
    print(f"  rows={len(df)} matches={df['match_id'].nunique()} arms={sorted(df['arm'].unique())} "
          f"({time.time()-t:.0f}s)", flush=True)
    _dump(df.to_dict(orient="records"), RAW / "set_r_results.json")


def step4_5_qwen_50():
    import pandas as pd

    from src.rules.checker import check, rules_for_label
    from src.vlm.client import VLMClient
    print("\n[4] Qwen on 50 seeded Set-S scenarios (label+rules raw) + [5] per-arm extractions",
          flush=True)
    mf = pd.read_csv(ROOT / "data/synthetic/manifest.csv").sample(50, random_state=0).reset_index(drop=True)
    client = VLMClient(backend="ollama")
    scenarios, ours1, ours2 = [], {}, {}
    t = time.time()
    for i, r in mf.iterrows():
        img = str(ROOT / "data/synthetic/images" / r["file"])
        scn = {"id": r["id"], "has_image": True, "image_path": img,
               "pitch_class": r["class"], "sky_class": r["class"]}
        label_raw = client.run(scn, mode="label", k=1)
        rules_raw = client.run(scn, mode="rules", k=1)
        # map to the two "Ours" arms
        from collections import Counter
        labs = [x.get("pitch") for x in label_raw if x.get("schema_ok")]
        top = Counter(labs).most_common(1)[0][0] if labs else "uninformative"
        o1 = rules_for_label(top)["rules"]
        sets = [x.get("rules", []) for x in rules_raw if x.get("schema_ok")]
        o2 = check(sets or [[]], mode="rules")["rules"] if sets else []
        scenarios.append({"id": r["id"], "file": r["file"], "intended_class": r["class"],
                          "kind": r["kind"], "label_raw": label_raw, "rules_raw": rules_raw,
                          "pred_label": top})
        ours1[r["id"]] = o1
        ours2[r["id"]] = o2
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/50 done ({time.time()-t:.0f}s)", flush=True)
    _dump({"backend": "ollama", "model": client.model, "n_scenarios": len(scenarios), "k": 1,
           "scenarios": scenarios}, RAW / "qwen_results.json")
    _dump({"arm": "Ours1_label", "note": "VLM label -> frozen rulebook", "rules_by_scenario": ours1},
          EXTR / "Ours1_label.json")
    _dump({"arm": "Ours2_rules", "note": "VLM writes rules directly (combined by tighter bound)",
           "rules_by_scenario": ours2}, EXTR / "Ours2_rules.json")


def main():
    _load_env()
    RAW.mkdir(parents=True, exist_ok=True)
    EXTR.mkdir(parents=True, exist_ok=True)
    step1_haiku_pilot()
    step2_set_s()       # Haiku: fast + cheap
    step3_set_r()       # Haiku: fast + cheap
    step4_5_qwen_50()   # local Qwen: slow, runs last

    from src.vlm.client import _cost_so_far
    print(f"\n[7] TOTAL API COST: ${_cost_so_far():.4f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
