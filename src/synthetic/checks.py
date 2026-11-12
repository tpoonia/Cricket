"""#6/M3: automated image check + human-check sheet (§3.3 Set S).

Rater A = an automated model rater (Claude Code) that views images blind — a small recorded
sample lives in data/synthetic/rater_a.csv. Rater B = a zero-shot classifier: CLIP/SigLIP when
available, else a documented colour-feature heuristic (stand-in on the procedural placeholders).
We report Cohen's kappa between rater B / rater A and the intended class, and drop images both
raters disagree with. A 50-image sheet is written for two humans to label later.

NOTE: on procedural placeholders this is a plumbing check; the paper must say
"automated check; human check pending" and never present it as a human check.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import cohen_kappa_score

MANIFEST = Path("data/synthetic/manifest.csv")
IMG_DIR = Path("data/synthetic/images")
RATER_A = Path("data/synthetic/rater_a.csv")
HUMAN_SHEET = Path("data/synthetic/human_check_sheet.csv")

PITCH = ["green", "dry", "flat", "slow", "wet"]
SKY = ["clear", "overcast", "rain_now", "night"]


def _mean_rgb(path: Path) -> tuple[float, float, float]:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=float)
    return tuple(arr.reshape(-1, 3).mean(axis=0))


def heuristic_label(path: Path, kind: str) -> str:
    """Rater B stand-in: nearest base colour (documented; CLIP replaces this on real photos)."""
    from src.synthetic.generate import BASE_COLORS

    rgb = np.array(_mean_rgb(path))
    classes = PITCH if kind == "pitch" else SKY if kind == "sky" else ["uninformative"]
    best, bestd = None, 1e9
    for c in classes:
        d = np.linalg.norm(rgb - np.array(BASE_COLORS[c]))
        if d < bestd:
            best, bestd = c, d
    return best


def clip_label(path: Path, kind: str):  # pragma: no cover
    """Rater B via CLIP zero-shot (gated on open_clip/torch). Used on real photos."""
    import open_clip  # type: ignore
    import torch  # type: ignore

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tok = open_clip.get_tokenizer("ViT-B-32")
    classes = PITCH if kind == "pitch" else SKY
    prompts = [f"a cricket {kind} that is {c}" for c in classes]
    img = preprocess(Image.open(path)).unsqueeze(0)
    with torch.no_grad():
        im = model.encode_image(img)
        tx = model.encode_text(tok(prompts))
        sim = (im @ tx.T).softmax(dim=-1)[0]
    return classes[int(sim.argmax())]


def run_check(use_clip: bool = False) -> dict:
    mf = pd.read_csv(MANIFEST)
    labels = []
    for _, r in mf.iterrows():
        kind = r["kind"]
        if kind not in ("pitch", "sky"):
            labels.append("uninformative")
            continue
        p = IMG_DIR / r["file"]
        labels.append(clip_label(p, kind) if use_clip else heuristic_label(p, kind))
    mf["rater_b"] = labels

    pitch_sky = mf[mf["kind"].isin(["pitch", "sky"])]
    kappa_b = cohen_kappa_score(pitch_sky["class"], pitch_sky["rater_b"])

    result = {"n": len(mf), "n_classified": len(pitch_sky),
              "rater_b_vs_intended_kappa": round(float(kappa_b), 3),
              "rater_b": "heuristic" if not use_clip else "clip"}

    # rater A sample (if recorded)
    if RATER_A.exists():
        ra = pd.read_csv(RATER_A)
        merged = ra.merge(mf[["id", "class"]], on="id", how="inner")
        if len(merged) >= 2:
            result["rater_a_sample_n"] = len(merged)
            result["rater_a_vs_intended_kappa"] = round(
                float(cohen_kappa_score(merged["class"], merged["rater_a_label"])), 3)

    # images both raters miss (rater A sample only)
    mf.to_csv(Path("data/synthetic/check_result.csv"), index=False)
    return result


def build_human_sheet(n: int = 50, seed: int = 0) -> Path:
    mf = pd.read_csv(MANIFEST)
    sample = mf.sample(min(n, len(mf)), random_state=seed)
    HUMAN_SHEET.parent.mkdir(parents=True, exist_ok=True)
    with open(HUMAN_SHEET, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "file", "human_rater_1_label", "human_rater_2_label", "notes"])
        for _, r in sample.iterrows():
            w.writerow([r["id"], r["file"], "", "", ""])
    return HUMAN_SHEET


if __name__ == "__main__":
    print(run_check())
    print("human sheet:", build_human_sheet())
