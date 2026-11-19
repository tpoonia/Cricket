"""Extraction metrics on Set S (§3.5): how correct are the VLM's inferred constraints.

Truth for a synthetic image = rules_for_label(true pitch class) from the frozen rulebook.
Metrics: schema validity, exact set match, per-constraint precision/recall/F1, NL4Opt accuracy,
per-class perception accuracy, abstention on no-rule inputs, k-run consistency (Fleiss' kappa).

Runs through the VLM client (mock offline). Clearly tagged as mock until real photos + a key exist.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from src.rules.checker import rules_for_label
from src.vlm.client import VLMClient

PITCH_TRUTH_CLASSES = ["green", "dry", "flat", "slow", "wet"]


def _rule_set(rules) -> set:
    return {(r["feature"], r["lo"], r["hi"]) for r in rules}


def _prf(pred: set, truth: set) -> tuple[int, int, int]:
    tp = len(pred & truth)
    fp = len(pred - truth)
    fn = len(truth - pred)
    return tp, fp, fn


def fleiss_kappa(label_lists: list[list[str]]) -> float:
    """Fleiss' kappa over k categorical ratings per item."""
    cats = sorted({c for item in label_lists for c in item})
    if not cats or not label_lists:
        return float("nan")
    cat_idx = {c: i for i, c in enumerate(cats)}
    n = len(label_lists[0])
    N = len(label_lists)
    mat = np.zeros((N, len(cats)))
    for i, item in enumerate(label_lists):
        for c in item:
            mat[i, cat_idx[c]] += 1
    p_j = mat.sum(axis=0) / (N * n)
    P_i = (np.square(mat).sum(axis=1) - n) / (n * (n - 1))
    P_bar = P_i.mean()
    P_e = np.square(p_j).sum()
    return float((P_bar - P_e) / (1 - P_e)) if (1 - P_e) > 1e-9 else float("nan")


def run_set_s_extraction(manifest: pd.DataFrame, mode: str = "label", k: int = 5,
                         backend: str = "mock") -> dict:
    client = VLMClient(backend=backend)
    pitch = manifest[manifest["kind"] == "pitch"].copy()
    uninformative = manifest[manifest["class"] == "uninformative"]

    per_class_correct: dict[str, list[int]] = {c: [] for c in PITCH_TRUTH_CLASSES}
    label_runs = []  # for Fleiss
    exact = []
    TP = FP = FN = D = 0
    schema_ok = 0
    total_runs = 0

    for _, r in pitch.iterrows():
        true_cls = r["class"]
        scn = {"id": r["id"], "pitch_class": true_cls, "sky_class": "clear", "has_image": True,
               "image_path": f"data/synthetic/images/{r['file']}"}
        runs = client.run(scn, mode=mode, k=k)
        total_runs += len(runs)
        schema_ok += sum(x.get("schema_ok", False) for x in runs)

        if mode == "label":
            labels = [x.get("pitch") for x in runs if x.get("schema_ok")]
            label_runs.append(labels)
            top = Counter(labels).most_common(1)[0][0] if labels else "uninformative"
            per_class_correct.setdefault(true_cls, []).append(int(top == true_cls))
            pred = _rule_set(rules_for_label(top)["rules"])
        else:
            from src.rules.checker import check

            sets = [x.get("rules", []) for x in runs if x.get("schema_ok")]
            pred = _rule_set(check(sets or [[]], mode="rules")["rules"] if sets else [])

        truth = _rule_set(rules_for_label(true_cls)["rules"])
        exact.append(int(pred == truth))
        tp, fp, fn = _prf(pred, truth)
        TP += tp; FP += fp; FN += fn
        D += max(len(truth), 1)

    # abstention on uninformative (should be empty rule set)
    abst = []
    for _, r in uninformative.iterrows():
        scn = {"id": r["id"], "pitch_class": "uninformative", "sky_class": "uninformative",
               "has_image": True, "image_path": f"data/synthetic/images/{r['file']}"}
        runs = client.run(scn, mode=mode, k=k)
        if mode == "label":
            abst.append(int(all(x.get("pitch") in ("uninformative", None) for x in runs)))
        else:
            abst.append(int(all(len(x.get("rules", [])) == 0 for x in runs)))

    precision = TP / (TP + FP) if (TP + FP) else float("nan")
    recall = TP / (TP + FN) if (TP + FN) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    nl4opt_acc = 1 - min(FP + FN, D) / D if D else float("nan")

    per_class_acc = {c: (float(np.mean(v)) if v else float("nan")) for c, v in per_class_correct.items()}
    return {
        "mode": mode, "backend": backend, "model": client.model, "k": k,
        "n_pitch_images": int(len(pitch)),
        "schema_validity": round(schema_ok / total_runs, 3) if total_runs else float("nan"),
        "exact_set_match": round(float(np.mean(exact)), 3) if exact else float("nan"),
        "constraint_precision": round(precision, 3),
        "constraint_recall": round(recall, 3),
        "constraint_f1": round(f1, 3),
        "nl4opt_acc": round(nl4opt_acc, 3),
        "abstention_uninformative": round(float(np.mean(abst)), 3) if abst else float("nan"),
        "consistency_fleiss_kappa": round(fleiss_kappa(label_runs), 3) if mode == "label" else None,
        "per_class_perception_acc": {c: round(v, 3) for c, v in per_class_acc.items()},
    }
