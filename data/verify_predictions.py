"""Recompute full-enrolment and TTA metrics from saved predictions.

Run: python verify_predictions.py
Requires: Python 3 and NumPy. No model execution or raw radar data are needed.
Partition/short-enrolment results are supplied in RESULT.json but are not
recomputed by this script because their held-out indexing differs by partition.
"""
from pathlib import Path
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
META = json.loads((ROOT / "metadata.json").read_text(encoding="utf-8"))
rows = []
max_error = 0.0


def recompute(y, pred, persons, q1_people):
    correct = pred == y
    per_person = {p: float(correct[persons == p].mean()) for p in np.unique(persons)}
    recalls, f1s = {}, []
    for k, name in enumerate(META["classes"]):
        tp = np.sum((y == k) & (pred == k))
        fp = np.sum((y != k) & (pred == k))
        fn = np.sum((y == k) & (pred != k))
        f1s.append(float(2 * tp / (2 * tp + fp + fn)))
        recalls[name] = float(tp / np.sum(y == k))
    mask = np.isin(persons, q1_people) & np.isin(y, [0, 3])
    return dict(macro_f1=float(np.mean(f1s)), accuracy=float(correct.mean()),
                q1=float(np.mean([per_person[p] for p in q1_people])),
                rest=float(np.mean([v for p, v in per_person.items() if p not in q1_people])),
                q1_wp_to_drink=float(np.mean(pred[mask] == 4)),
                recall=recalls, per_person=per_person)


def compare(actual, expected):
    global max_error
    for key, value in actual.items():
        if isinstance(value, dict):
            compare(value, expected[key])
        else:
            error = abs(value - expected[key])
            max_error = max(max_error, error)
            if error > 1e-12:
                raise AssertionError(f"Mismatch for {key}: {value} vs {expected[key]}")


for site in ("development", "confirmation"):
    saved = json.loads((ROOT / site / "RESULT.json").read_text(encoding="utf-8"))
    with np.load(ROOT / site / "PREDICTIONS.npz", allow_pickle=False) as arrays:
        y, persons = arrays["labels"], arrays["persons"]
        groups = [("main", model, method, values)
                  for model, methods in saved["main"].items()
                  for method, values in methods.items()]
        groups += [("tta", "wavelet_mamba", method, values)
                   for method, values in saved["tta"].items()]
        for group, model, method, values in groups:
            collected = []
            for i, seed in enumerate(META["seeds"]):
                key = (f"main__{model}__{method}__{seed}" if group == "main"
                       else f"tta__{method.replace('/', '__')}__{seed}")
                raw = arrays[key]
                pred = raw.argmax(axis=1) if raw.ndim == 2 else raw
                metrics = recompute(y, pred, persons, saved["q1_persons"])
                compare(metrics, values["per_seed"][i])
                collected.append(metrics)
                rows.append(dict(site=site, group=group, backbone=model,
                                 method=method, seed=seed,
                                 **{k: metrics[k] for k in ("macro_f1", "accuracy", "q1", "rest", "q1_wp_to_drink")}))
            for key, summary in values["summary"].items():
                data = [m[key] for m in collected]
                compare({"mean": float(np.mean(data)), "sd": float(np.std(data, ddof=1))}, summary)

with (ROOT / "recomputed_metrics.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
report = {"prediction_runs_verified": len(rows), "maximum_absolute_error": max_error,
          "scope": "Full-enrolment main models and TTA: macro-F1, recording accuracy, person-mean quartile accuracy, class recall, per-person accuracy and walking/picking-up-to-drinking error; seed means and sample standard deviations."}
(ROOT / "verification_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
