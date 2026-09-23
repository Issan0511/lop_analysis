#!/usr/bin/env python3
"""Final scoring of the registered predictions (main E0-E16 and addendum F1-F8).

    python3 analysis/sgd_postfit_cifar_0923/score_predictions.py [--root results/sgd_postfit_cifar_0923]

E0-E5, E7-E16 were scored before A_ce finished (prediction_scores_partial.json) and are carried
over unchanged.  E6 (Q7) and F1-F8 are scored here.  Every predicted value used below is first
checked against the sha256-registered text (assert the literal fragment is in the file), so a
transcription error fails loudly.  Brier = sum over options of (p - outcome)^2; a "hit" is the
unique most probable option (a tie is not a hit).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

E6 = {  # (fragment in the registered file, probabilities over the five verdicts)
    "Claude": ("PREDICTIONS_claude.md",
               "| E6 | MATCHED_CE_HARMLESS 0.42 / MATCHED_CE_HARMFUL 0.03 / ADAM_WORSE_AT_MATCHED_CE 0.11 / "
               "SGD_WORSE_AT_MATCHED_CE 0.02 / MIXED 0.42 |",
               {"MATCHED_CE_HARMLESS": .42, "MATCHED_CE_HARMFUL": .03, "ADAM_WORSE_AT_MATCHED_CE": .11,
                "SGD_WORSE_AT_MATCHED_CE": .02, "MIXED": .42}),
    "Codex": ("PREDICTIONS_codex_raw.txt",
              "E6: MATCHED_CE_HARMLESS=0.70 MATCHED_CE_HARMFUL=0.02 ADAM_WORSE_AT_MATCHED_CE=0.09 "
              "SGD_WORSE_AT_MATCHED_CE=0.03 MIXED=0.16",
              {"MATCHED_CE_HARMLESS": .70, "MATCHED_CE_HARMFUL": .02, "ADAM_WORSE_AT_MATCHED_CE": .09,
               "SGD_WORSE_AT_MATCHED_CE": .03, "MIXED": .16}),
}
F = {  # addendum 1
    "Claude": ("PREDICTIONS_add1_claude.md", {
        "F1": ("| F1 | Q9（SA_abab の中心）: BOUNDED / SLOWING / GROWING | 0.30 / 0.55 / 0.15 |",
               {"BOUNDED": .30, "SLOWING": .55, "GROWING": .15}),
        "F2": ("| F2 | Q10: 半差が BOUNDED_DIFF | 0.85 |", .85),
        "F3": ("中央値 4、80% 区間 [0.5, 15]", (4, .5, 15)),
        "F4": ("中央値 350、80% 区間 [200, 700]", (350, 200, 700)),
        "F5": ("中央値 700、80% 区間 [300, 2,000]", (700, 300, 2000)),
        "F6": ("中央値 0.25、80% 区間 [0.05, 0.8]", (.25, .05, .8)),
        "F7": ("t1–5 の 1.2 倍以下（SGD では遅れない） | 0.60 |", .60),
        "F8": ("1 slot 以上が発散する | 0.10 |", .10)}),
    "Codex": ("PREDICTIONS_add1_codex_raw.txt", {
        "F1": ("F1: BOUNDED=0.22 SLOWING=0.50 GROWING=0.28", {"BOUNDED": .22, "SLOWING": .50, "GROWING": .28}),
        "F2": ("F2: p=0.85", .85),
        "F3": ("F3: 中央値=2.50 80%区間=[0.10, 12.00]", (2.5, .1, 12)),
        "F4": ("F4: 中央値=320 80%区間=[150, 800]", (320, 150, 800)),
        "F5": ("F5: 中央値=900 80%区間=[300, 3000]", (900, 300, 3000)),
        "F6": ("F6: 中央値=0.18 80%区間=[0.01, 0.75]", (.18, .01, .75)),
        "F7": ("F7: p=0.75", .75),
        "F8": ("F8: p=0.15", .15)}),
}


def brier_cat(p: dict, outcome: str) -> float:
    return sum((q - (k == outcome)) ** 2 for k, q in p.items())


def hit_cat(p: dict, outcome: str) -> bool:
    top = max(p.values())
    return [k for k, q in p.items() if q == top] == [outcome]


def med_seed(rows, col, tasks):
    per = []
    for s in range(10):
        v = [float(r[col]) for r in rows if int(r["seed"]) == s and int(r["task"]) in tasks
             and r[col] not in ("", None)]
        if v:
            per.append(np.mean(v))
    return float(np.median(per))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "results/sgd_postfit_cifar_0923"))
    root = Path(ap.parse_args().root)
    part = json.loads((root / "prediction_scores_partial.json").read_text())
    verdict = json.loads((root / "verdict.json").read_text())
    ab = json.loads((root / "abab_center.json").read_text())

    # outcomes
    q7 = verdict["verdict"]["Q7_matched_ce"]
    n1_end = {}
    for arm in ("SA_abab", "SA_iid"):
        v = []
        for s in range(10):
            z = np.load(root / arm / "trace" / f"LR_std_seed{s}.npz")
            v.append(float(z["n1"][z["task"] == 50][-1]))
        n1_end[arm] = float(np.median(v))
    rows_iid = list(csv.DictReader(open(root / "SA_iid" / "per_task.csv")))
    h_early, h_late = med_seed(rows_iid, "hit999", range(1, 6)), med_seed(rows_iid, "hit999", range(41, 51))
    diverged = {arm: sum(1 for r in csv.DictReader(open(root / arm / "per_task.csv"))
                         if int(r["task"]) == 50 and r["memo_acc"] in ("", None)) for arm in ("SA_abab", "SA_iid")}
    out_F = {"F1": ab["verdict"]["Q9"], "F2": ab["verdict"]["Q10"] == "BOUNDED_DIFF",
             "F3": ab["SA_abab"]["median"]["g_late"], "F4": n1_end["SA_abab"], "F5": n1_end["SA_iid"],
             "F6": ab["Q11_ratio_center_to_iid"], "F7": h_late <= 1.2 * h_early,
             "F8": sum(diverged.values()) > 0}

    res = {"outcomes_main": {**part["outcomes"], "E6": q7}, "outcomes_add1": out_F,
           "SA_iid_hit999_t1_5": h_early, "SA_iid_hit999_t41_50": h_late, "diverged_t50": diverged}
    for who in ("Claude", "Codex"):
        fname, frag, p = E6[who]
        assert frag in (root / fname).read_text(), (who, "E6 fragment not found")
        per = dict(part[who]["per_event"])
        per["E6"] = brier_cat(p, q7)
        hits = part[who]["hits"] + int(hit_cat(p, q7))
        res[who] = {"main": {"per_event": per, "mean": float(np.mean(list(per.values()))), "hits": hits,
                             "n": len(per), "E6_tie_at_top": len([q for q in p.values() if q == max(p.values())]) > 1}}
        fname, preds = F[who]
        text = (root / fname).read_text()
        add = {"brier": {}, "hit": {}, "numeric": {}}
        for k, (frag, p) in preds.items():
            assert frag in text, (who, k, "fragment not found")
            o = out_F[k]
            if isinstance(p, dict):
                add["brier"][k], add["hit"][k] = brier_cat(p, o), hit_cat(p, o)
            elif isinstance(p, float):
                add["brier"][k], add["hit"][k] = (p - float(o)) ** 2, (p > .5) == bool(o)
            else:
                m, lo, hi = p
                add["numeric"][k] = {"median": m, "lo80": lo, "hi80": hi, "actual": o,
                                     "inside": lo <= o <= hi, "abs_err": abs(o - m)}
        add["brier_mean"] = float(np.mean(list(add["brier"].values())))
        add["hits"] = int(sum(add["hit"].values()))
        add["n_cat_bin"] = len(add["hit"])
        add["inside_80"] = int(sum(v["inside"] for v in add["numeric"].values()))
        res[who]["add1"] = add
    (root / "prediction_scores.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    for who in ("Claude", "Codex"):
        m, a = res[who]["main"], res[who]["add1"]
        print(f"{who}: main Brier {m['mean']:.3f} hits {m['hits']}/{m['n']} (E6 {m['per_event']['E6']:.3f}) | "
              f"add1 Brier {a['brier_mean']:.3f} hits {a['hits']}/{a['n_cat_bin']} inside80 {a['inside_80']}/4 "
              f"{ {k: round(v['abs_err'], 3) for k, v in a['numeric'].items()} }")
    print("outcomes add1", out_F, "SA_iid hit999", h_early, h_late)


if __name__ == "__main__":
    main()
