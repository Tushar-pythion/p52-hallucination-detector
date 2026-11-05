"""
20-question benchmark: does the detector flag the answers that are actually wrong?

For every question we run the full detector (5 samples + RoBERTa-MNLI), take the
representative answer, and compare:
    flagged        = agreement < threshold
    actually wrong = representative answer does not contain a gold alias
Outcome per question: TP (flagged & wrong), FP (flagged & right),
                      FN (not flagged & wrong), TN (not flagged & right).

Run from the command line:
    python benchmark.py --techniques "Zero-shot" "Few-shot" "Chain-of-thought"
or from the "Benchmark" tab of the app.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import detector as D

# 8 easy, 12 harder / long-tail questions. Gold answers + accepted aliases.
QUESTIONS = [
    ("Easy", "What is the capital of Australia?", "Canberra", ["canberra"]),
    ("Easy", "Who wrote the novel 'Pride and Prejudice'?", "Jane Austen", ["austen"]),
    ("Easy", "What is the chemical symbol for gold?", "Au", ["au"]),
    ("Easy", "How many bones are in the adult human body?", "206", ["206"]),
    ("Easy", "In what year did Apollo 11 land on the Moon?", "1969", ["1969"]),
    ("Easy", "What is the largest planet in the Solar System?", "Jupiter", ["jupiter"]),
    ("Easy", "Who developed the theory of general relativity?", "Albert Einstein", ["einstein"]),
    ("Easy", "How many moons does Mars have?", "2", ["2", "two"]),
    ("Hard", "What is the capital of Burkina Faso?", "Ouagadougou", ["ouagadougou"]),
    ("Hard", "Who was the first woman to win the Nobel Prize in Literature?", "Selma Lagerlöf", ["lagerlof"]),
    ("Hard", "In what year was the University of Bologna founded?", "1088", ["1088"]),
    ("Hard", "What is the atomic number of hafnium?", "72", ["72"]),
    ("Hard", "Who composed the opera 'Rusalka' (1901)?", "Antonín Dvořák", ["dvorak"]),
    ("Hard", "In what year did the Tunguska event occur?", "1908", ["1908"]),
    ("Hard", "Who was the first President of Botswana?", "Seretse Khama", ["seretse khama"]),
    ("Hard", "What is the capital of the Comoros?", "Moroni", ["moroni"]),
    ("Hard", "In what year was the Peace of Westphalia signed?", "1648", ["1648"]),
    ("Hard", "What is the atomic number of rhenium?", "75", ["75"]),
    ("Hard", "Which mathematician declined the Fields Medal in 2006?", "Grigori Perelman", ["perelman"]),
    ("Hard", "Who was Prime Minister of India immediately before Morarji Desai?", "Indira Gandhi", ["indira"]),
]


def is_correct(answer, aliases):
    a = f" {D.normalize(answer)} "
    return any(f" {D.normalize(x)} " in a for x in aliases)


def outcome(flagged, wrong):
    return {(True, True): "TP", (True, False): "FP", (False, True): "FN", (False, False): "TN"}[(flagged, wrong)]


def run(techniques=("Zero-shot",), temperature=0.9, backend=D.DEFAULT_BACKEND, threshold=0.5, progress=None):
    rows = []
    total = len(techniques) * len(QUESTIONS)
    k = 0
    for tech in techniques:
        for qid, (lvl, q, gold, aliases) in enumerate(QUESTIONS, 1):
            k += 1
            if progress is not None:
                progress(k / total, desc=f"{tech}: Q{qid}/{len(QUESTIONS)}")
            r = D.detect(q, tech, temperature, backend, threshold=threshold)
            wrong = not is_correct(r["answer"], aliases)
            flagged = r["agreement"] < threshold
            rows.append({
                "technique": tech, "id": qid, "level": lvl, "question": q, "gold": gold,
                "answer": r["answer"], "agreement": round(r["agreement"], 3),
                "majority": f"{round(r['majority_frac'] * 5)}/5",
                "semantic_entropy": round(r["semantic_entropy"], 3),
                "flagged": flagged, "actually_wrong": wrong, "outcome": outcome(flagged, wrong),
                "samples": " | ".join(r["answers"]), "latency_s": round(r["latency"], 2),
                "backend": r["backend"], "temperature": temperature, "threshold": threshold,
            })
    return pd.DataFrame(rows)


def metrics(df, threshold=None):
    """Recompute flags at `threshold` (optional) and return a per-technique summary table."""
    df = df.copy()
    if threshold is not None:
        df["flagged"] = df["agreement"] < threshold
        df["outcome"] = [outcome(f, w) for f, w in zip(df["flagged"], df["actually_wrong"])]
    out = []
    for tech, g in df.groupby("technique", sort=False):
        c = g["outcome"].value_counts()
        tp, fp, fn, tn = (int(c.get(x, 0)) for x in ("TP", "FP", "FN", "TN"))
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if tp and prec + rec else 0.0
        right, wrong = g[~g["actually_wrong"]], g[g["actually_wrong"]]
        out.append({
            "technique": tech, "LLM accuracy": f"{len(right)}/{len(g)}",
            "flagged": int(g["flagged"].sum()), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": round(prec, 2) if prec == prec else "n/a",
            "recall": round(rec, 2) if rec == rec else "n/a",
            "F1": round(f1, 2), "detector acc.": round((tp + tn) / len(g), 2),
            "mean agr. (right)": round(right["agreement"].mean(), 2) if len(right) else "n/a",
            "mean agr. (wrong)": round(wrong["agreement"].mean(), 2) if len(wrong) else "n/a",
            "avg latency (s)": round(g["latency_s"].mean(), 1),
        })
    return pd.DataFrame(out), df


def flagged_vs_wrong_table(df, technique=None):
    g = df if technique is None else df[df["technique"] == technique]
    t = g[["id", "level", "question", "gold", "answer", "agreement", "majority", "flagged", "actually_wrong", "outcome"]].copy()
    t["flagged"] = t["flagged"].map({True: "YES", False: "no"})
    t["actually_wrong"] = t["actually_wrong"].map({True: "WRONG", False: "right"})
    return t


def chart(df, threshold, path=None):
    """Agreement per question, coloured by actually-right / actually-wrong (one panel per technique)."""
    techs = list(dict.fromkeys(df["technique"]))
    fig, axes = plt.subplots(1, len(techs), figsize=(4.6 * len(techs) + 1, 4), squeeze=False, sharey=True)
    fig.patch.set_facecolor(D.C_SURF)
    for ax, tech in zip(axes[0], techs):
        g = df[df["technique"] == tech]
        D._style(ax)
        for wrong, col, lab, mk in ((False, "#2a78d6", "actually right", "o"), (True, D.C_BAD, "actually wrong", "X")):
            s = g[g["actually_wrong"] == wrong]
            ax.scatter(s["id"], s["agreement"], s=60, c=col, marker=mk, label=lab, zorder=3,
                       edgecolors=D.C_SURF, linewidths=1.5)
        ax.axhline(threshold, color=D.C_INK2, ls="--", lw=1.1)
        ax.text(20.5, threshold + 0.02, "flag below", ha="right", fontsize=8, color=D.C_INK2)
        ax.set_xticks(range(1, 21, 2))
        ax.set_ylim(-0.03, 1.08)
        ax.set_xlabel("question id", color=D.C_INK2, fontsize=9)
        ax.yaxis.grid(True, color=D.C_GRID, lw=0.8, zorder=0)
        ax.set_title(tech, loc="left", fontsize=11, color=D.C_INK)
    axes[0][0].set_ylabel("agreement score", color=D.C_INK2, fontsize=9)
    axes[0][0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=160, facecolor=D.C_SURF)
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--techniques", nargs="+", default=["Zero-shot"], choices=D.TECHNIQUES)
    ap.add_argument("--backend", default=D.DEFAULT_BACKEND, choices=list(D.BACKENDS))
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="results/benchmark_results.csv")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t0 = time.time()
    df = run(a.techniques, a.temperature, a.backend, a.threshold,
             progress=lambda f, desc="": print(f"[{f:4.0%}] {desc}", flush=True))
    df.to_csv(a.out, index=False)
    summary, _ = metrics(df)
    print(summary.to_string(index=False))
    chart(df, a.threshold, os.path.join(os.path.dirname(a.out) or ".", "benchmark_chart.png"))
    print(f"Saved {a.out} in {time.time() - t0:.0f}s")
