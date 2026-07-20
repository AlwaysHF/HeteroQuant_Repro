#!/usr/bin/env python3
"""Plot post-quant MoE FFN DuQuant-vs-OSE benchmark CSV."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple


def load_rows(path: str) -> List[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def require_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_latency(rows: List[dict], output_dir: str) -> None:
    plt = require_matplotlib()
    full = [r for r in rows if not r["variant"].startswith("component_")]
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for row in full:
        groups[(row["preset"], row["active_experts"])].append(row)

    for (preset, experts), items in sorted(groups.items()):
        variants = sorted({r["variant"] for r in items})
        tokens = sorted({int(r["tokens_per_expert"]) for r in items})
        by_key = {(r["variant"], int(r["tokens_per_expert"])): float(r["latency_ms"]) for r in items}

        fig, ax = plt.subplots(figsize=(9, 4.8))
        width = 0.8 / max(len(variants), 1)
        x = list(range(len(tokens)))
        for idx, variant in enumerate(variants):
            vals = [by_key.get((variant, t), float("nan")) for t in tokens]
            xpos = [v + (idx - (len(variants) - 1) / 2) * width for v in x]
            ax.bar(xpos, vals, width=width, label=variant)
        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in tokens])
        ax.set_xlabel("Tokens per active expert")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"Post-quant W8A8 FFN latency: {preset}, active experts={experts}")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"latency_{preset}_E{experts}.png"), dpi=200)
        plt.close(fig)


def plot_speedup(rows: List[dict], output_dir: str) -> None:
    plt = require_matplotlib()
    ours = [r for r in rows if r["variant"] == "ours_ose_w8a8"]
    groups: Dict[str, List[dict]] = defaultdict(list)
    for row in ours:
        groups[row["preset"]].append(row)

    for preset, items in sorted(groups.items()):
        experts = sorted({int(r["active_experts"]) for r in items})
        tokens = sorted({int(r["tokens_per_expert"]) for r in items})
        by_key = {(int(r["active_experts"]), int(r["tokens_per_expert"])): float(r["speedup_over_duquant_once"]) for r in items}

        fig, ax = plt.subplots(figsize=(9, 4.8))
        width = 0.8 / max(len(experts), 1)
        x = list(range(len(tokens)))
        for idx, e in enumerate(experts):
            vals = [by_key.get((e, t), float("nan")) for t in tokens]
            xpos = [v + (idx - (len(experts) - 1) / 2) * width for v in x]
            ax.bar(xpos, vals, width=width, label=f"E={e}")
        ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in tokens])
        ax.set_xlabel("Tokens per active expert")
        ax.set_ylabel("Speedup vs duquant_once")
        ax.set_title(f"Ours OSE W8A8 speedup over fair DuQuant baseline: {preset}")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.legend(title="Active experts")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"ours_speedup_vs_duquant_once_{preset}.png"), dpi=200)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot post-quant MoE FFN benchmark results.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_rows(args.input_csv)
    plot_latency(rows, args.output_dir)
    plot_speedup(rows, args.output_dir)
    print(f"[plot-post-quant-ffn] plots: {args.output_dir}")


if __name__ == "__main__":
    main()
