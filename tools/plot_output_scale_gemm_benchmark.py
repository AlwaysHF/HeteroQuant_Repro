#!/usr/bin/env python3
import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ORDER = ["per_channel", "2_channels", "4_channels", "8_channels", "16_channels", "per_tensor"]
LABELS = {
    "per_channel": "Per-channel",
    "2_channels": "2",
    "4_channels": "4",
    "8_channels": "8",
    "16_channels": "16",
    "per_tensor": "Per-tensor",
}


def read_csv(path: str) -> List[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mean(values):
    return sum(values) / max(len(values), 1)


def std(values):
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def aggregate_speedups(rows: List[dict], scope: str) -> Dict[str, dict]:
    grouped = defaultdict(list)
    for row in rows:
        if row["scope"] != scope:
            continue
        if not row.get("speedup_over_per_channel"):
            continue
        grouped[row["granularity"]].append(float(row["speedup_over_per_channel"]))
    return {
        gran: {
            "mean": mean(values),
            "std": std(values),
            "n": len(values),
        }
        for gran, values in grouped.items()
    }


def read_reductions(metadata_rows: List[dict]) -> Dict[str, str]:
    return {
        row["granularity"]: f"{float(row['scale_reduction_vs_per_channel']):.0f}x fewer scales"
        for row in metadata_rows
    }


def plot_speedup(rows: List[dict], metadata_rows: List[dict], scope: str, output_dir: str) -> None:
    agg = aggregate_speedups(rows, scope)
    reductions = read_reductions(metadata_rows)
    xs = np.arange(len(ORDER))
    means = [agg.get(g, {"mean": np.nan})["mean"] for g in ORDER]
    errs = [agg.get(g, {"std": 0.0})["std"] for g in ORDER]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    bars = ax.bar(xs, means, yerr=errs, capsize=4, color="#5B8FD9", edgecolor="#2F5F9F", linewidth=0.8)
    ax.axhline(1.0, color="#555555", linewidth=1.0, linestyle="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[g] for g in ORDER], rotation=20, ha="right")
    ax.set_xlabel("Output channels per scale")
    ax.set_ylabel("Speedup over per-channel")
    ax.set_title(f"{scope}: output-scale granularity speedup")
    ymin = max(0.0, min([v for v in means if not np.isnan(v)] + [1.0]) * 0.95)
    ymax = max([v for v in means if not np.isnan(v)] + [1.0]) * 1.08
    if ymax - ymin < 0.08:
        pad = 0.04
        ymin = min([v for v in means if not np.isnan(v)] + [1.0]) - pad
        ymax = max([v for v in means if not np.isnan(v)] + [1.0]) + pad
    ax.set_ylim(ymin, ymax)
    ax.grid(axis="y", alpha=0.25)

    for bar, gran, val in zip(bars, ORDER, means):
        if np.isnan(val):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.3f}x\n{reductions.get(gran, '')}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"{scope}_speedup.{ext}"), dpi=220)
    plt.close(fig)


def plot_metadata(metadata_rows: List[dict], output_dir: str) -> None:
    rows_by_gran = {row["granularity"]: row for row in metadata_rows}
    xs = np.arange(len(ORDER))
    counts = [float(rows_by_gran[g]["scale_count_per_expert"]) for g in ORDER]
    bytes_ = [float(rows_by_gran[g]["fp16_scale_bytes_per_expert"]) for g in ORDER]

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.bar(xs, counts, color="#79A66A", edgecolor="#41683A", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[g] for g in ORDER], rotation=20, ha="right")
    ax.set_xlabel("Output channels per scale")
    ax.set_ylabel("Scale count per OLMoE expert (log)")
    ax.set_title("Scale metadata count")
    ax.grid(axis="y", alpha=0.25, which="both")
    for x, count, b in zip(xs, counts, bytes_):
        ax.text(x, count, f"{int(count)}\n{int(b)} B", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"scale_metadata.{ext}"), dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = read_csv(args.input_csv)
    metadata_rows = read_csv(args.metadata_csv)
    scopes = sorted({row["scope"] for row in rows})
    for scope in scopes:
        plot_speedup(rows, metadata_rows, scope, args.output_dir)
    plot_metadata(metadata_rows, args.output_dir)
    print(f"[plot-output-scale] wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
