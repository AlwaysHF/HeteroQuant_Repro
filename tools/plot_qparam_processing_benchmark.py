#!/usr/bin/env python3
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    rows = read_rows(args.input_csv)
    os.makedirs(args.output_dir, exist_ok=True)

    labels = [row["group_size"] for row in rows]
    speedups = [float(row["mean_operator_speedup_vs_group1"]) for row in rows]
    reductions = [float(row["mean_qparam_group_reduction_vs_group1"]) for row in rows]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    bars = ax.bar(x, speedups, width=0.72, color="#D8D8D8", edgecolor="#B8B8B8", label="QParam processing speedup")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35)
    ax.set_xlabel("Output-channel group size")
    ax.set_ylabel("QParam processing speedup (x)")
    ax.set_title("QParam processing operator speedup")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_ylim(0, max(speedups + [1.0]) * 1.15)

    for bar, val, red in zip(bars, speedups, reductions):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.2f}x\n{red:.0f}x fewer",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.legend(loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.output_dir, f"qparam_processing_speedup.{ext}"), dpi=220)
    plt.close(fig)
    print(f"[plot-qparam-process] wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
