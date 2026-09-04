"""Plots for the O5 pick-and-place sweep (results/pickplace_sweep.csv),
matching the style of eval/plot_results.py which covers the offline PBVS
model. Run after run_sweep.py.

    PYTHONPATH=src python3 plot_pickplace_sweep.py
"""
from __future__ import annotations
import argparse, csv, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")  # headless -- no display needed to save PNGs
import matplotlib.pyplot as plt


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_float(v):
    """csv.DictWriter writes a None value as an EMPTY cell (verified:
    writing {'x': None} produces the two-byte cell '' , not the text
    'None'). Runs that produced zero cycles leave mean_cycle_time_s etc.
    as None -- skip those points rather than crashing on float(''). Also
    guards the literal string 'None' in case a different csv writer or a
    hand-edited file ever produces one."""
    if v in (None, "", "None"):
        return None
    return float(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/pickplace_sweep.csv")
    ap.add_argument("--out", default="results/plots")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = read_csv(a.csv)

    metrics = [
        ("pick_success_rate", "pick success rate", "pickplace_success_vs_speed.png"),
        ("throughput_per_min", "throughput (deliveries/min)", "pickplace_throughput_vs_speed.png"),
        ("sorting_accuracy", "sorting accuracy (physical placement audit)", "pickplace_accuracy_vs_speed.png"),
        ("mean_cycle_time_s", "mean cycle time (s)", "pickplace_cycletime_vs_speed.png"),
    ]
    for metric, ylabel, name in metrics:
        groups = defaultdict(list)
        for r in rows:
            y = to_float(r[metric])
            if y is None:
                continue
            groups[float(r["spawn_interval"])].append((float(r["speed"]), y))
        if not groups:
            print(f"skipping {name}: no data points for {metric}")
            continue
        plt.figure(figsize=(7, 4.5))
        for si, vals in sorted(groups.items()):
            vals.sort()
            plt.plot([x for x, _ in vals], [y for _, y in vals],
                      marker="o", label=f"spawn_interval={si:g}s")
        plt.xlabel("belt speed (m/s)")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs belt speed")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(a.out, name), dpi=180)
        plt.close()
        print(f"wrote {os.path.join(a.out, name)}")


if __name__ == "__main__":
    main()
