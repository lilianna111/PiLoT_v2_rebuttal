#!/usr/bin/env python3

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RUN_FIELDS = [
    "condition", "sequence", "dx_m", "dy_m", "xy_magnitude_m", "status",
    "expected_frames", "successful_frames", "frame_completion_rate",
    "optimizer_failures", "first_final_error_xy_m", "last_final_error_xy_m",
    "median_final_error_xy_m", "rmse_final_error_xy_m",
    "median_final_error_3d_m", "rmse_final_error_3d_m",
    "median_final_yaw_error_deg", "recall_xy_1m", "recall_xy_3m",
    "recall_xy_5m", "recall_xy_10m", "convergence_frame_xy_3m_10frames",
    "avg_total_ms",
]

LEVEL_FIELDS = [
    "xy_magnitude_m", "runs", "complete_runs", "run_completion_rate",
    "expected_frames", "successful_frames", "frame_completion_rate",
    "optimizer_failures", "median_first_final_error_xy_m",
    "median_last_final_error_xy_m", "median_final_error_xy_m",
    "rmse_final_error_xy_m", "median_final_yaw_error_deg",
    "recall_xy_1m", "recall_xy_3m", "recall_xy_5m", "recall_xy_10m",
    "avg_total_ms",
]


def read_json(path):
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def number(row, key):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def successful_rows(rows):
    return [
        row for row in rows
        if row.get("optimization_success") == "1"
        and number(row, "final_error_xy_m") is not None
    ]


def sustained_convergence_frame(rows, threshold=3.0, window=10):
    valid = successful_rows(rows)
    for start in range(0, len(valid) - window + 1):
        chunk = valid[start:start + window]
        frames = [int(row["frame"]) for row in chunk]
        errors = [number(row, "final_error_xy_m") for row in chunk]
        if frames == list(range(frames[0], frames[0] + window)) and all(
            error <= threshold for error in errors
        ):
            return frames[0]
    return ""


def write_csv(path, fields, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_summary(run_root, config_path):
    config = read_json(config_path)
    prefix = config_path.name[:-len("_config.json")]
    status = read_json(config_path.with_name(prefix + "_status.json"))
    rows = read_rows(config_path.with_name(prefix + "_frames.csv"))
    valid = successful_rows(rows)
    xy_errors = [number(row, "final_error_xy_m") for row in valid]
    errors_3d = [number(row, "final_error_3d_m") for row in valid]
    yaw_errors = [number(row, "final_yaw_error_deg") for row in valid]
    total_ms = []
    for row in valid:
        values = [number(row, key) for key in ("crop_ms", "localization_ms")]
        if all(value is not None for value in values):
            total_ms.append(sum(values))

    expected = int(status.get("expected_frames", config.get("expected_frames", 0)))
    successful = int(status.get("successful_frames", len(valid)))
    dx_m = float(config["prior_dx_m"])
    dy_m = float(config["prior_dy_m"])
    condition = config_path.relative_to(run_root).parts[0]

    def median(values):
        return float(np.median(values)) if values else ""

    def rmse(values):
        return float(np.sqrt(np.mean(np.square(values)))) if values else ""

    def recall(threshold):
        return sum(error <= threshold for error in xy_errors) / expected if expected else ""

    summary = {
        "condition": condition,
        "sequence": config.get("sequence", prefix),
        "dx_m": dx_m,
        "dy_m": dy_m,
        "xy_magnitude_m": float(math.hypot(dx_m, dy_m)),
        "status": status.get("status", "missing"),
        "expected_frames": expected,
        "successful_frames": successful,
        "frame_completion_rate": successful / expected if expected else "",
        "optimizer_failures": sum(row.get("optimization_success") == "0" for row in rows),
        "first_final_error_xy_m": xy_errors[0] if xy_errors else "",
        "last_final_error_xy_m": xy_errors[-1] if xy_errors else "",
        "median_final_error_xy_m": median(xy_errors),
        "rmse_final_error_xy_m": rmse(xy_errors),
        "median_final_error_3d_m": median(errors_3d),
        "rmse_final_error_3d_m": rmse(errors_3d),
        "median_final_yaw_error_deg": median(yaw_errors),
        "recall_xy_1m": recall(1.0),
        "recall_xy_3m": recall(3.0),
        "recall_xy_5m": recall(5.0),
        "recall_xy_10m": recall(10.0),
        "convergence_frame_xy_3m_10frames": sustained_convergence_frame(rows),
        "avg_total_ms": float(np.mean(total_ms)) if total_ms else "",
        "_rows": rows,
        "_xy_errors": xy_errors,
        "_errors_3d": errors_3d,
        "_yaw_errors": yaw_errors,
        "_total_ms": total_ms,
    }
    return summary


def aggregate_by_level(summaries):
    groups = defaultdict(list)
    for summary in summaries:
        groups[summary["xy_magnitude_m"]].append(summary)

    output = []
    for magnitude, group in sorted(groups.items()):
        expected = sum(item["expected_frames"] for item in group)
        successful = sum(item["successful_frames"] for item in group)
        xy_errors = [error for item in group for error in item["_xy_errors"]]
        yaw_errors = [error for item in group for error in item["_yaw_errors"]]
        total_ms = [value for item in group for value in item["_total_ms"]]
        first_errors = [item["first_final_error_xy_m"] for item in group if item["first_final_error_xy_m"] != ""]
        last_errors = [item["last_final_error_xy_m"] for item in group if item["last_final_error_xy_m"] != ""]

        def median(values):
            return float(np.median(values)) if values else ""

        def recall(threshold):
            return sum(error <= threshold for error in xy_errors) / expected if expected else ""

        complete_runs = sum(item["status"] == "complete" for item in group)
        output.append({
            "xy_magnitude_m": magnitude,
            "runs": len(group),
            "complete_runs": complete_runs,
            "run_completion_rate": complete_runs / len(group),
            "expected_frames": expected,
            "successful_frames": successful,
            "frame_completion_rate": successful / expected if expected else "",
            "optimizer_failures": sum(item["optimizer_failures"] for item in group),
            "median_first_final_error_xy_m": median(first_errors),
            "median_last_final_error_xy_m": median(last_errors),
            "median_final_error_xy_m": median(xy_errors),
            "rmse_final_error_xy_m": float(np.sqrt(np.mean(np.square(xy_errors)))) if xy_errors else "",
            "median_final_yaw_error_deg": median(yaw_errors),
            "recall_xy_1m": recall(1.0),
            "recall_xy_3m": recall(3.0),
            "recall_xy_5m": recall(5.0),
            "recall_xy_10m": recall(10.0),
            "avg_total_ms": float(np.mean(total_ms)) if total_ms else "",
        })
    return output


def save_plots(run_root, summaries, levels):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=True)
    axes = axes.ravel()
    magnitudes = sorted({item["xy_magnitude_m"] for item in summaries})
    for axis, magnitude in zip(axes, magnitudes):
        for item in summaries:
            if item["xy_magnitude_m"] != magnitude:
                continue
            valid = successful_rows(item["_rows"])
            frames = [int(row["frame"]) for row in valid]
            errors = [number(row, "final_error_xy_m") for row in valid]
            axis.plot(frames, errors, linewidth=1, label=item["condition"])
        axis.axhline(3.0, color="black", linestyle="--", linewidth=0.8)
        axis.set_title(f"Initial ECEF-XY error: {magnitude:g} m")
        axis.set_xlabel("Frame")
        axis.set_ylabel("Final ECEF-XY error (m)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(run_root / "translation_error_curves.png", dpi=200)
    plt.close(fig)

    x = [item["xy_magnitude_m"] for item in levels]
    median_error = [item["median_final_error_xy_m"] for item in levels]
    completion = [100.0 * item["run_completion_rate"] for item in levels]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(x, median_error, marker="o")
    axes[0].set_xlabel("Initial ECEF-XY error (m)")
    axes[0].set_ylabel("Median final ECEF-XY error (m)")
    axes[0].grid(alpha=0.25)
    axes[1].plot(x, completion, marker="o")
    axes[1].set_ylim(-5, 105)
    axes[1].set_xlabel("Initial ECEF-XY error (m)")
    axes[1].set_ylabel("Complete runs (%)")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(run_root / "translation_summary.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    config_paths = sorted(run_root.glob("**/*_config.json"))
    summaries = [run_summary(run_root, path) for path in config_paths]
    if not summaries:
        raise SystemExit(f"No experiment configs found under {run_root}")

    levels = aggregate_by_level(summaries)
    write_csv(
        run_root / "summary_by_run.csv",
        RUN_FIELDS,
        [{key: item[key] for key in RUN_FIELDS} for item in summaries],
    )
    write_csv(run_root / "summary_by_magnitude.csv", LEVEL_FIELDS, levels)
    save_plots(run_root, summaries, levels)
    print(f"Summarized {len(summaries)} runs under {run_root}")


if __name__ == "__main__":
    main()
