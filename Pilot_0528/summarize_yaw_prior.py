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
    "condition", "sequence", "injected_yaw_deg", "yaw_magnitude_deg",
    "actual_prior_yaw_error_deg", "process_status", "trajectory_result",
    "sequence_success", "expected_frames", "successful_frames",
    "frame_completion_rate", "optimizer_failures", "yaw_recovery_frame",
    "divergence_frame", "first_final_yaw_error_deg",
    "second_final_yaw_error_deg", "last_final_yaw_error_deg",
    "median_final_yaw_error_deg", "rmse_final_yaw_error_deg",
    "recall_yaw_1deg", "recall_yaw_3deg", "recall_yaw_5deg",
    "recall_yaw_10deg", "last_final_error_xy_m", "median_final_error_xy_m",
    "rmse_final_error_xy_m", "avg_total_ms",
]

LEVEL_FIELDS = [
    "yaw_magnitude_deg", "runs", "successful_sequences",
    "sequence_success_rate", "complete_processes", "diverged_sequences",
    "no_recovery_sequences", "expected_frames", "successful_frames",
    "frame_completion_rate", "optimizer_failures",
    "median_yaw_recovery_frame", "max_yaw_recovery_frame",
    "median_first_final_yaw_error_deg", "median_second_final_yaw_error_deg",
    "median_last_final_yaw_error_deg", "median_final_yaw_error_deg",
    "rmse_final_yaw_error_deg", "recall_yaw_1deg", "recall_yaw_3deg",
    "recall_yaw_5deg", "recall_yaw_10deg", "median_final_error_xy_m",
    "rmse_final_error_xy_m", "avg_total_ms",
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
        and number(row, "final_yaw_error_deg") is not None
    ]


def first_consecutive_window(rows, predicate, window=10):
    valid = successful_rows(rows)
    for start in range(0, len(valid) - window + 1):
        chunk = valid[start:start + window]
        frames = [int(row["frame"]) for row in chunk]
        if frames != list(range(frames[0], frames[0] + window)):
            continue
        if all(predicate(row) for row in chunk):
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
    yaw_errors = [number(row, "final_yaw_error_deg") for row in valid]
    xy_errors = [number(row, "final_error_xy_m") for row in valid]
    xy_errors = [value for value in xy_errors if value is not None]
    total_ms = []
    for row in valid:
        values = [number(row, key) for key in ("crop_ms", "localization_ms")]
        if all(value is not None for value in values):
            total_ms.append(sum(values))

    recovery_frame = first_consecutive_window(
        rows,
        lambda row: number(row, "final_yaw_error_deg") <= 3.0,
    )
    divergence_frame = first_consecutive_window(
        rows,
        lambda row: (
            number(row, "final_yaw_error_deg") > 30.0
            or number(row, "final_error_xy_m") > 30.0
        ),
    )
    process_status = status.get("status", "missing")
    if process_status != "complete":
        trajectory_result = "process_failed"
    elif divergence_frame != "":
        trajectory_result = "diverged"
    elif recovery_frame == "":
        trajectory_result = "no_yaw_recovery"
    else:
        trajectory_result = "success"

    expected = int(status.get("expected_frames", config.get("expected_frames", 0)))
    successful = max(int(status.get("successful_frames", 0)), len(valid))
    injected_yaw = float(config["prior_yaw_deg"])

    def median(values):
        return float(np.median(values)) if values else ""

    def rmse(values):
        return float(np.sqrt(np.mean(np.square(values)))) if values else ""

    def recall(threshold):
        return sum(error <= threshold for error in yaw_errors) / expected if expected else ""

    return {
        "condition": config_path.relative_to(run_root).parts[0],
        "sequence": config.get("sequence", prefix),
        "injected_yaw_deg": injected_yaw,
        "yaw_magnitude_deg": abs(injected_yaw),
        "actual_prior_yaw_error_deg": config.get("actual_prior_yaw_error_deg", ""),
        "process_status": process_status,
        "trajectory_result": trajectory_result,
        "sequence_success": int(trajectory_result == "success"),
        "expected_frames": expected,
        "successful_frames": successful,
        "frame_completion_rate": successful / expected if expected else "",
        "optimizer_failures": sum(row.get("optimization_success") == "0" for row in rows),
        "yaw_recovery_frame": recovery_frame,
        "divergence_frame": divergence_frame,
        "first_final_yaw_error_deg": yaw_errors[0] if yaw_errors else "",
        "second_final_yaw_error_deg": yaw_errors[1] if len(yaw_errors) > 1 else "",
        "last_final_yaw_error_deg": yaw_errors[-1] if yaw_errors else "",
        "median_final_yaw_error_deg": median(yaw_errors),
        "rmse_final_yaw_error_deg": rmse(yaw_errors),
        "recall_yaw_1deg": recall(1.0),
        "recall_yaw_3deg": recall(3.0),
        "recall_yaw_5deg": recall(5.0),
        "recall_yaw_10deg": recall(10.0),
        "last_final_error_xy_m": xy_errors[-1] if xy_errors else "",
        "median_final_error_xy_m": median(xy_errors),
        "rmse_final_error_xy_m": rmse(xy_errors),
        "avg_total_ms": float(np.mean(total_ms)) if total_ms else "",
        "_rows": rows,
        "_yaw_errors": yaw_errors,
        "_xy_errors": xy_errors,
        "_total_ms": total_ms,
    }


def aggregate_by_level(summaries):
    groups = defaultdict(list)
    for summary in summaries:
        groups[summary["yaw_magnitude_deg"]].append(summary)

    output = []
    for magnitude, group in sorted(groups.items()):
        expected = sum(item["expected_frames"] for item in group)
        successful = sum(item["successful_frames"] for item in group)
        yaw_errors = [value for item in group for value in item["_yaw_errors"]]
        xy_errors = [value for item in group for value in item["_xy_errors"]]
        total_ms = [value for item in group for value in item["_total_ms"]]
        recovery_frames = [
            item["yaw_recovery_frame"] for item in group
            if item["yaw_recovery_frame"] != ""
        ]

        def values(key):
            return [item[key] for item in group if item[key] != ""]

        def median(items):
            return float(np.median(items)) if items else ""

        def recall(threshold):
            return sum(error <= threshold for error in yaw_errors) / expected if expected else ""

        output.append({
            "yaw_magnitude_deg": magnitude,
            "runs": len(group),
            "successful_sequences": sum(item["sequence_success"] for item in group),
            "sequence_success_rate": sum(item["sequence_success"] for item in group) / len(group),
            "complete_processes": sum(item["process_status"] == "complete" for item in group),
            "diverged_sequences": sum(item["trajectory_result"] == "diverged" for item in group),
            "no_recovery_sequences": sum(item["trajectory_result"] == "no_yaw_recovery" for item in group),
            "expected_frames": expected,
            "successful_frames": successful,
            "frame_completion_rate": successful / expected if expected else "",
            "optimizer_failures": sum(item["optimizer_failures"] for item in group),
            "median_yaw_recovery_frame": median(recovery_frames),
            "max_yaw_recovery_frame": max(recovery_frames) if recovery_frames else "",
            "median_first_final_yaw_error_deg": median(values("first_final_yaw_error_deg")),
            "median_second_final_yaw_error_deg": median(values("second_final_yaw_error_deg")),
            "median_last_final_yaw_error_deg": median(values("last_final_yaw_error_deg")),
            "median_final_yaw_error_deg": median(yaw_errors),
            "rmse_final_yaw_error_deg": float(np.sqrt(np.mean(np.square(yaw_errors)))) if yaw_errors else "",
            "recall_yaw_1deg": recall(1.0),
            "recall_yaw_3deg": recall(3.0),
            "recall_yaw_5deg": recall(5.0),
            "recall_yaw_10deg": recall(10.0),
            "median_final_error_xy_m": median(xy_errors),
            "rmse_final_error_xy_m": float(np.sqrt(np.mean(np.square(xy_errors)))) if xy_errors else "",
            "avg_total_ms": float(np.mean(total_ms)) if total_ms else "",
        })
    return output


def save_plots(run_root, summaries, levels):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.ravel()
    magnitudes = sorted({item["yaw_magnitude_deg"] for item in summaries})
    for axis, magnitude in zip(axes, magnitudes):
        for item in summaries:
            if item["yaw_magnitude_deg"] != magnitude:
                continue
            valid = successful_rows(item["_rows"])
            frames = [int(row["frame"]) for row in valid]
            errors = [number(row, "final_yaw_error_deg") for row in valid]
            label = f"{item['condition']}:{item['sequence']}"
            axis.plot(frames, errors, linewidth=0.8, alpha=0.8, label=label)
        axis.axhline(3.0, color="black", linestyle="--", linewidth=0.8)
        axis.set_title(f"Initial yaw error: {magnitude:g} deg")
        axis.set_xlabel("Frame")
        axis.set_ylabel("Final yaw error (deg)")
        axis.grid(alpha=0.25)
        if len(summaries) <= 14:
            axis.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(run_root / "yaw_error_curves.png", dpi=200)
    plt.close(fig)

    x = [item["yaw_magnitude_deg"] for item in levels]
    median_error = [item["median_final_yaw_error_deg"] for item in levels]
    success_rate = [100.0 * item["sequence_success_rate"] for item in levels]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(x, median_error, marker="o")
    axes[0].set_xlabel("Initial yaw error (deg)")
    axes[0].set_ylabel("Median final yaw error (deg)")
    axes[0].grid(alpha=0.25)
    axes[1].plot(x, success_rate, marker="o")
    axes[1].set_ylim(-5, 105)
    axes[1].set_xlabel("Initial yaw error (deg)")
    axes[1].set_ylabel("Successful sequences (%)")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(run_root / "yaw_summary.png", dpi=200)
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
