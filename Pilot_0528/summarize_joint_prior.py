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
    "condition", "sequence", "dx_m", "dy_m", "translation_m",
    "injected_yaw_deg", "yaw_magnitude_deg", "actual_prior_dx_m",
    "actual_prior_dy_m", "actual_prior_yaw_error_deg", "process_status",
    "trajectory_result", "sequence_success", "baseline_trajectory_result",
    "baseline_stable", "expected_frames",
    "successful_frames", "frame_completion_rate", "optimizer_failures",
    "joint_recovery_frame", "divergence_frame",
    "first_final_error_xy_m", "first_final_yaw_error_deg",
    "second_final_error_xy_m", "second_final_yaw_error_deg",
    "last_final_error_xy_m", "last_final_yaw_error_deg",
    "median_final_error_xy_m", "rmse_final_error_xy_m",
    "median_final_yaw_error_deg", "rmse_final_yaw_error_deg",
    "recall_joint_1", "recall_joint_3", "recall_joint_5",
    "recall_joint_10", "avg_total_ms",
]

LEVEL_FIELDS = [
    "translation_m", "yaw_magnitude_deg", "runs", "successful_sequences",
    "sequence_success_rate", "complete_processes", "process_failed_sequences",
    "diverged_sequences", "no_recovery_sequences", "expected_frames",
    "successful_frames", "frame_completion_rate", "optimizer_failures",
    "baseline_stable_runs", "successful_baseline_stable_runs",
    "conditional_sequence_success_rate", "conditional_process_failed",
    "conditional_diverged", "conditional_no_recovery",
    "median_joint_recovery_frame", "max_joint_recovery_frame",
    "median_first_final_error_xy_m", "median_first_final_yaw_error_deg",
    "median_last_final_error_xy_m", "median_last_final_yaw_error_deg",
    "median_final_error_xy_m", "rmse_final_error_xy_m",
    "median_final_yaw_error_deg", "rmse_final_yaw_error_deg",
    "recall_joint_1", "recall_joint_3", "recall_joint_5",
    "recall_joint_10", "avg_total_ms",
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
    xy_errors = [number(row, "final_error_xy_m") for row in valid]
    yaw_errors = [number(row, "final_yaw_error_deg") for row in valid]
    total_ms = []
    for row in valid:
        values = [number(row, key) for key in ("crop_ms", "localization_ms")]
        if all(value is not None for value in values):
            total_ms.append(sum(values))

    recovery_frame = first_consecutive_window(
        rows,
        lambda row: (
            number(row, "final_error_xy_m") <= 3.0
            and number(row, "final_yaw_error_deg") <= 3.0
        ),
    )
    divergence_frame = first_consecutive_window(
        rows,
        lambda row: (
            number(row, "final_error_xy_m") > 30.0
            or number(row, "final_yaw_error_deg") > 30.0
        ),
    )
    process_status = status.get("status", "missing")
    if process_status != "complete":
        trajectory_result = "process_failed"
    elif divergence_frame != "":
        trajectory_result = "diverged"
    elif recovery_frame == "":
        trajectory_result = "no_joint_recovery"
    else:
        trajectory_result = "success"

    expected = int(status.get("expected_frames", config.get("expected_frames", 0)))
    successful = max(int(status.get("successful_frames", 0)), len(valid))
    dx_m = float(config["prior_dx_m"])
    dy_m = float(config["prior_dy_m"])
    injected_yaw = float(config["prior_yaw_deg"])
    actual_delta = config.get("actual_prior_error_ecef_m", ["", "", ""])

    def item(values, index):
        if not values:
            return ""
        if index < 0:
            index += len(values)
        return values[index] if 0 <= index < len(values) else ""

    def median(values):
        return float(np.median(values)) if values else ""

    def rmse(values):
        return float(np.sqrt(np.mean(np.square(values)))) if values else ""

    def joint_recall(threshold):
        count = sum(
            xy_error <= threshold and yaw_error <= threshold
            for xy_error, yaw_error in zip(xy_errors, yaw_errors)
        )
        return count / expected if expected else ""

    return {
        "condition": config_path.relative_to(run_root).parts[0],
        "sequence": config.get("sequence", prefix),
        "dx_m": dx_m,
        "dy_m": dy_m,
        "translation_m": float(math.hypot(dx_m, dy_m)),
        "injected_yaw_deg": injected_yaw,
        "yaw_magnitude_deg": abs(injected_yaw),
        "actual_prior_dx_m": item(actual_delta, 0),
        "actual_prior_dy_m": item(actual_delta, 1),
        "actual_prior_yaw_error_deg": config.get("actual_prior_yaw_error_deg", ""),
        "process_status": process_status,
        "trajectory_result": trajectory_result,
        "sequence_success": int(trajectory_result == "success"),
        "expected_frames": expected,
        "successful_frames": successful,
        "frame_completion_rate": successful / expected if expected else "",
        "optimizer_failures": sum(row.get("optimization_success") == "0" for row in rows),
        "joint_recovery_frame": recovery_frame,
        "divergence_frame": divergence_frame,
        "first_final_error_xy_m": item(xy_errors, 0),
        "first_final_yaw_error_deg": item(yaw_errors, 0),
        "second_final_error_xy_m": item(xy_errors, 1),
        "second_final_yaw_error_deg": item(yaw_errors, 1),
        "last_final_error_xy_m": item(xy_errors, -1),
        "last_final_yaw_error_deg": item(yaw_errors, -1),
        "median_final_error_xy_m": median(xy_errors),
        "rmse_final_error_xy_m": rmse(xy_errors),
        "median_final_yaw_error_deg": median(yaw_errors),
        "rmse_final_yaw_error_deg": rmse(yaw_errors),
        "recall_joint_1": joint_recall(1.0),
        "recall_joint_3": joint_recall(3.0),
        "recall_joint_5": joint_recall(5.0),
        "recall_joint_10": joint_recall(10.0),
        "avg_total_ms": float(np.mean(total_ms)) if total_ms else "",
        "_rows": rows,
        "_xy_errors": xy_errors,
        "_yaw_errors": yaw_errors,
        "_total_ms": total_ms,
    }


def aggregate_by_level(summaries):
    groups = defaultdict(list)
    for summary in summaries:
        key = (summary["translation_m"], summary["yaw_magnitude_deg"])
        groups[key].append(summary)

    output = []
    for (translation_m, yaw_magnitude), group in sorted(groups.items()):
        expected = sum(item["expected_frames"] for item in group)
        successful = sum(item["successful_frames"] for item in group)
        xy_errors = [value for item in group for value in item["_xy_errors"]]
        yaw_errors = [value for item in group for value in item["_yaw_errors"]]
        total_ms = [value for item in group for value in item["_total_ms"]]
        paired_errors = [
            pair
            for item in group
            for pair in zip(item["_xy_errors"], item["_yaw_errors"])
        ]
        recovery_frames = [
            item["joint_recovery_frame"] for item in group
            if item["sequence_success"] and item["joint_recovery_frame"] != ""
        ]

        def values(key):
            return [item[key] for item in group if item[key] != ""]

        def median(items):
            return float(np.median(items)) if items else ""

        def recall(threshold):
            count = sum(xy <= threshold and yaw <= threshold for xy, yaw in paired_errors)
            return count / expected if expected else ""

        success_count = sum(item["sequence_success"] for item in group)
        baseline_stable = [item for item in group if item["baseline_stable"]]
        conditional_successes = sum(item["sequence_success"] for item in baseline_stable)
        output.append({
            "translation_m": translation_m,
            "yaw_magnitude_deg": yaw_magnitude,
            "runs": len(group),
            "successful_sequences": success_count,
            "sequence_success_rate": success_count / len(group),
            "complete_processes": sum(item["process_status"] == "complete" for item in group),
            "process_failed_sequences": sum(item["trajectory_result"] == "process_failed" for item in group),
            "diverged_sequences": sum(item["trajectory_result"] == "diverged" for item in group),
            "no_recovery_sequences": sum(item["trajectory_result"] == "no_joint_recovery" for item in group),
            "expected_frames": expected,
            "successful_frames": successful,
            "frame_completion_rate": successful / expected if expected else "",
            "optimizer_failures": sum(item["optimizer_failures"] for item in group),
            "baseline_stable_runs": len(baseline_stable),
            "successful_baseline_stable_runs": conditional_successes,
            "conditional_sequence_success_rate": (
                conditional_successes / len(baseline_stable) if baseline_stable else ""
            ),
            "conditional_process_failed": sum(
                item["trajectory_result"] == "process_failed" for item in baseline_stable
            ),
            "conditional_diverged": sum(
                item["trajectory_result"] == "diverged" for item in baseline_stable
            ),
            "conditional_no_recovery": sum(
                item["trajectory_result"] == "no_joint_recovery" for item in baseline_stable
            ),
            "median_joint_recovery_frame": median(recovery_frames),
            "max_joint_recovery_frame": max(recovery_frames) if recovery_frames else "",
            "median_first_final_error_xy_m": median(values("first_final_error_xy_m")),
            "median_first_final_yaw_error_deg": median(values("first_final_yaw_error_deg")),
            "median_last_final_error_xy_m": median(values("last_final_error_xy_m")),
            "median_last_final_yaw_error_deg": median(values("last_final_yaw_error_deg")),
            "median_final_error_xy_m": median(xy_errors),
            "rmse_final_error_xy_m": float(np.sqrt(np.mean(np.square(xy_errors)))) if xy_errors else "",
            "median_final_yaw_error_deg": median(yaw_errors),
            "rmse_final_yaw_error_deg": float(np.sqrt(np.mean(np.square(yaw_errors)))) if yaw_errors else "",
            "recall_joint_1": recall(1.0),
            "recall_joint_3": recall(3.0),
            "recall_joint_5": recall(5.0),
            "recall_joint_10": recall(10.0),
            "avg_total_ms": float(np.mean(total_ms)) if total_ms else "",
        })
    return output


def annotate_baseline_status(summaries):
    baselines = {
        item["sequence"]: item["trajectory_result"]
        for item in summaries
        if item["translation_m"] == 0.0 and item["yaw_magnitude_deg"] == 0.0
    }
    for item in summaries:
        baseline_result = baselines.get(item["sequence"], "missing")
        item["baseline_trajectory_result"] = baseline_result
        item["baseline_stable"] = int(baseline_result == "success")


def error_by_frame(group, key):
    by_frame = defaultdict(list)
    for item in group:
        for row in successful_rows(item["_rows"]):
            value = number(row, key)
            if value is not None:
                by_frame[int(row["frame"])].append(value)
    frames = sorted(by_frame)
    medians = [float(np.median(by_frame[frame])) for frame in frames]
    lower = [float(np.percentile(by_frame[frame], 25)) for frame in frames]
    upper = [float(np.percentile(by_frame[frame], 75)) for frame in frames]
    return frames, medians, lower, upper


def save_plots(run_root, summaries, levels):
    groups = defaultdict(list)
    for summary in summaries:
        groups[(summary["translation_m"], summary["yaw_magnitude_deg"])].append(summary)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for (translation_m, yaw_magnitude), group in sorted(groups.items()):
        label = f"{translation_m:g} m / {yaw_magnitude:g} deg"
        for axis, key in zip(axes, ("final_error_xy_m", "final_yaw_error_deg")):
            frames, medians, lower, upper = error_by_frame(group, key)
            line = axis.plot(frames, medians, linewidth=1.2, label=label)[0]
            axis.fill_between(frames, lower, upper, color=line.get_color(), alpha=0.15)
    axes[0].axhline(3.0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Final ECEF-XY error (m)")
    axes[1].axhline(3.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Final yaw error (deg)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_root / "joint_error_curves.png", dpi=200)
    plt.close(fig)

    labels = [
        f"{item['translation_m']:g}m/{item['yaw_magnitude_deg']:g}deg"
        for item in levels
    ]
    x = np.arange(len(labels))
    success_rate = [100.0 * item["sequence_success_rate"] for item in levels]
    conditional_success_rate = [
        np.nan if item["conditional_sequence_success_rate"] == ""
        else 100.0 * item["conditional_sequence_success_rate"]
        for item in levels
    ]
    recovery = [
        np.nan if item["median_joint_recovery_frame"] == ""
        else item["median_joint_recovery_frame"]
        for item in levels
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    width = 0.38
    axes[0].bar(x - width / 2, success_rate, width, label="All sequences")
    axes[0].bar(
        x + width / 2,
        conditional_success_rate,
        width,
        label="Baseline-stable sequences",
    )
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Successful sequences (%)")
    axes[0].legend(fontsize=8)
    axes[1].bar(x, recovery)
    axes[1].set_ylabel("Median joint recovery frame")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=20)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(run_root / "joint_summary.png", dpi=200)
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

    annotate_baseline_status(summaries)
    levels = aggregate_by_level(summaries)
    public_summaries = [
        {key: item[key] for key in RUN_FIELDS} for item in summaries
    ]
    write_csv(
        run_root / "summary_by_run.csv",
        RUN_FIELDS,
        public_summaries,
    )
    write_csv(
        run_root / "successful_runs.csv",
        RUN_FIELDS,
        [item for item in public_summaries if item["sequence_success"]],
    )
    write_csv(
        run_root / "failed_runs.csv",
        RUN_FIELDS,
        [item for item in public_summaries if not item["sequence_success"]],
    )
    write_csv(run_root / "summary_by_level.csv", LEVEL_FIELDS, levels)
    save_plots(run_root, summaries, levels)
    print(f"Summarized {len(summaries)} runs under {run_root}")


if __name__ == "__main__":
    main()
