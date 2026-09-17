#!/usr/bin/env python3
"""Measure independent Coarse/Mid/Fine convergence basins.

The GT TDOM/DSM crop and its sampled 3D points stay fixed.  Each feature
level is initialized independently from the same ECEF-X/Y or yaw-perturbed
pose, so the result measures that level's own basin rather than a cascade.
"""

import argparse
import copy
import csv
import glob
import json
import os
import re
import subprocess
import time
import traceback
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import torch
import yaml

from pixloc.crop.proj2map import generate_ref_map
from pixloc.crop.ray_casting import TargetLocation
from pixloc.crop.transform_colmap import transform_colmap_pose_intrinsic
from pixloc.crop.utils import read_DSM_config
from pixloc.localization import RenderLocalizer
from pixloc.localization.base_refiner import build_c2w_batch
from pixloc.pixlib.datasets.view import read_image
from pixloc.pixlib.geometry import Camera
from pixloc.utils.get_depth import generate_render_camera, get_3D_samples_v3, zero_pad
from pixloc.utils.transform import (
    ECEF_to_WGS84,
    WGS84_to_ECEF,
    euler_angles_to_matrix_ECEF,
    get_rotation_enu_in_ecef,
)


DEFAULT_SEQUENCES = [
    "DJI_20250612182732_0001_V",
    "DJI_20250612183852_0005_V",
    "DJI_20250612194903_0021_V",
    "DJI_20250612193930_0012_V",
    "DJI_20250612194150_0014_V",
    "DJI_20250612194622_0018_V",
]

LEVEL_NAMES = {2: "Coarse", 1: "Mid", 0: "Fine"}
RAW_FIELDS = [
    "sequence", "frame_index", "image", "perturbation", "axis",
    "signed_offset", "magnitude", "level", "level_name", "feature_scale",
    "initial_pixel_displacement", "initial_valid_ratio", "initial_cost",
    "initial_cost_ratio", "initial_feature_cost", "initial_feature_cost_ratio",
    "final_valid_ratio", "final_cost", "optimizer_failed",
    "measured_initial_position_error_m", "measured_initial_yaw_error_deg",
    "final_position_error_m", "final_yaw_error_deg", "final_rotation_error_deg",
    "primary_initial_error", "primary_final_error", "improved",
    "reached_accuracy", "converged",
    "nearest_seed_residual",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="PiLoT_v2 Coarse-to-Fine independent convergence-basin experiment"
    )
    parser.add_argument("sequences", nargs="*", default=DEFAULT_SEQUENCES)
    parser.add_argument("--config", default="configs/feicuiwan_m4t.yaml")
    parser.add_argument("--frames-per-sequence", type=int, default=20)
    parser.add_argument("--sample-num", type=int, default=500)
    parser.add_argument("--translation-max-m", type=float, default=15.0)
    parser.add_argument("--translation-step-m", type=float, default=1.0)
    parser.add_argument("--yaw-max-deg", type=float, default=15.0)
    parser.add_argument("--yaw-step-deg", type=float, default=1.0)
    parser.add_argument("--success-threshold-m", type=float, default=3.0)
    parser.add_argument("--success-threshold-deg", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dataset-root",
        default="/media/amax/AE0E2AFD0E2ABE69/datasets",
    )
    parser.add_argument(
        "--output-base",
        default="/media/amax/PS2000/rebuttal/0915cross2fine",
    )
    return parser.parse_args()


def normalize_name(name):
    base = os.path.basename(name)
    if "_" not in base:
        return os.path.splitext(base)[0] + "_0.png"
    return base


def image_sort_key(path):
    match = re.match(r"(\d+)", Path(path).stem)
    return (int(match.group(1)) if match else 10**12, Path(path).name)


def load_poses(path):
    poses = {}
    ordered = []
    with open(path, "r") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 7:
                continue
            lon, lat, alt, roll, pitch, yaw = map(float, parts[1:7])
            item = {
                "translation": [lon, lat, alt],
                "euler": [pitch, roll, yaw],
            }
            poses[normalize_name(parts[0])] = item
            ordered.append(item)
    if not ordered:
        raise ValueError(f"No poses found in {path}")
    return poses, ordered


def load_auxiliary(path, value_count):
    by_name, ordered = {}, []
    if not os.path.exists(path):
        return by_name, ordered
    with open(path, "r") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < value_count + 1:
                continue
            try:
                values = [float(v) for v in parts[1:1 + value_count]]
            except ValueError:
                continue
            by_name[normalize_name(parts[0])] = values
            ordered.append(values)
    return by_name, ordered


def get_auxiliary(by_name, ordered, image_name, frame_index):
    key = normalize_name(image_name)
    if key in by_name:
        return by_name[key]
    if frame_index < len(ordered):
        return ordered[frame_index]
    return None


def create_output_dir(base):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output = Path(base) / f"run_{stamp}_{os.getpid()}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def build_cameras(config, device):
    default_confs = config["default_confs"]
    cam_query = copy.deepcopy(default_confs["cam_query"])
    resize_ratio = float(cam_query["width"]) / float(cam_query["max_size"])
    fx, fy, cx, cy = cam_query["params"]
    width, height = cam_query["width"], cam_query["height"]
    raw_query_camera = np.array([width, height, cx, cy, fx, fy], dtype=np.float64)
    render_camera_osg = raw_query_camera / resize_ratio

    cam_query["params"] = np.asarray(cam_query["params"], dtype=np.float64) / resize_ratio
    cam_query["width"] = cam_query["width"] / resize_ratio
    cam_query["height"] = cam_query["height"] / resize_ratio
    query_camera = Camera.from_colmap(cam_query).to(device)
    render_camera = generate_render_camera(render_camera_osg).float().to(device)
    return query_camera, render_camera, raw_query_camera, resize_ratio, render_camera_osg


def crop_reference(pose_data, name, map_paths, map_data, work_dir, locator):
    dsm_path, npy_path = map_paths
    query_poses, query_intrinsics, _, _ = transform_colmap_pose_intrinsic(pose_data)
    _, _, area_min_z, _, _, _ = map_data
    return generate_ref_map(
        dsm_path,
        pose_data,
        npy_path,
        map_data[0],
        query_intrinsics,
        query_poses,
        name,
        area_min_z,
        map_data[3],
        map_data[4],
        map_data[5],
        str(work_dir),
        str(work_dir),
        map_data[1],
        area_min_z,
        crop_padding=0,
        debug=False,
        save_to_disk=False,
        locator=locator,
    )


def sample_reference(crop_data, render_camera_osg, padding, sample_num, rng):
    color = crop_data["dom_crop"]
    points3d = crop_data["point_cloud_crop"]
    valid = np.isfinite(points3d).all(axis=-1) & (points3d[..., 2] > 0)
    if not np.any(valid):
        raise ValueError("GT crop has no valid 3D point")

    target_w, target_h = int(render_camera_osg[0]), int(render_camera_osg[1])
    color = cv2.resize(color, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    points3d = cv2.resize(
        points3d.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR
    )
    mask = cv2.resize(
        valid.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST
    )
    if padding:
        pad_h = (16 - target_h % 16) % 16
        pad_w = (16 - target_w % 16) % 16
        if pad_h or pad_w:
            color = cv2.copyMakeBorder(
                color, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE
            )
            points3d = cv2.copyMakeBorder(
                points3d, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE
            )
            mask = cv2.copyMakeBorder(
                mask, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE
            )

    mask = mask.astype(bool)
    points_valid = points3d[mask]
    coords = np.argwhere(mask)
    if len(points_valid) == 0:
        raise ValueError("GT crop has no valid 3D point after resizing")
    count = min(sample_num, len(points_valid))
    chosen = rng.choice(len(points_valid), count, replace=False)
    sampled_points = points_valid[chosen]
    sampled_coords = coords[chosen]
    p2d = np.stack(
        [sampled_coords[:, 1], sampled_coords[:, 0]], axis=1
    ).astype(np.float32)
    return color, sampled_points, torch.from_numpy(p2d).view(1, 1, -1, 2)


def perturbation_values(maximum, step):
    if maximum < 0 or step <= 0:
        raise ValueError("Perturbation maximum must be >= 0 and step must be > 0")
    values = np.arange(step, maximum + 1e-9, step, dtype=np.float64)
    if maximum > 0 and (len(values) == 0 or values[-1] < maximum - 1e-9):
        values = np.r_[values, maximum]
    return values


def nearest_seed_residual(kind, axis, signed_offset):
    # Fixed parameters used by get_3D_samples_v3 at this revision.
    if kind == "translation":
        values = np.arange(-10.0, 10.0 + 1e-9, 2.0)
        grid = np.array([(x, y) for x in values for y in values])
        prior = np.array([
            signed_offset if axis == "X" else 0.0,
            signed_offset if axis == "Y" else 0.0,
        ])
        return float(np.linalg.norm(prior[None] + grid, axis=1).min())
    yaw_seeds = np.array([11, 9, 7, 5, 3, 1, -1, -3, -5, -7, -9, -11])
    wrapped = (signed_offset + yaw_seeds + 180.0) % 360.0 - 180.0
    return float(np.abs(wrapped).min())


def build_candidate_matrices(gt_euler, gt_translation, args):
    gt_matrix = np.asarray(
        euler_angles_to_matrix_ECEF(gt_euler, gt_translation), dtype=np.float64
    )
    matrices, metadata = [], []

    def add(matrix, kind, axis, signed):
        matrices.append(matrix)
        metadata.append({
            "perturbation": kind,
            "axis": axis,
            "signed_offset": float(signed),
            "magnitude": float(abs(signed)),
            "nearest_seed_residual": nearest_seed_residual(kind, axis, signed),
        })

    add(gt_matrix.copy(), "translation", "none", 0.0)
    for magnitude in perturbation_values(args.translation_max_m, args.translation_step_m):
        for axis, component in (("X", 0), ("Y", 1)):
            for sign in (1.0, -1.0):
                candidate = gt_matrix.copy()
                candidate[component, 3] += sign * magnitude
                add(candidate, "translation", axis, sign * magnitude)

    add(gt_matrix.copy(), "yaw", "yaw", 0.0)
    for magnitude in perturbation_values(args.yaw_max_deg, args.yaw_step_deg):
        for sign in (1.0, -1.0):
            euler = list(gt_euler)
            euler[2] += sign * magnitude
            candidate = np.asarray(
                euler_angles_to_matrix_ECEF(euler, gt_translation), dtype=np.float64
            )
            add(candidate, "yaw", "yaw", sign * magnitude)
    return np.stack(matrices), metadata, gt_matrix


def disable_debug_output(localizer):
    optimizers = localizer.refiner.optimizer
    if not isinstance(optimizers, (list, tuple)):
        optimizers = [optimizers]
    for optimizer in optimizers:
        cost_fn = optimizer.cost_fn
        for name in (
            "lm_timing_print", "angle_debug_print", "depth_debug_print",
            "depth_debug_verbose", "depth_verify_query_print",
        ):
            if hasattr(cost_fn, name):
                setattr(cost_fn, name, False)


def prepare_features(refiner, render_frame, query_image, camera_width, render_camera,
                     points3d, p2d, visible):
    render_padded = zero_pad(camera_width, render_frame)
    query_padded = zero_pad(camera_width, query_image)
    features_ref, scales_ref = refiner.dense_feature_extraction(render_padded)
    features_query, scales_query = refiner.dense_feature_extraction(query_padded)
    points3d, p2d, visible, _ = refiner._filter_points_by_uncertainty(
        features_ref, render_camera, points3d, p2d, visible, top_k=500
    )

    features_ref = [feature.to(refiner.device) for feature in features_ref]
    features_query = [feature.to(refiner.device) for feature in features_query]
    if refiner.conf.compute_uncertainty:
        weights_ref = [feature[-1:] for feature in features_ref]
        weights_query = [feature[-1:] for feature in features_query]
        features_ref = [feature[:-1] for feature in features_ref]
        features_query = [feature[:-1] for feature in features_query]
    else:
        weights_ref = weights_query = [None] * len(features_query)
    if refiner.conf.normalize_descriptors:
        features_query = [
            torch.nn.functional.normalize(feature, dim=0)
            for feature in features_query
        ]
    return (
        features_ref, features_query, weights_ref, weights_query,
        scales_ref, scales_query, points3d, p2d, visible,
    )


def scale_reference_points(p2d, feature, render_camera):
    scale = float(feature.shape[-1]) / float(render_camera.size[0])
    scaled = p2d.float().clone().view(-1, 2)
    scaled *= scale
    return scaled.unsqueeze(0)


def pixel_displacements(poses, points3d, camera, zero_indices):
    projected, valid = camera.world2image(poses * points3d)
    output = torch.empty(poses.shape[0], device=projected.device)
    for index in range(poses.shape[0]):
        zero_index = zero_indices[index]
        both = valid[index] & valid[zero_index]
        if both.any():
            output[index] = torch.linalg.norm(
                projected[index, both] - projected[zero_index, both], dim=-1
            ).mean()
        else:
            output[index] = float("nan")
    return output.detach().cpu().numpy()


def wrap_angle(angle):
    return (angle + 180.0) % 360.0 - 180.0


def pose_errors(c2w_matrices, gt_matrix):
    gt_t = gt_matrix[:3, 3]
    gt_r = gt_matrix[:3, :3]
    gt_yaw = Rotation.from_matrix(
        get_rotation_enu_in_ecef(*ECEF_to_WGS84(gt_t)[:2]).T @ gt_r
    ).as_euler("xyz", degrees=True)[2]
    rows = []
    for matrix in c2w_matrices:
        t = matrix[:3, 3]
        r = matrix[:3, :3]
        position_error = float(np.linalg.norm(t - gt_t))
        relative = gt_r.T @ r
        rotation_error = float(np.degrees(np.arccos(np.clip(
            (np.trace(relative) - 1.0) / 2.0, -1.0, 1.0
        ))))
        lon, lat, _ = ECEF_to_WGS84(t)
        local_r = get_rotation_enu_in_ecef(lon, lat).T @ r
        yaw = Rotation.from_matrix(local_r).as_euler("xyz", degrees=True)[2]
        yaw_error = float(abs(wrap_angle(yaw - gt_yaw)))
        rows.append((position_error, yaw_error, rotation_error))
    return rows


def run_level(refiner, level, prepared, query_camera, render_camera, poses,
              render_pose, dd, origin, mul, gt_depth, gt_roll, gt_pitch,
              dsm_path, render_t_ecef, metadata, gt_matrix, args):
    (
        features_ref, features_query, weights_ref, weights_query,
        scales_ref, scales_query, points3d, p2d, visible,
    ) = prepared
    f_ref, f_query = features_ref[level], features_query[level]
    query_camera_level = query_camera.scale(scales_query[level])
    render_camera_level = render_camera.scale(scales_ref[level])
    p2d_level = scale_reference_points(p2d, f_ref, render_camera)
    weights = (
        (weights_ref[level], weights_query[level])
        if refiner.conf.compute_uncertainty else None
    )
    optimizer = refiner.optimizer[level] if isinstance(
        refiner.optimizer, (list, tuple)
    ) else refiner.optimizer

    zero_by_kind = {}
    for index, item in enumerate(metadata):
        if item["magnitude"] == 0:
            zero_by_kind[item["perturbation"]] = index
    zero_indices = [zero_by_kind[item["perturbation"]] for item in metadata]
    displacements = pixel_displacements(
        poses, points3d, query_camera, zero_indices
    )

    final_poses, failed, _ = optimizer.run(
        points3d,
        f_ref,
        f_query,
        poses.to(f_query),
        query_camera_level.to_tensor().to(f_query),
        render_pose.to(f_query),
        render_camera_level.to_tensor().to(f_query),
        W_ref_query=weights,
        p2d_r=p2d_level,
        visible_r=visible,
        gt_depth=gt_depth,
        gt_roll=gt_roll,
        gt_pitch=gt_pitch,
        dd=dd,
        mul=mul,
        origin=origin,
        dsm_path=dsm_path,
        render_T_ecef=render_t_ecef,
        record_initial_diagnostics=True,
    )
    diagnostics = optimizer.last_run_diagnostics
    initial_cost = diagnostics["initial_cost_sum"].numpy()
    initial_feature_cost = diagnostics["initial_feature_cost_sum"].numpy()
    final_cost = diagnostics["final_cost_sum"].numpy()
    initial_valid = diagnostics["initial_valid_ratio"].numpy()
    final_valid = diagnostics["final_valid_ratio"].numpy()
    failed = failed.detach().cpu().numpy().astype(bool)

    baseline_cost = {
        kind: initial_cost[index]
        for kind, index in zero_by_kind.items()
    }
    baseline_feature_cost = {
        kind: initial_feature_cost[index]
        for kind, index in zero_by_kind.items()
    }
    feature_scale = scales_query[level]
    if isinstance(feature_scale, (tuple, list)):
        feature_scale = feature_scale[0]
    initial_c2w = build_c2w_batch(poses, dd, mul, origin).detach().cpu().numpy()
    initial_errors = pose_errors(initial_c2w, gt_matrix)
    c2w = build_c2w_batch(final_poses, dd, mul, origin).detach().cpu().numpy()
    errors = pose_errors(c2w, gt_matrix)
    rows = []
    for index, item in enumerate(metadata):
        kind = item["perturbation"]
        initial_position_error, initial_yaw_error, _ = initial_errors[index]
        position_error, yaw_error, rotation_error = errors[index]
        primary_initial_error = (
            initial_position_error if kind == "translation" else initial_yaw_error
        )
        primary_error = position_error if kind == "translation" else yaw_error
        threshold = (
            args.success_threshold_m
            if kind == "translation" else args.success_threshold_deg
        )
        magnitude = item["magnitude"]
        improved = primary_initial_error > 1e-6 and primary_error < primary_initial_error
        reached_accuracy = (not failed[index]) and primary_error <= threshold
        # A level has captured a pose when it keeps an already acceptable pose
        # inside the target bound, or moves an outside pose toward the solution.
        converged = (not failed[index]) and (
            (primary_initial_error <= threshold and primary_error <= threshold)
            or (primary_initial_error > threshold and primary_error < primary_initial_error)
        )
        denominator = max(abs(float(baseline_cost[kind])), 1e-12)
        feature_denominator = max(abs(float(baseline_feature_cost[kind])), 1e-12)
        rows.append({
            **item,
            "level": level,
            "level_name": LEVEL_NAMES[level],
            "feature_scale": float(feature_scale),
            "initial_pixel_displacement": float(displacements[index]),
            "initial_valid_ratio": float(initial_valid[index]),
            "initial_cost": float(initial_cost[index]),
            "initial_cost_ratio": float(initial_cost[index] / denominator),
            "initial_feature_cost": float(initial_feature_cost[index]),
            "initial_feature_cost_ratio": float(
                initial_feature_cost[index] / feature_denominator
            ),
            "final_valid_ratio": float(final_valid[index]),
            "final_cost": float(final_cost[index]),
            "optimizer_failed": bool(failed[index]),
            "measured_initial_position_error_m": initial_position_error,
            "measured_initial_yaw_error_deg": initial_yaw_error,
            "final_position_error_m": position_error,
            "final_yaw_error_deg": yaw_error,
            "final_rotation_error_deg": rotation_error,
            "primary_initial_error": primary_initial_error,
            "primary_final_error": primary_error,
            "improved": bool(improved),
            "reached_accuracy": bool(reached_accuracy),
            "converged": bool(converged),
        })
    return rows, final_poses


def aggregate_results(raw_path, output):
    data = pd.read_csv(raw_path)
    if data.empty:
        raise RuntimeError("No successful frame was recorded")
    grouped = data.groupby(
        ["perturbation", "magnitude", "level", "level_name"], as_index=False
    ).agg(
        trials=("converged", "size"),
        captured_count=("converged", "sum"),
        capture_rate=("converged", "mean"),
        reached_accuracy_rate=("reached_accuracy", "mean"),
        optimizer_failure_rate=("optimizer_failed", "mean"),
        initial_cost_ratio_mean=("initial_cost_ratio", "mean"),
        initial_cost_ratio_median=("initial_cost_ratio", "median"),
        initial_feature_cost_ratio_mean=("initial_feature_cost_ratio", "mean"),
        initial_feature_cost_ratio_median=("initial_feature_cost_ratio", "median"),
        initial_feature_cost_ratio_std=("initial_feature_cost_ratio", "std"),
        initial_pixel_displacement_mean=("initial_pixel_displacement", "mean"),
        initial_valid_ratio_mean=("initial_valid_ratio", "mean"),
        final_error_median=("primary_final_error", "median"),
        final_error_p90=("primary_final_error", lambda x: x.quantile(0.90)),
    )
    grouped["initial_feature_cost_ratio_ci95"] = (
        1.96
        * grouped["initial_feature_cost_ratio_std"].fillna(0.0)
        / np.sqrt(grouped["trials"].clip(lower=1))
    )
    grouped.to_csv(output / "summary.csv", index=False)
    grouped[grouped["magnitude"].isin([0.0, 5.0, 10.0, 15.0])].to_csv(
        output / "rebuttal_table.csv", index=False
    )

    coverage_rows = []
    for magnitude in sorted(data["magnitude"].unique()):
        trans = data[
            (data.perturbation == "translation") & (data.magnitude == magnitude)
        ]
        yaw = data[(data.perturbation == "yaw") & (data.magnitude == magnitude)]
        coverage_rows.append({
            "prior_magnitude": magnitude,
            "translation_nearest_seed_residual_m": (
                trans.nearest_seed_residual.max() if len(trans) else np.nan
            ),
            "yaw_nearest_seed_residual_deg": (
                yaw.nearest_seed_residual.max() if len(yaw) else np.nan
            ),
        })
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(output / "fixed_sampling_coverage.csv", index=False)

    make_plots(grouped, output)


def make_plots(summary, output):
    colors = {"Coarse": "#1f77b4", "Mid": "#ff7f0e", "Fine": "#2ca02c"}
    for kind, unit in (("translation", "m"), ("yaw", "deg")):
        subset = summary[summary.perturbation == kind]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for level_name in ("Coarse", "Mid", "Fine"):
            curve = subset[subset.level_name == level_name].sort_values("magnitude")
            axes[0].plot(
                curve.magnitude, curve.initial_feature_cost_ratio_mean,
                marker="o", label=level_name, color=colors[level_name],
            )
            axes[0].fill_between(
                curve.magnitude,
                curve.initial_feature_cost_ratio_mean - curve.initial_feature_cost_ratio_ci95,
                curve.initial_feature_cost_ratio_mean + curve.initial_feature_cost_ratio_ci95,
                color=colors[level_name], alpha=0.12,
            )
            axes[1].plot(
                curve.initial_pixel_displacement_mean,
                curve.initial_feature_cost_ratio_mean,
                marker="o", label=level_name, color=colors[level_name],
            )
            axes[2].plot(
                curve.magnitude, 100.0 * curve.capture_rate,
                marker="o", label=level_name, color=colors[level_name],
            )
        axes[0].set_xlabel(f"Initial {kind} error ({unit})")
        axes[0].set_ylabel("Photometric cost / zero-error cost")
        axes[0].grid(alpha=0.3)
        axes[1].set_xlabel("Initial displacement (input-image pixels)")
        axes[1].set_ylabel("Photometric cost / zero-error cost")
        axes[1].grid(alpha=0.3)
        axes[2].set_xlabel(f"Initial {kind} error ({unit})")
        axes[2].set_ylabel("Independent-level capture rate (%)")
        axes[2].set_ylim(-2, 102)
        axes[2].grid(alpha=0.3)
        axes[2].legend()
        fig.tight_layout()
        fig.savefig(output / f"{kind}_basin.png", dpi=200)
        plt.close(fig)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA")
    device = torch.device("cuda")
    output = create_output_dir(args.output_base)
    work_dir = output / "crop_cache"
    work_dir.mkdir()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    query_camera, render_camera, raw_camera, resize_ratio, render_camera_osg = (
        build_cameras(config, device)
    )
    conf = config["default_confs"]["from_render_test"]
    localizer = RenderLocalizer(conf, device=device)
    disable_debug_output(localizer)

    dsm_dir = Path(args.dataset_root) / "DSM/0.3/0.3"
    dom_path = dsm_dir / "DSM0.3_ortho_merge.tif"
    dsm_path = dsm_dir / "DSM0.3_DSM_merge.tif"
    npy_path = dsm_dir / "DSM0.3.npy"
    for required in (dom_path, dsm_path, npy_path):
        if not required.exists():
            raise FileNotFoundError(required)
    map_data = read_DSM_config(str(dsm_path), str(dom_path), str(npy_path))
    locator = TargetLocation({"ray_casting": {}}, use_dsm=False)

    settings = vars(args).copy()
    settings.update({
        "git_revision": git_revision(),
        "method": "independent level cost/basin analysis from identical perturbed poses",
        "translation_coordinates": "direct ECEF X/Y; ECEF Z unchanged",
        "yaw_coordinates": "local ENU camera yaw; roll/pitch unchanged",
        "map_crop_pose": "ground truth and fixed for all perturbations",
        "fixed_translation_sampling": {"range_m": 10, "step_m": 2},
        "fixed_yaw_offsets_deg": [11, 9, 7, 5, 3, 1, -1, -3, -5, -7, -9, -11],
        "fixed_prefilter_keep": int(os.environ.get("PILOT_PREFILTER_KEEP", "256")),
        "fixed_coarse_to_mid_keep": 128,
    })
    with open(output / "experiment_config.json", "w") as handle:
        json.dump(settings, handle, indent=2)
    with open(output / "run_status.json", "w") as handle:
        json.dump({"status": "running", "completed_frames": 0}, handle, indent=2)

    raw_path = output / "raw_results.csv"
    failure_path = output / "frame_failures.csv"
    completed_frames = 0
    with open(raw_path, "w", newline="") as raw_handle, open(
        failure_path, "w", newline=""
    ) as failure_handle:
        writer = csv.DictWriter(raw_handle, fieldnames=RAW_FIELDS)
        writer.writeheader()
        failure_writer = csv.DictWriter(
            failure_handle, fieldnames=["sequence", "frame_index", "image", "error"]
        )
        failure_writer.writeheader()

        for sequence_index, sequence in enumerate(args.sequences):
            pose_path = Path(args.dataset_root) / "poses" / f"{sequence}.txt"
            image_dir = Path(args.dataset_root) / "images" / sequence
            poses, ordered_poses = load_poses(pose_path)
            origin = torch.as_tensor(
                WGS84_to_ECEF(ordered_poses[0]["translation"]),
                device=device,
                dtype=torch.float64,
            )
            images = sorted(
                glob.glob(str(image_dir / "*.png"))
                + glob.glob(str(image_dir / "*.jpg"))
                + glob.glob(str(image_dir / "*.JPG")),
                key=image_sort_key,
            )
            eligible = [
                (index, path, poses[normalize_name(path)])
                for index, path in enumerate(images)
                if normalize_name(path) in poses
            ]
            if not eligible:
                raise ValueError(f"No image/pose pairs for {sequence}")
            select_count = min(args.frames_per_sequence, len(eligible))
            selected = np.unique(
                np.linspace(0, len(eligible) - 1, select_count).round().astype(int)
            )

            depth_by_name, depth_ordered = load_auxiliary(
                str(Path(args.dataset_root) / "depth_1" / f"{sequence}.txt"), 1
            )
            angle_by_name, angle_ordered = load_auxiliary(
                str(Path(args.dataset_root) / "angle" / f"{sequence}.txt"), 2
            )
            print(f"{sequence}: selected {len(selected)} / {len(eligible)} frames")

            for selected_index in selected:
                frame_index, image_path, pose = eligible[int(selected_index)]
                image_name = os.path.basename(image_path)
                try:
                    gt_euler = pose["euler"]
                    gt_translation = pose["translation"]
                    pose_data = list(gt_translation) + [
                        gt_euler[1], gt_euler[0], gt_euler[2]
                    ]
                    crop_data = crop_reference(
                        pose_data,
                        f"{sequence}_{frame_index}",
                        (str(dsm_path), str(npy_path)),
                        map_data,
                        work_dir,
                        locator,
                    )
                    rng = np.random.default_rng(
                        args.seed + sequence_index * 100000 + frame_index
                    )
                    render_frame, sampled_points, p2d = sample_reference(
                        crop_data,
                        render_camera_osg,
                        config["default_confs"]["padding"],
                        args.sample_num,
                        rng,
                    )
                    query_image = read_image(
                        image_path,
                        scale=resize_ratio,
                        distortion=config["default_confs"]["cam_query"]["distortion"],
                        query_camera=raw_camera,
                    )
                    matrices, metadata, gt_matrix = build_candidate_matrices(
                        gt_euler, gt_translation, args
                    )
                    render_matrix = torch.as_tensor(
                        gt_matrix, device=device, dtype=torch.float64
                    )
                    points3d, render_pose, poses_initial, dd, p2d, render_t_ecef = (
                        get_3D_samples_v3(
                            mkpts_r=p2d.to(device),
                            depth_mat=torch.as_tensor(sampled_points, device=device),
                            T_c2w=render_matrix,
                            camera=render_camera,
                            euler_angles=gt_euler,
                            translation=gt_translation,
                            query_euler_angles=gt_euler,
                            query_translation=gt_translation,
                            origin=origin,
                            T_query_in_ECEF_c2w=matrices,
                            mul=config["default_confs"]["refine"]["mul"],
                        )
                    )
                    visible = torch.ones(
                        (1, p2d.view(-1, 2).shape[0]), device=device, dtype=torch.bool
                    )
                    prepared = prepare_features(
                        localizer.refiner,
                        render_frame,
                        query_image,
                        int(query_camera.size[0].item()),
                        render_camera,
                        points3d,
                        p2d,
                        visible,
                    )
                    depth = get_auxiliary(
                        depth_by_name, depth_ordered, image_name, frame_index
                    )
                    angles = get_auxiliary(
                        angle_by_name, angle_ordered, image_name, frame_index
                    )
                    gt_depth = depth[0] if depth else None
                    gt_roll, gt_pitch = angles if angles else (None, None)

                    frame_rows = []
                    for level in (2, 1, 0):
                        level_rows, _ = run_level(
                            localizer.refiner,
                            level,
                            prepared,
                            query_camera,
                            render_camera,
                            poses_initial,
                            render_pose,
                            dd,
                            origin,
                            config["default_confs"]["refine"]["mul"],
                            gt_depth,
                            gt_roll,
                            gt_pitch,
                            str(dsm_path),
                            render_t_ecef,
                            metadata,
                            gt_matrix,
                            args,
                        )
                        frame_rows.extend(level_rows)

                    for row in frame_rows:
                        writer.writerow({
                            "sequence": sequence,
                            "frame_index": frame_index,
                            "image": image_name,
                            **row,
                        })
                    raw_handle.flush()
                    completed_frames += 1
                    print(
                        f"  frame {frame_index}: {len(frame_rows)} rows saved "
                        f"({completed_frames} frames complete)"
                    )
                except Exception as error:
                    failure_writer.writerow({
                        "sequence": sequence,
                        "frame_index": frame_index,
                        "image": image_name,
                        "error": repr(error),
                    })
                    failure_handle.flush()
                    print(f"  frame {frame_index}: FAILED: {error}")
                    traceback.print_exc()
                finally:
                    torch.cuda.empty_cache()

    aggregate_results(raw_path, output)
    with open(output / "run_status.json", "w") as handle:
        json.dump({
            "status": "complete",
            "completed_frames": completed_frames,
            "requested_frames": args.frames_per_sequence * len(args.sequences),
        }, handle, indent=2)
    print(f"Coarse-to-Fine basin experiment saved to: {output}")


if __name__ == "__main__":
    main()
