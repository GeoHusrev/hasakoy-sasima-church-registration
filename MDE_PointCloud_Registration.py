"""
Public release script for the registration workflow prepared for the manuscript:

    Robust Multi-Sensor Point Cloud Registration for Cultural Heritage Documentation:
    A Multi-Population Differential Evolution Approach

Code author
-----------
Dr. Ahmet Emin Karkınlı
Department of Geomatics Engineering, Faculty of Engineering,
Nigde Omer Halisdemir University, Nigde 51240, Turkiye
E-mail: akarkinli@ohu.edu.tr

Related manuscript authors
--------------------------
Artur Janowski; Ahmet Emin Karkınlı; Leyla Kaderli;
Betul Gul Husrevoglu; Mustafa Husrevoglu

Purpose of this public code release
-----------------------------------
This script was prepared as the public registration-pipeline implementation for the
manuscript listed above. It reproduces the main computational workflow reported in the
study for the registration of TLS and UAV-derived point clouds from the Hasakoy (Sasima)
Church case study.

The script reproduces the main computational pipeline described in the manuscript:
    1. Independent centroid translation of the source and target point clouds.
    2. Preprocessing with voxel downsampling and normal estimation.
    3. Coarse alignment using FPFH + RANSAC.
    4. Fine alignment using two alternatives started from the same coarse transform:
       - Trimmed ICP (baseline)
       - Multi-population Differential Evolution, MDE (proposed)
    5. Repeated experiments, descriptive statistics, paired significance testing,
       and convergence plots.

Primary MDE source
------------------
The MDE component implemented in this script follows the algorithmic basis reported in:

    Karkinli, A.E. Detection of object boundary from point cloud by using
    multi-population based differential evolution algorithm.
    Neural Computing and Applications 2023, 35, 5193-5206.
    https://doi.org/10.1007/s00521-022-07969-w

Notes
-----
- The script works with LAS/LAZ point clouds directly. Any format supported by
  Open3D can also be used.
- Registration is performed in a centroid-shifted coordinate system, matching the
  manuscript workflow. Final transforms are additionally converted back to the
  original coordinate system and saved.
- The public code is restricted to the registration pipeline. Data generation,
  UAV photogrammetric reconstruction, and manufacturer-specific TLS station merging
  are not included here.
  
Disclaimer: 
    This code is provided for research and academic use only, without any express or implied warranty. 
    While reasonable efforts were made to ensure consistency with the methodology described in the associated manuscript, 
    the authors make no guarantee that the code will run without modification 
    in all computing environments or reproduce identical numerical results under all conditions. 
    Users are responsible for verifying the suitability of the code for their own data, 
    hardware, software environment, and research purposes.
"""

from __future__ import annotations

__title__ = "Robust Multi-Sensor Point Cloud Registration for Cultural Heritage Documentation: A Multi-Population Differential Evolution Approach"
__code_author__ = "Ahmet Emin Karkinli"
__affiliation__ = "Department of Geomatics Engineering, Faculty of Engineering, Nigde Omer Halisdemir University, Nigde 51240, Turkiye"
__email__ = "akarkinli@ohu.edu.tr"
__mde_reference__ = (
    "Karkinli, A.E. Detection of object boundary from point cloud by using multi-population "
    "based differential evolution algorithm. Neural Computing and Applications 2023, 35, "
    "5193-5206. https://doi.org/10.1007/s00521-022-07969-w"
)

import argparse
import copy
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from scipy.stats import rankdata


@dataclass
class Config:
    """Configuration aligned with the manuscript."""

    source_tls_path: str = "source_tls.las"
    target_uav_path: str = "target_uav.las"
    results_dir: str = "results_registration_pipeline"

    voxel_coarse: float = 1.0
    voxel_fine: float = 0.05
    normal_radius_multiplier: float = 5.0
    normal_max_nn: int = 50

    fpfh_radius_multiplier: float = 5.0
    ransac_distance_multiplier: float = 2.0
    ransac_edge_length_checker: float = 0.90
    ransac_mutual_filter: bool = True
    ransac_n: int = 3
    ransac_max_iteration: int = 100000
    ransac_confidence: float = 0.999

    tricp_max_iter: int = 50
    tricp_trim_ratio: float = 0.90
    tricp_max_corr_dist: float = 2.0
    tricp_tol_rot: float = 1e-5
    tricp_tol_trans: float = 1e-4

    mde_population_size: int = 10
    mde_dimension: int = 6
    mde_max_cycle: int = 250
    mde_sample_size: int = 2000
    mde_early_stop_patience: int = 30
    mde_jobs: Optional[int] = None

    n_runs: int = 30
    figure_dpi: int = 600
    figure_size: Tuple[float, float] = (8.0, 6.0)
    snapshot_sample_size: int = 90000

    trim_for_reporting: float = 0.90
    random_seed: Optional[int] = None
    save_snapshots_for_first_run: bool = True


def ensure_directories(config: Config) -> Dict[str, str]:
    """Create the output folder structure."""
    results_dir = config.results_dir
    figs_dir = os.path.join(results_dir, "figures")
    tables_dir = os.path.join(results_dir, "tables")
    transforms_dir = os.path.join(results_dir, "transforms")
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(transforms_dir, exist_ok=True)
    return {
        "results": results_dir,
        "figures": figs_dir,
        "tables": tables_dir,
        "transforms": transforms_dir,
    }


def set_global_seed(seed: Optional[int]) -> None:
    """Set the NumPy random seed if a deterministic seed is provided."""
    if seed is not None:
        np.random.seed(seed)


def load_point_cloud(file_path: str) -> o3d.geometry.PointCloud:
    """Load a point cloud from a LAS/LAZ file or any Open3D-supported format."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if file_path.lower().endswith((".las", ".laz")):
        las = laspy.read(file_path)
        points = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)

        try:
            if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
                colors = np.vstack((las.red, las.green, las.blue)).T.astype(np.float64)
                max_val = float(np.max(colors))
                if max_val > 0.0:
                    colors /= max_val
                    point_cloud.colors = o3d.utility.Vector3dVector(colors)
        except Exception:
            pass
    else:
        point_cloud = o3d.io.read_point_cloud(file_path)

    if len(point_cloud.points) == 0:
        raise ValueError(f"No points were loaded from: {file_path}")

    return point_cloud


def center_point_cloud(
    point_cloud: o3d.geometry.PointCloud,
) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """Return a deep-copied point cloud translated to its centroid and the original centroid."""
    centered = copy.deepcopy(point_cloud)
    center = np.asarray(centered.get_center(), dtype=np.float64)
    centered.translate(-center)
    return centered, center


def prepare_downsampled_cloud(
    point_cloud: o3d.geometry.PointCloud,
    voxel_size: float,
    normal_radius_multiplier: float,
    normal_max_nn: int,
) -> o3d.geometry.PointCloud:
    """Downsample a point cloud and estimate normals."""
    down = point_cloud.voxel_down_sample(voxel_size)
    radius = voxel_size * normal_radius_multiplier
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=normal_max_nn)
    )
    return down


def convert_centered_transform_to_original(
    transform_centered: np.ndarray,
    source_center: np.ndarray,
    target_center: np.ndarray,
) -> np.ndarray:
    """
    Convert a transform estimated on independently centered clouds back to the
    original coordinate system.

    If x_s^c = x_s - c_s and x_t^c = x_t - c_t, then
        x_t^c = R x_s^c + t
    implies
        x_t = R x_s + (t - R c_s + c_t).
    """
    transform_original = np.eye(4, dtype=np.float64)
    rotation = transform_centered[:3, :3]
    translation = transform_centered[:3, 3]
    transform_original[:3, :3] = rotation
    transform_original[:3, 3] = translation - rotation @ source_center + target_center
    return transform_original


def rmse_bidirectional_trimmed(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transform: np.ndarray,
    trim_ratio: float = 0.90,
) -> float:
    """
    Bidirectional trimmed RMSE used for reported results and coarse-alignment checks.
    The source cloud is transformed, nearest-neighbor distances are computed in both
    directions, the worst 10% are discarded, and the RMSE of the retained distances is
    returned.
    """
    source_t = copy.deepcopy(source)
    target_t = copy.deepcopy(target)
    source_t.transform(transform)

    d_source_to_target = np.asarray(source_t.compute_point_cloud_distance(target_t))
    d_target_to_source = np.asarray(target_t.compute_point_cloud_distance(source_t))
    distances = np.concatenate([d_source_to_target, d_target_to_source])

    if distances.size == 0:
        return float("inf")

    distances_sorted = np.sort(distances)
    n_keep = max(1, int(len(distances_sorted) * trim_ratio))
    return float(np.sqrt(np.mean(distances_sorted[:n_keep] ** 2)))


def rmse_one_way(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transform: np.ndarray,
) -> float:
    """
    One-way RMSE used only to define adaptive MDE search bounds, following the manuscript.
    """
    source_t = copy.deepcopy(source)
    source_t.transform(transform)
    distances = np.asarray(source_t.compute_point_cloud_distance(target))
    if distances.size == 0:
        return float("inf")
    return float(np.sqrt(np.mean(distances ** 2)))


def global_registration_fpfh_ransac(
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    config: Config,
) -> np.ndarray:
    """
    Coarse alignment with FPFH + RANSAC, including edge-length and distance checkers.
    """
    radius_feature = config.voxel_coarse * config.fpfh_radius_multiplier
    distance_threshold = config.voxel_coarse * config.ransac_distance_multiplier

    print("Computing FPFH features for coarse registration...")
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )

    print("Running RANSAC feature matching...")
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        config.ransac_mutual_filter,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        config.ransac_n,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                config.ransac_edge_length_checker
            ),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(
            config.ransac_max_iteration,
            config.ransac_confidence,
        ),
    )

    print(
        f"RANSAC completed. Fitness = {result.fitness:.6f}, "
        f"inlier RMSE = {result.inlier_rmse:.6f}"
    )
    return result.transformation


def svd_rigid_transform(
    source_points: np.ndarray,
    target_points: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the rigid transform that maps source_points to target_points."""
    if weights is None:
        weights = np.ones(source_points.shape[0], dtype=np.float64)

    weights = np.asarray(weights, dtype=np.float64)
    w_sum = float(np.sum(weights))
    if w_sum <= 0.0:
        return np.eye(3), np.zeros(3)

    mu_source = np.sum(source_points * weights[:, None], axis=0) / w_sum
    mu_target = np.sum(target_points * weights[:, None], axis=0) / w_sum

    source_centered = source_points - mu_source
    target_centered = target_points - mu_target
    covariance = source_centered.T @ (target_centered * weights[:, None])

    u_mat, _, vt_mat = np.linalg.svd(covariance)
    rotation = vt_mat.T @ u_mat.T
    if np.linalg.det(rotation) < 0:
        vt_mat[2, :] *= -1.0
        rotation = vt_mat.T @ u_mat.T

    translation = mu_target - rotation @ mu_source
    return rotation, translation


def trimmed_icp(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    config: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic Trimmed ICP used as the manuscript baseline.
    """
    source_points = np.asarray(source_pcd.points)
    target_points = np.asarray(target_pcd.points)
    target_tree = cKDTree(target_points)

    transform = init_transform.copy()
    history: List[float] = []

    for iteration in range(config.tricp_max_iter):
        rotation_current = transform[:3, :3]
        translation_current = transform[:3, 3]
        source_transformed = (rotation_current @ source_points.T).T + translation_current

        distances, indices = target_tree.query(source_transformed, k=1)
        valid_mask = distances < config.tricp_max_corr_dist
        if not np.any(valid_mask):
            print("[TR-ICP] No valid correspondences under the distance threshold.")
            break

        src_corr = source_transformed[valid_mask]
        tgt_corr = target_points[indices[valid_mask]]
        dist_corr = distances[valid_mask]

        n_corr = len(dist_corr)
        n_keep = max(3, int(n_corr * config.tricp_trim_ratio))
        keep_indices = np.argsort(dist_corr)[:n_keep]

        src_keep = src_corr[keep_indices]
        tgt_keep = tgt_corr[keep_indices]
        dist_keep = dist_corr[keep_indices]

        rmse_iter = float(np.sqrt(np.mean(dist_keep ** 2)))
        history.append(rmse_iter)
        print(
            f"[TR-ICP] Iteration {iteration + 1:03d}: "
            f"RMSE = {rmse_iter:.6f} m, retained pairs = {n_keep}/{n_corr}"
        )

        rotation_update, translation_update = svd_rigid_transform(src_keep, tgt_keep)

        transform_new = np.eye(4, dtype=np.float64)
        transform_new[:3, :3] = rotation_update @ rotation_current
        transform_new[:3, 3] = rotation_update @ translation_current + translation_update

        angle = math.acos(
            max(-1.0, min(1.0, (np.trace(rotation_update) - 1.0) / 2.0))
        )
        translation_step = float(np.linalg.norm(translation_update))

        transform = transform_new
        if angle < config.tricp_tol_rot and translation_step < config.tricp_tol_trans:
            print("[TR-ICP] Converged.")
            break

    return transform, np.asarray(history, dtype=np.float64)


def build_adaptive_bounds(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Adaptive MDE bounds based on the one-way RMSE after coarse alignment.
    """
    initial_rmse = rmse_one_way(source_pcd, target_pcd, init_transform)
    if not np.isfinite(initial_rmse):
        initial_rmse = 5.0

    if initial_rmse < 1.0:
        translation_range = initial_rmse * 2.0
        rotation_range = 1.0
    elif initial_rmse < 5.0:
        translation_range = initial_rmse * 1.5
        rotation_range = 2.0
    else:
        translation_range = initial_rmse
        rotation_range = 5.0

    low = np.array(
        [
            -translation_range,
            -translation_range,
            -translation_range,
            -rotation_range,
            -rotation_range,
            -rotation_range,
        ],
        dtype=np.float64,
    )
    up = np.array(
        [
            translation_range,
            translation_range,
            translation_range,
            rotation_range,
            rotation_range,
            rotation_range,
        ],
        dtype=np.float64,
    )

    return low, up, float(initial_rmse), float(translation_range), float(rotation_range)


def border_control(candidates: np.ndarray, low: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Clip candidate solutions to the adaptive search bounds."""
    return np.clip(candidates, low, up)


def _fitness_single_candidate(
    params: np.ndarray,
    source_sample: np.ndarray,
    target_tree: cKDTree,
    coarse_transform: np.ndarray,
    keep_ratio: float,
) -> float:
    """Evaluate one MDE candidate using a trimmed MSE fitness."""
    tx, ty, tz, rx, ry, rz = params
    rotation_fine = R.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
    translation_fine = np.array([tx, ty, tz], dtype=np.float64)

    rotation_coarse = coarse_transform[:3, :3]
    translation_coarse = coarse_transform[:3, 3]

    rotation_total = rotation_coarse @ rotation_fine
    translation_total = rotation_coarse @ translation_fine + translation_coarse

    source_transformed = (rotation_total @ source_sample.T).T + translation_total
    distances, _ = target_tree.query(source_transformed, k=1)
    if distances.size == 0:
        return float("inf")

    squared = np.sort(distances ** 2)
    n_keep = max(1, int(len(squared) * keep_ratio))
    return float(np.mean(squared[:n_keep]))


def fitness_eval_mde_parallel(
    params_batch: np.ndarray,
    source_points_full: np.ndarray,
    target_points: np.ndarray,
    coarse_transform: np.ndarray,
    sample_size: int,
    keep_ratio: float,
    n_jobs: Optional[int],
) -> np.ndarray:
    """
    Parallel robust fitness evaluation used by MDE.
    """
    if sample_size > 0 and sample_size < len(source_points_full):
        sample_indices = np.random.choice(len(source_points_full), sample_size, replace=False)
        source_sample = source_points_full[sample_indices, :]
    else:
        source_sample = source_points_full

    if len(target_points) == 0:
        return np.full(params_batch.shape[0], np.inf, dtype=np.float64)

    target_tree = cKDTree(target_points)
    if n_jobs is None or n_jobs < 1:
        n_jobs = os.cpu_count() or 1

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(
                _fitness_single_candidate,
                params_batch[i],
                source_sample,
                target_tree,
                coarse_transform,
                keep_ratio,
            )
            for i in range(params_batch.shape[0])
        ]
        scores = np.array([future.result() for future in futures], dtype=np.float64)

    return scores


def mde_registration(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    coarse_transform: np.ndarray,
    config: Config,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Original Karkinli-style MDE with adaptive bounds, trimmed fitness,
    parallel evaluation, and early stopping.
    """
    if seed is not None:
        np.random.seed(seed)

    source_points = np.asarray(source_pcd.points)
    target_points = np.asarray(target_pcd.points)

    low, up, initial_rmse, translation_range, rotation_range = build_adaptive_bounds(
        source_pcd, target_pcd, coarse_transform
    )
    print(f"[MDE] One-way RMSE for adaptive bounds: {initial_rmse:.6f} m")
    print(
        f"[MDE] Search bounds: translation ±{translation_range:.3f} m, "
        f"rotation ±{rotation_range:.3f} degrees"
    )

    n = config.mde_population_size
    d = config.mde_dimension
    t_multi = 3

    population = np.random.rand(t_multi * n, d) * (up - low) + low
    fitness = fitness_eval_mde_parallel(
        population,
        source_points,
        target_points,
        coarse_transform,
        config.mde_sample_size,
        config.tricp_trim_ratio,
        config.mde_jobs,
    )

    best_index = int(np.argmin(fitness))
    best_solution = population[best_index, :].copy()
    best_value = float(fitness[best_index])
    temp = population[:n, :].copy()
    noise = np.zeros((n, d), dtype=np.float64)
    history: List[float] = []
    no_improvement_count = 0

    print(f"[MDE] Initial best fitness: {best_value:.10e}")

    for cycle in range(1, config.mde_max_cycle + 1):
        selected_indices = np.random.permutation(t_multi * n)[:n]
        sub_population = population[selected_indices, :].copy()
        sub_fitness = fitness[selected_indices].copy()

        for i in range(n):
            scale = abs(
                np.random.randint(0, 2)
                - (np.random.rand() ** np.random.randint(1, 11))
            ) * (np.random.randn() ** np.random.randint(1, 6))

            while True:
                pair = np.random.permutation(n)[:2]
                if not np.any(pair == i):
                    break

            for j in range(d):
                dx = sub_population[pair[0], j] if np.random.rand() < 0.5 else best_solution[j]
                dy = sub_population[i, j] if np.random.rand() < 0.5 else sub_population[pair[1], j]
                temp[i, j] = sub_population[pair[1], j] + scale * (dx - dy)

        if (np.random.rand() ** np.random.randint(1, 6)) < 0.5:
            c = 1
        else:
            c = d

        random_binary = np.random.randint(0, 2, size=(n, c))
        random_base = np.random.rand(n, d)
        random_power = np.random.randint(1, 6, size=(n, c))
        power_matrix = random_base ** random_power
        map_matrix = np.abs(random_binary - power_matrix) < 0.5
        if map_matrix.shape[1] == 1:
            map_matrix = np.repeat(map_matrix, d, axis=1)

        trials = sub_population + map_matrix * (temp + noise - sub_population)
        trials = border_control(trials, low, up)

        trial_fitness = fitness_eval_mde_parallel(
            trials,
            source_points,
            target_points,
            coarse_transform,
            config.mde_sample_size,
            config.tricp_trim_ratio,
            config.mde_jobs,
        )

        improved = trial_fitness < sub_fitness
        sub_fitness[improved] = trial_fitness[improved]
        sub_population[improved, :] = trials[improved, :]

        population[selected_indices, :] = sub_population
        fitness[selected_indices] = sub_fitness

        current_best_index = int(np.argmin(fitness))
        current_best_value = float(fitness[current_best_index])
        if current_best_value < best_value:
            best_value = current_best_value
            best_solution = population[current_best_index, :].copy()
            no_improvement_count = 0
            print(f"[MDE] Improved at cycle {cycle:03d}: {best_value:.10e}")
        else:
            no_improvement_count += 1

        history.append(best_value)

        if no_improvement_count >= config.mde_early_stop_patience:
            print(
                f"[MDE] Early stopping at cycle {cycle} after "
                f"{config.mde_early_stop_patience} cycles without improvement."
            )
            break

        exponent_matrix = np.random.randint(-12, -8, size=(n, d))
        noise = sub_population * (10.0 ** exponent_matrix) * (np.random.rand(n, 1) - 0.5)

    delta_transform = np.eye(4, dtype=np.float64)
    delta_transform[:3, :3] = R.from_euler("xyz", best_solution[3:], degrees=True).as_matrix()
    delta_transform[:3, 3] = best_solution[:3]

    metadata = {
        "history": np.asarray(history, dtype=np.float64),
        "best_solution": np.asarray(best_solution, dtype=np.float64),
        "best_fitness": np.array([best_value], dtype=np.float64),
    }
    return delta_transform, metadata


def save_overlay_snapshot(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    transform: np.ndarray,
    output_path: str,
    title: str,
    sample_size: int,
    dpi: int,
) -> None:
    """Save a 2D XY projection of the aligned point clouds."""
    source_plot = copy.deepcopy(source_pcd)
    target_plot = copy.deepcopy(target_pcd)
    source_plot.transform(transform)

    source_points = np.asarray(source_plot.points)
    target_points = np.asarray(target_plot.points)

    n_source = min(len(source_points), sample_size)
    n_target = min(len(target_points), sample_size)
    source_indices = (
        np.random.choice(len(source_points), n_source, replace=False)
        if len(source_points) > n_source
        else np.arange(len(source_points))
    )
    target_indices = (
        np.random.choice(len(target_points), n_target, replace=False)
        if len(target_points) > n_target
        else np.arange(len(target_points))
    )

    source_sub = source_points[source_indices]
    target_sub = target_points[target_indices]

    plt.figure(figsize=(10, 8))
    target_colors = np.asarray(target_plot.colors)
    if len(target_colors) == len(target_points):
        plt.scatter(
            target_sub[:, 0],
            target_sub[:, 1],
            s=0.4,
            c=target_colors[target_indices],
            label="UAV target",
        )
    else:
        plt.scatter(
            target_sub[:, 0],
            target_sub[:, 1],
            s=0.4,
            c="gray",
            label="UAV target",
        )

    plt.scatter(
        source_sub[:, 0],
        source_sub[:, 1],
        s=0.4,
        c="red",
        label="TLS source",
    )
    plt.axis("equal")
    plt.legend(markerscale=6)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def pad_and_average(histories: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Pad variable-length convergence histories and return mean and standard deviation."""
    max_len = max(len(h) for h in histories)
    padded = np.full((len(histories), max_len), np.nan, dtype=np.float64)
    for i, history in enumerate(histories):
        padded[i, : len(history)] = history
    return np.nanmean(padded, axis=0), np.nanstd(padded, axis=0)


def wilcoxon_z_from_differences(differences: np.ndarray) -> float:
    """
    Approximate the Wilcoxon signed-rank z value from paired differences.
    The sign follows the mean signed difference.
    """
    diffs = np.asarray(differences, dtype=np.float64)
    diffs = diffs[~np.isclose(diffs, 0.0)]
    if diffs.size == 0:
        return float("nan")

    abs_diffs = np.abs(diffs)
    ranks = rankdata(abs_diffs, method="average")
    w_plus = float(np.sum(ranks[diffs > 0]))
    n = diffs.size
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    if var_w <= 0.0:
        return float("nan")

    z = (w_plus - mean_w) / math.sqrt(var_w)
    return float(z)


def save_summary_outputs(
    results_df: pd.DataFrame,
    statistics_summary: Dict[str, float],
    icp_histories: Sequence[np.ndarray],
    mde_histories: Sequence[np.ndarray],
    directories: Dict[str, str],
    config: Config,
) -> None:
    """Save tables, statistics, and figures."""
    tables_dir = directories["tables"]
    figures_dir = directories["figures"]

    results_csv = os.path.join(tables_dir, "run_results.csv")
    results_df.to_csv(results_csv, index=False)

    with open(os.path.join(tables_dir, "run_results.tex"), "w", encoding="utf-8") as handle:
        handle.write(results_df.to_latex(index=False, float_format="%.4f"))

    summary_table = pd.DataFrame(
        {
            "Method": [
                "Initial (Centered)",
                "Coarse (RANSAC)",
                "TR-ICP (Baseline)",
                "MDE (Proposed)",
            ],
            "Mean (m)": [
                results_df["rmse_initial"].mean(),
                results_df["rmse_global"].mean(),
                results_df["rmse_tricp"].mean(),
                results_df["rmse_mde"].mean(),
            ],
            "Std. Dev. (m)": [
                results_df["rmse_initial"].std(ddof=1),
                results_df["rmse_global"].std(ddof=1),
                results_df["rmse_tricp"].std(ddof=1),
                results_df["rmse_mde"].std(ddof=1),
            ],
            "Min (m)": [
                results_df["rmse_initial"].min(),
                results_df["rmse_global"].min(),
                results_df["rmse_tricp"].min(),
                results_df["rmse_mde"].min(),
            ],
            "Max (m)": [
                results_df["rmse_initial"].max(),
                results_df["rmse_global"].max(),
                results_df["rmse_tricp"].max(),
                results_df["rmse_mde"].max(),
            ],
        }
    )
    summary_table.to_csv(os.path.join(tables_dir, "table2_summary.csv"), index=False)
    with open(os.path.join(tables_dir, "table2_summary.tex"), "w", encoding="utf-8") as handle:
        handle.write(summary_table.to_latex(index=False, float_format="%.4f"))

    with open(os.path.join(tables_dir, "statistics_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(statistics_summary, handle, indent=2)

    mean_icp, std_icp = pad_and_average(icp_histories)
    mean_mde, std_mde = pad_and_average(mde_histories)

    plt.figure(figsize=config.figure_size)
    plt.boxplot(
        [results_df["rmse_tricp"].values, results_df["rmse_mde"].values],
        labels=["TR-ICP", "MDE"],
        widths=0.5,
        showfliers=False,
    )
    x_icp = np.random.normal(loc=1.0, scale=0.03, size=len(results_df))
    x_mde = np.random.normal(loc=2.0, scale=0.03, size=len(results_df))
    plt.scatter(x_icp, results_df["rmse_tricp"].values, s=18, alpha=0.8)
    plt.scatter(x_mde, results_df["rmse_mde"].values, s=18, alpha=0.8)
    plt.ylabel("RMSE (m)")
    plt.title(f"Distribution of RMSE Values ({len(results_df)} Runs)")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "rmse_distribution_boxplot.png"), dpi=config.figure_dpi)
    plt.savefig(os.path.join(figures_dir, "rmse_distribution_boxplot.svg"))
    plt.close()

    plt.figure(figsize=config.figure_size)
    plt.plot(mean_icp, label="TR-ICP (Baseline)")
    plt.fill_between(
        np.arange(len(mean_icp)),
        mean_icp - std_icp,
        mean_icp + std_icp,
        alpha=0.25,
    )
    plt.plot(mean_mde, label="MDE (Proposed)", linestyle="--")
    plt.fill_between(
        np.arange(len(mean_mde)),
        mean_mde - std_mde,
        mean_mde + std_mde,
        alpha=0.25,
    )
    plt.yscale("log")
    plt.xlabel("Iteration / Cycle")
    plt.ylabel("Optimization Error")
    plt.title(f"Convergence Analysis (Average of {len(results_df)} Runs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "convergence_analysis.png"), dpi=config.figure_dpi)
    plt.savefig(os.path.join(figures_dir, "convergence_analysis.svg"))
    plt.close()


def save_transform_matrix(path: str, transform: np.ndarray) -> None:
    """Save a 4x4 transform matrix as a text file."""
    np.savetxt(path, transform, fmt="%.10f")


def run_single_experiment(
    config: Config,
    run_index: int,
    directories: Dict[str, str],
    seed: Optional[int],
) -> Dict[str, object]:
    """Run one complete registration experiment."""
    print(f"\n{'=' * 12} Run {run_index + 1}/{config.n_runs} {'=' * 12}")

    source_tls_original = load_point_cloud(config.source_tls_path)
    target_uav_original = load_point_cloud(config.target_uav_path)

    source_tls_centered, source_center = center_point_cloud(source_tls_original)
    target_uav_centered, target_center = center_point_cloud(target_uav_original)

    source_tls_coarse = prepare_downsampled_cloud(
        source_tls_centered,
        config.voxel_coarse,
        config.normal_radius_multiplier,
        config.normal_max_nn,
    )
    target_uav_coarse = prepare_downsampled_cloud(
        target_uav_centered,
        config.voxel_coarse,
        config.normal_radius_multiplier,
        config.normal_max_nn,
    )
    source_tls_fine = prepare_downsampled_cloud(
        source_tls_centered,
        config.voxel_fine,
        config.normal_radius_multiplier,
        config.normal_max_nn,
    )
    target_uav_fine = prepare_downsampled_cloud(
        target_uav_centered,
        config.voxel_fine,
        config.normal_radius_multiplier,
        config.normal_max_nn,
    )

    transform_identity = np.eye(4, dtype=np.float64)
    rmse_initial = rmse_bidirectional_trimmed(
        source_tls_fine,
        target_uav_fine,
        transform_identity,
        config.trim_for_reporting,
    )
    print(f"Initial centered RMSE: {rmse_initial:.6f} m")

    print("\nStep 1: Coarse alignment with FPFH + RANSAC")
    transform_global = global_registration_fpfh_ransac(
        source_tls_coarse,
        target_uav_coarse,
        config,
    )
    rmse_global = rmse_bidirectional_trimmed(
        source_tls_fine,
        target_uav_fine,
        transform_global,
        config.trim_for_reporting,
    )
    print(f"Coarse RMSE after RANSAC: {rmse_global:.6f} m")

    if rmse_global > rmse_initial:
        print(
            "RANSAC produced a worse result than the initial centroid-translated state. "
            "The coarse transform is reverted to the identity matrix."
        )
        transform_global = transform_identity
        rmse_global = rmse_initial

    print("\nStep 2A: Fine alignment with Trimmed ICP")
    transform_tricp, tricp_history = trimmed_icp(
        source_tls_fine,
        target_uav_fine,
        transform_global,
        config,
    )
    rmse_tricp = rmse_bidirectional_trimmed(
        source_tls_fine,
        target_uav_fine,
        transform_tricp,
        config.trim_for_reporting,
    )
    print(f"Final TR-ICP RMSE: {rmse_tricp:.6f} m")

    print("\nStep 2B: Fine alignment with MDE")
    start_mde = time.time()
    transform_delta_mde, mde_meta = mde_registration(
        source_tls_fine,
        target_uav_fine,
        transform_global,
        config,
        seed=seed,
    )
    elapsed_mde = time.time() - start_mde
    transform_mde = transform_global @ transform_delta_mde
    rmse_mde = rmse_bidirectional_trimmed(
        source_tls_fine,
        target_uav_fine,
        transform_mde,
        config.trim_for_reporting,
    )
    print(f"Final MDE RMSE: {rmse_mde:.6f} m")
    print(f"MDE runtime: {elapsed_mde:.2f} seconds")

    transform_global_original = convert_centered_transform_to_original(
        transform_global, source_center, target_center
    )
    transform_tricp_original = convert_centered_transform_to_original(
        transform_tricp, source_center, target_center
    )
    transform_mde_original = convert_centered_transform_to_original(
        transform_mde, source_center, target_center
    )

    transform_dir = directories["transforms"]
    save_transform_matrix(
        os.path.join(transform_dir, f"run_{run_index + 1:02d}_global_centered.txt"),
        transform_global,
    )
    save_transform_matrix(
        os.path.join(transform_dir, f"run_{run_index + 1:02d}_tricp_centered.txt"),
        transform_tricp,
    )
    save_transform_matrix(
        os.path.join(transform_dir, f"run_{run_index + 1:02d}_mde_centered.txt"),
        transform_mde,
    )
    save_transform_matrix(
        os.path.join(transform_dir, f"run_{run_index + 1:02d}_global_original.txt"),
        transform_global_original,
    )
    save_transform_matrix(
        os.path.join(transform_dir, f"run_{run_index + 1:02d}_tricp_original.txt"),
        transform_tricp_original,
    )
    save_transform_matrix(
        os.path.join(transform_dir, f"run_{run_index + 1:02d}_mde_original.txt"),
        transform_mde_original,
    )

    if run_index == 0 and config.save_snapshots_for_first_run:
        fig_dir = directories["figures"]
        save_overlay_snapshot(
            source_tls_centered,
            target_uav_centered,
            transform_identity,
            os.path.join(fig_dir, "initial_centered_state.png"),
            "Initial Position: Centroid-Translated Clouds",
            config.snapshot_sample_size,
            config.figure_dpi,
        )
        save_overlay_snapshot(
            source_tls_centered,
            target_uav_centered,
            transform_global,
            os.path.join(fig_dir, "coarse_alignment_ransac.png"),
            "Coarse Alignment After FPFH + RANSAC",
            config.snapshot_sample_size,
            config.figure_dpi,
        )
        save_overlay_snapshot(
            source_tls_centered,
            target_uav_centered,
            transform_tricp,
            os.path.join(fig_dir, "fine_alignment_tricp.png"),
            "Fine Alignment with TR-ICP",
            config.snapshot_sample_size,
            config.figure_dpi,
        )
        save_overlay_snapshot(
            source_tls_centered,
            target_uav_centered,
            transform_mde,
            os.path.join(fig_dir, "fine_alignment_mde.png"),
            "Fine Alignment with MDE",
            config.snapshot_sample_size,
            config.figure_dpi,
        )

    return {
        "run": run_index + 1,
        "rmse_initial": rmse_initial,
        "rmse_global": rmse_global,
        "rmse_tricp": rmse_tricp,
        "rmse_mde": rmse_mde,
        "mde_time_seconds": elapsed_mde,
        "tricp_history": tricp_history,
        "mde_history": mde_meta["history"],
        "global_transform_centered": transform_global,
        "tricp_transform_centered": transform_tricp,
        "mde_transform_centered": transform_mde,
        "global_transform_original": transform_global_original,
        "tricp_transform_original": transform_tricp_original,
        "mde_transform_original": transform_mde_original,
    }


def run_all_experiments(config: Config) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run all repeated experiments and save the outputs."""
    directories = ensure_directories(config)
    with open(os.path.join(directories["results"], "config.json"), "w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    results: List[Dict[str, object]] = []
    tricp_histories: List[np.ndarray] = []
    mde_histories: List[np.ndarray] = []

    for run_idx in range(config.n_runs):
        run_seed = None if config.random_seed is None else config.random_seed + run_idx
        single_result = run_single_experiment(config, run_idx, directories, run_seed)
        results.append(single_result)
        tricp_histories.append(single_result["tricp_history"])
        mde_histories.append(single_result["mde_history"])

    results_df = pd.DataFrame(
        {
            "run": [r["run"] for r in results],
            "rmse_initial": [r["rmse_initial"] for r in results],
            "rmse_global": [r["rmse_global"] for r in results],
            "rmse_tricp": [r["rmse_tricp"] for r in results],
            "rmse_mde": [r["rmse_mde"] for r in results],
            "mde_time_seconds": [r["mde_time_seconds"] for r in results],
        }
    )

    paired_differences = results_df["rmse_mde"].values - results_df["rmse_tricp"].values

    if len(paired_differences) >= 3:
        shapiro_stat, shapiro_p = stats.shapiro(paired_differences)
    else:
        shapiro_stat, shapiro_p = float("nan"), float("nan")

    if np.isnan(shapiro_p):
        test_name = "not_enough_samples"
        test_stat = float("nan")
        p_value = float("nan")
        wilcoxon_z = float("nan")
    elif shapiro_p > 0.05:
        test_name = "paired_t_test"
        test_stat, p_value = stats.ttest_rel(
            results_df["rmse_mde"].values,
            results_df["rmse_tricp"].values,
        )
        wilcoxon_z = float("nan")
    else:
        test_name = "wilcoxon_signed_rank"
        wilcoxon_result = stats.wilcoxon(
            results_df["rmse_mde"].values,
            results_df["rmse_tricp"].values,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            mode="auto",
        )
        test_stat = float(wilcoxon_result.statistic)
        p_value = float(wilcoxon_result.pvalue)
        wilcoxon_z = wilcoxon_z_from_differences(paired_differences)

    mean_diff = float(np.mean(paired_differences))
    std_diff = float(np.std(paired_differences, ddof=1)) if len(paired_differences) > 1 else float("nan")
    cohens_d = abs(mean_diff) / std_diff if std_diff not in (0.0, float("nan")) and np.isfinite(std_diff) else float("nan")

    statistics_summary = {
        "test_name": test_name,
        "test_statistic": float(test_stat) if np.isfinite(test_stat) else test_stat,
        "p_value": float(p_value) if np.isfinite(p_value) else p_value,
        "wilcoxon_z_approx": float(wilcoxon_z) if np.isfinite(wilcoxon_z) else wilcoxon_z,
        "shapiro_statistic": float(shapiro_stat) if np.isfinite(shapiro_stat) else shapiro_stat,
        "shapiro_p_value": float(shapiro_p) if np.isfinite(shapiro_p) else shapiro_p,
        "mean_difference_mde_minus_tricp": mean_diff,
        "cohens_d_abs": float(cohens_d) if np.isfinite(cohens_d) else cohens_d,
    }

    save_summary_outputs(
        results_df,
        statistics_summary,
        tricp_histories,
        mde_histories,
        directories,
        config,
    )

    print("\nSummary statistics")
    print(results_df.describe())
    print("\nPaired comparison summary")
    print(json.dumps(statistics_summary, indent=2))

    return results_df, statistics_summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Public registration pipeline for the Hasakoy/Sasima manuscript"
    )
    parser.add_argument("--source_tls", default="source_tls.las", help="Path to the TLS source point cloud")
    parser.add_argument("--target_uav", default="target_uav.las", help="Path to the UAV target/reference point cloud")
    parser.add_argument("--results_dir", default="results_registration_pipeline", help="Output directory")
    parser.add_argument("--runs", type=int, default=30, help="Number of repeated experiments")
    parser.add_argument("--seed", type=int, default=None, help="Optional base random seed")
    parser.add_argument("--mde_jobs", type=int, default=None, help="Number of parallel workers for MDE fitness")
    parser.add_argument("--disable_snapshots", action="store_true", help="Disable snapshot generation for the first run")
    parser.add_argument("--snapshot_sample_size", type=int, default=90000, help="Point sample size for snapshots")
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> Config:
    """Create a Config instance from CLI arguments."""
    return Config(
        source_tls_path=args.source_tls,
        target_uav_path=args.target_uav,
        results_dir=args.results_dir,
        n_runs=args.runs,
        random_seed=args.seed,
        mde_jobs=args.mde_jobs,
        save_snapshots_for_first_run=not args.disable_snapshots,
        snapshot_sample_size=args.snapshot_sample_size,
    )


def main() -> None:
    """Main entry point."""
    args = parse_args()
    config = build_config_from_args(args)
    set_global_seed(config.random_seed)
    run_all_experiments(config)


if __name__ == "__main__":
    main()
