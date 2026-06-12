"""
PSO, AO, and MDE comparison script for TLS-UAV point cloud registration.

This script reuses the public registration pipeline implemented in
MDE_PointCloud_Registration.py and evaluates Particle Swarm Optimization (PSO),
the Aquila Optimizer (AO), and the Multi-population Differential Evolution (MDE)
algorithm of Karkinli (2023) from the same accepted coarse transformation under
common search settings. The script also runs the Fast Point Feature Histogram
(FPFH) neighborhood-radius sensitivity analysis.

The metaheuristic methods share the same 6-DoF search vector, adaptive
translation and rotation bounds, trimmed nearest-neighbor objective, population
parameter, and maximum iteration or cycle budget so that the fine-tuning
evaluation is fair.

Code author
-----------
Dr. Ahmet Emin Karkınlı
Department of Geomatics Engineering, Faculty of Engineering,
Nigde Omer Halisdemir University, Nigde 51240, Turkiye
E-mail: akarkinli@ohu.edu.tr

Related manuscript authors
--------------------------
Ahmet Emin Karkınlı; Artur Janowski; Leyla Kaderli;
Betul Gul Husrevoglu; Mustafa Husrevoglu

PSO reference
-------------
Kennedy, J.; Eberhart, R. Particle swarm optimization. In Proceedings of
ICNN'95 - International Conference on Neural Networks, Perth, WA, Australia,
27 November - 1 December 1995; pp. 1942-1948.
https://doi.org/10.1109/ICNN.1995.488968

AO reference
------------
Abualigah, L.; Yousri, D.; Abd Elaziz, M.; Ewees, A.A.; Al-qaness, M.A.A.;
Gandomi, A.H. Aquila Optimizer: A novel meta-heuristic optimization algorithm.
Computers & Industrial Engineering 2021, 157, 107250.
https://doi.org/10.1016/j.cie.2021.107250

MDE reference
-------------
Karkinli, A.E. Detection of object boundary from point cloud by using
multi-population based differential evolution algorithm. Neural Computing and
Applications 2023, 35, 5193-5206.
https://doi.org/10.1007/s00521-022-07969-w

Disclaimer
----------
This code is provided for research and academic use only, without any express
or implied warranty. While reasonable efforts were made to ensure consistency
with the methodology described in the associated manuscript, the authors make
no guarantee that the code will run without modification in all computing
environments or reproduce identical numerical results under all conditions.
"""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import psutil
from scipy.spatial.transform import Rotation as R


ROOT = Path(__file__).resolve().parent
PIPELINE_PATH = ROOT / "MDE_PointCloud_Registration.py"
DEFAULT_METHOD_ORDER = ["PSO", "AO", "MDE"]
MEALPY_METHODS = {"AO"}

spec = importlib.util.spec_from_file_location("registration_pipeline", PIPELINE_PATH)
reg = importlib.util.module_from_spec(spec)
sys.modules["registration_pipeline"] = reg
assert spec.loader is not None
spec.loader.exec_module(reg)


def resolve_data_path(filename: str) -> Path:
    """Find a point-cloud file in the release layout or the manuscript layout."""
    candidates = [
        ROOT / "PointClouds" / filename,
        ROOT / filename,
        ROOT.parent / "PointClouds" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_method_order() -> List[str]:
    """Read the requested methods from COMPARISON_METHODS."""
    raw = os.environ.get("COMPARISON_METHODS")
    if not raw:
        return DEFAULT_METHOD_ORDER.copy()
    methods = [item.strip().upper() for item in raw.split(",") if item.strip()]
    valid = {"PSO", "AO", "MDE"}
    invalid = [method for method in methods if method not in valid]
    if invalid:
        raise ValueError(f"Unsupported COMPARISON_METHODS entries: {invalid}")
    return methods


def load_mealpy_optimizer(method: str):
    """Load the requested MEALPY optimizer class."""
    extra_path = os.environ.get("MEALPY_EXTRA_PATH")
    if extra_path and extra_path not in sys.path:
        sys.path.insert(0, extra_path)
    try:
        from mealpy import AO
    except ImportError as exc:
        raise ImportError(
            "MEALPY is required for the AO comparator. Install mealpy or set "
            "MEALPY_EXTRA_PATH to a directory containing the package."
        ) from exc
    if method == "AO":
        return AO.OriginalAO
    raise ValueError(f"Unsupported MEALPY method: {method}")


def compose_delta_transform(solution: np.ndarray) -> np.ndarray:
    """Convert [tx, ty, tz, rx, ry, rz] into a homogeneous transform."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R.from_euler("xyz", solution[3:], degrees=True).as_matrix()
    transform[:3, 3] = solution[:3]
    return transform


def decompose_fine_correction(coarse_transform: np.ndarray, final_transform: np.ndarray) -> Dict[str, float]:
    """Return translation and Euler-angle correction from coarse to final alignment."""
    delta = np.linalg.inv(coarse_transform) @ final_transform
    rx, ry, rz = R.from_matrix(delta[:3, :3]).as_euler("xyz", degrees=True)
    tx, ty, tz = delta[:3, 3]
    return {
        "tx_m": float(tx),
        "ty_m": float(ty),
        "tz_m": float(tz),
        "rx_deg": float(rx),
        "ry_deg": float(ry),
        "rz_deg": float(rz),
        "translation_norm_m": float(np.linalg.norm(delta[:3, 3])),
        "rotation_angle_deg": float(R.from_matrix(delta[:3, :3]).magnitude() * 180.0 / np.pi),
    }


def pso_registration(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    coarse_transform: np.ndarray,
    config: reg.Config,
    seed: Optional[int] = None,
    swarm_size: int = 10,
    max_iterations: int = 250,
    patience: int = 30,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Particle Swarm Optimization using the same trimmed fitness as MDE."""
    if seed is not None:
        np.random.seed(seed)

    source_points = np.asarray(source_pcd.points)
    target_points = np.asarray(target_pcd.points)
    low, up, initial_rmse, translation_range, rotation_range = reg.build_adaptive_bounds(
        source_pcd, target_pcd, coarse_transform, config
    )
    print(
        f"[PSO] One-way RMSE for adaptive bounds: {initial_rmse:.6f} m; "
        f"translation +/-{translation_range:.3f} m, rotation +/-{rotation_range:.3f} degrees"
    )

    dim = config.mde_dimension
    inertia = 0.7298
    cognitive = 1.49618
    social = 1.49618

    positions = np.random.rand(swarm_size, dim) * (up - low) + low
    velocity_scale = 0.10 * (up - low)
    velocities = (np.random.rand(swarm_size, dim) - 0.5) * velocity_scale

    fitness = reg.fitness_eval_mde_parallel(
        positions,
        source_points,
        target_points,
        coarse_transform,
        config.mde_sample_size,
        config.mde_trim_ratio,
        config.mde_jobs,
    )
    personal_best = positions.copy()
    personal_best_fitness = fitness.copy()
    best_idx = int(np.argmin(fitness))
    global_best = positions[best_idx].copy()
    global_best_fitness = float(fitness[best_idx])

    history: List[float] = [global_best_fitness]
    no_improvement = 0
    print(f"[PSO] Initial best fitness: {global_best_fitness:.10e}")

    for iteration in range(1, max_iterations + 1):
        r1 = np.random.rand(swarm_size, dim)
        r2 = np.random.rand(swarm_size, dim)
        velocities = (
            inertia * velocities
            + cognitive * r1 * (personal_best - positions)
            + social * r2 * (global_best - positions)
        )
        positions = reg.border_control(positions + velocities, low, up)

        fitness = reg.fitness_eval_mde_parallel(
            positions,
            source_points,
            target_points,
            coarse_transform,
            config.mde_sample_size,
            config.mde_trim_ratio,
            config.mde_jobs,
        )

        improved = fitness < personal_best_fitness
        personal_best[improved] = positions[improved]
        personal_best_fitness[improved] = fitness[improved]

        best_idx = int(np.argmin(personal_best_fitness))
        current_best = float(personal_best_fitness[best_idx])
        if current_best < global_best_fitness:
            global_best_fitness = current_best
            global_best = personal_best[best_idx].copy()
            no_improvement = 0
            print(f"[PSO] Improved at iteration {iteration:03d}: {global_best_fitness:.10e}")
        else:
            no_improvement += 1

        history.append(global_best_fitness)
        if no_improvement >= patience:
            print(f"[PSO] Early stopping at iteration {iteration} after {patience} iterations without improvement.")
            break

    return compose_delta_transform(global_best), {
        "history": np.asarray(history, dtype=np.float64),
        "best_solution": np.asarray(global_best, dtype=np.float64),
        "best_fitness": np.array([global_best_fitness], dtype=np.float64),
    }


def mealpy_registration(
    method: str,
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    coarse_transform: np.ndarray,
    config: reg.Config,
    seed: Optional[int] = None,
    population_size: int = 10,
    max_epochs: int = 250,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Run a MEALPY optimizer on the same bounded 6-DoF registration objective."""
    optimizer_cls = load_mealpy_optimizer(method)
    from mealpy import FloatVar
    from scipy.spatial import cKDTree

    if seed is not None:
        np.random.seed(seed)

    source_points = np.asarray(source_pcd.points)
    target_points = np.asarray(target_pcd.points)
    low, up, initial_rmse, translation_range, rotation_range = reg.build_adaptive_bounds(
        source_pcd, target_pcd, coarse_transform, config
    )
    print(
        f"[{method}] One-way RMSE for adaptive bounds: {initial_rmse:.6f} m; "
        f"translation +/-{translation_range:.3f} m, rotation +/-{rotation_range:.3f} degrees"
    )

    if len(target_points) == 0:
        raise ValueError("The target point cloud is empty.")
    target_tree = cKDTree(target_points)

    def objective(solution: np.ndarray) -> float:
        if config.mde_sample_size > 0 and config.mde_sample_size < len(source_points):
            sample_indices = np.random.choice(len(source_points), config.mde_sample_size, replace=False)
            source_sample = source_points[sample_indices, :]
        else:
            source_sample = source_points
        return float(
            reg._fitness_single_candidate(
                np.asarray(solution, dtype=np.float64),
                source_sample,
                target_tree,
                coarse_transform,
                config.mde_trim_ratio,
            )
        )

    problem = {
        "bounds": FloatVar(lb=tuple(low), ub=tuple(up), name="delta"),
        "minmax": "min",
        "obj_func": objective,
        "log_to": "none",
    }
    model = optimizer_cls(epoch=max_epochs, pop_size=population_size)
    termination = {"max_epoch": max_epochs, "max_early_stop": config.mde_early_stop_patience}
    solve_mode = os.environ.get("MEALPY_SOLVE_MODE", "thread" if (config.mde_jobs or 0) > 1 else "single")
    n_workers = config.mde_jobs if solve_mode in {"thread", "process"} else None
    best = model.solve(problem, mode=solve_mode, n_workers=n_workers, seed=seed, termination=termination)
    history = np.asarray(model.history.list_global_best_fit, dtype=np.float64)
    if history.size == 0:
        history = np.asarray([best.target.fitness], dtype=np.float64)

    return compose_delta_transform(np.asarray(best.solution, dtype=np.float64)), {
        "history": history,
        "best_solution": np.asarray(best.solution, dtype=np.float64),
        "best_fitness": np.array([float(best.target.fitness)], dtype=np.float64),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    """Write a list of dictionaries to CSV."""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, object]], method: str) -> Dict[str, object]:
    """Create compact descriptive statistics for one method."""
    rmse = np.asarray([float(row["rmse_m"]) for row in rows if row["method"] == method], dtype=np.float64)
    runtime = np.asarray([float(row["runtime_s"]) for row in rows if row["method"] == method], dtype=np.float64)
    iterations = np.asarray([float(row["iterations_or_cycles"]) for row in rows if row["method"] == method], dtype=np.float64)
    if rmse.size == 0:
        return {}
    return {
        "method": method,
        "rmse_mean_m": float(np.mean(rmse)),
        "rmse_median_m": float(np.median(rmse)),
        "rmse_std_m": float(np.std(rmse, ddof=1)) if rmse.size > 1 else 0.0,
        "rmse_min_m": float(np.min(rmse)),
        "rmse_max_m": float(np.max(rmse)),
        "runtime_mean_s": float(np.mean(runtime)),
        "runtime_std_s": float(np.std(runtime, ddof=1)) if runtime.size > 1 else 0.0,
        "iterations_or_cycles_mean": float(np.mean(iterations)),
    }


def main() -> None:
    """Run the PSO vs MDE comparison and the FPFH radius sensitivity test."""
    output_dir = Path(os.environ.get("COMPARISON_OUTPUT_DIR", str(ROOT.parent / "comparison_results")))
    tables_dir = output_dir / "tables"
    transforms_dir = output_dir / "transforms"
    tables_dir.mkdir(parents=True, exist_ok=True)
    transforms_dir.mkdir(parents=True, exist_ok=True)

    n_runs = int(os.environ.get("COMPARISON_RUNS", "30"))
    base_seed = int(os.environ.get("COMPARISON_SEED", "100"))
    coarse_seed = int(os.environ.get("COMPARISON_COARSE_SEED", "1"))
    run_offset = int(os.environ.get("COMPARISON_RUN_OFFSET", "0"))
    population_size = int(os.environ.get("COMPARISON_POPULATION_SIZE", "10"))
    max_cycles = int(os.environ.get("COMPARISON_MAX_CYCLES", "250"))
    early_stop_patience = int(os.environ.get("COMPARISON_PATIENCE", "30"))
    method_order = parse_method_order()
    run_fpfh_sensitivity = os.environ.get("COMPARISON_RUN_FPFH", "1").strip().lower() not in {"0", "false", "no"}

    config = reg.Config(
        source_tls_path=str(resolve_data_path("TLS_PointCloud.las")),
        target_uav_path=str(resolve_data_path("UAV_PointCloud.las")),
        results_dir=str(output_dir),
        n_runs=n_runs,
        mde_population_size=population_size,
        mde_max_cycle=max_cycles,
        mde_early_stop_patience=early_stop_patience,
        mde_jobs=int(os.environ.get("COMPARISON_JOBS", "8")),
        random_seed=base_seed,
        save_snapshots_for_first_run=False,
    )

    environment = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / 1024**3, 2),
        "config": asdict(config),
        "coarse_seed": coarse_seed,
        "run_offset": run_offset,
        "population_parameter": population_size,
        "max_cycles": max_cycles,
        "patience": early_stop_patience,
        "method_order": method_order,
    }
    (tables_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    print("Loading point clouds...")
    source_original = reg.load_point_cloud(config.source_tls_path)
    target_original = reg.load_point_cloud(config.target_uav_path)
    source_centered, source_center = reg.center_point_cloud(source_original)
    target_centered, target_center = reg.center_point_cloud(target_original)

    print("Preparing downsampled clouds...")
    source_coarse = reg.prepare_downsampled_cloud(
        source_centered, config.voxel_coarse, config.normal_radius_multiplier, config.normal_max_nn
    )
    target_coarse = reg.prepare_downsampled_cloud(
        target_centered, config.voxel_coarse, config.normal_radius_multiplier, config.normal_max_nn
    )
    source_fine = reg.prepare_downsampled_cloud(
        source_centered, config.voxel_fine, config.normal_radius_multiplier, config.normal_max_nn
    )
    target_fine = reg.prepare_downsampled_cloud(
        target_centered, config.voxel_fine, config.normal_radius_multiplier, config.normal_max_nn
    )

    counts = {
        "source_fine_points": len(source_fine.points),
        "target_fine_points": len(target_fine.points),
        "source_coarse_points": len(source_coarse.points),
        "target_coarse_points": len(target_coarse.points),
    }
    (tables_dir / "point_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))

    identity = np.eye(4, dtype=np.float64)
    initial_rmse = reg.rmse_bidirectional_trimmed(source_fine, target_fine, identity, config.trim_for_reporting)

    fixed_coarse_path = os.environ.get("COMPARISON_FIXED_COARSE_PATH")
    if fixed_coarse_path:
        print(f"Loading fixed coarse alignment from {fixed_coarse_path}...")
        coarse_transform = np.loadtxt(fixed_coarse_path)
        coarse_rmse = reg.rmse_bidirectional_trimmed(source_fine, target_fine, coarse_transform, config.trim_for_reporting)
    else:
        print("Computing fixed coarse alignment...")
        np.random.seed(coarse_seed)
        o3d.utility.random.seed(coarse_seed)
        coarse_transform = reg.global_registration_fpfh_ransac(source_coarse, target_coarse, config)
        coarse_rmse = reg.rmse_bidirectional_trimmed(source_fine, target_fine, coarse_transform, config.trim_for_reporting)
        if coarse_rmse > initial_rmse:
            coarse_transform = identity.copy()
            coarse_rmse = initial_rmse

    np.savetxt(transforms_dir / "fixed_coarse_centered.txt", coarse_transform, fmt="%.10f")
    np.savetxt(
        transforms_dir / "fixed_coarse_original.txt",
        reg.convert_centered_transform_to_original(coarse_transform, source_center, target_center),
        fmt="%.10f",
    )

    rows: List[Dict[str, object]] = []

    for run_idx in range(n_runs):
        seed = base_seed + run_offset + run_idx
        run_number = run_offset + run_idx + 1
        print(f"\n========== Comparison run {run_number} seed={seed} ==========")

        stochastic_methods = []
        for method in method_order:
            if method == "PSO":
                stochastic_methods.append(
                    (
                        "PSO",
                        lambda seed=seed: pso_registration(
                            source_fine,
                            target_fine,
                            coarse_transform,
                            config,
                            seed=seed,
                            swarm_size=population_size,
                            max_iterations=max_cycles,
                            patience=early_stop_patience,
                        ),
                    )
                )
            elif method == "MDE":
                stochastic_methods.append(
                    (
                        "MDE",
                        lambda seed=seed: reg.mde_registration(source_fine, target_fine, coarse_transform, config, seed=seed),
                    )
                )
            elif method in MEALPY_METHODS:
                stochastic_methods.append(
                    (
                        method,
                        lambda method=method, seed=seed: mealpy_registration(
                            method,
                            source_fine,
                            target_fine,
                            coarse_transform,
                            config,
                            seed=seed,
                            population_size=population_size,
                            max_epochs=max_cycles,
                        ),
                    )
                )

        for method, runner in stochastic_methods:
            print(f"Running {method}...")
            start = time.time()
            delta_transform, metadata = runner()
            elapsed = time.time() - start
            final_transform = coarse_transform @ delta_transform
            rmse = reg.rmse_bidirectional_trimmed(source_fine, target_fine, final_transform, config.trim_for_reporting)
            np.savetxt(
                transforms_dir / f"{method.lower()}_run_{run_number:02d}_centered.txt",
                final_transform,
                fmt="%.10f",
            )
            row = {
                "run": run_number,
                "method": method,
                "rmse_m": float(rmse),
                "runtime_s": float(elapsed),
                "iterations_or_cycles": int(len(metadata["history"])),
                "initial_rmse_m": float(initial_rmse),
                "coarse_rmse_m": float(coarse_rmse),
            }
            row.update(decompose_fine_correction(coarse_transform, final_transform))
            rows.append(row)
            write_csv(tables_dir / "runtime_6dof_results.csv", rows)

    write_csv(tables_dir / "runtime_6dof_results.csv", rows)
    summaries = [summarize(rows, method) for method in method_order]
    summaries = [row for row in summaries if row]
    write_csv(tables_dir / "method_summary.csv", summaries)

    if not run_fpfh_sensitivity:
        print("Skipping FPFH radius sensitivity because COMPARISON_RUN_FPFH=0.")
        print("Comparison experiments completed.")
        return

    print("Running FPFH radius sensitivity...")
    sensitivity_rows: List[Dict[str, object]] = []
    for multiplier in [3.0, 5.0, 7.0]:
        sensitivity_config = copy.copy(config)
        sensitivity_config.fpfh_radius_multiplier = multiplier
        np.random.seed(coarse_seed)
        o3d.utility.random.seed(coarse_seed)
        start = time.time()
        transform = reg.global_registration_fpfh_ransac(source_coarse, target_coarse, sensitivity_config)
        rmse = reg.rmse_bidirectional_trimmed(source_fine, target_fine, transform, config.trim_for_reporting)
        elapsed = time.time() - start
        sensitivity_rows.append(
            {
                "fpfh_radius_multiplier": multiplier,
                "fpfh_radius_m": multiplier * sensitivity_config.voxel_coarse,
                "coarse_rmse_m": float(rmse),
                "runtime_s": float(elapsed),
            }
        )
        write_csv(tables_dir / "fpfh_radius_sensitivity.csv", sensitivity_rows)

    write_csv(tables_dir / "fpfh_radius_sensitivity.csv", sensitivity_rows)
    print("Comparison experiments completed.")


if __name__ == "__main__":
    main()
