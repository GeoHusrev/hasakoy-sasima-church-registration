# TLS-UAV Point Cloud Registration for the Hasakoy (Sasima) Church

Public source code and point cloud data for the multi-sensor registration
workflow of the Hasakoy (Sasima) Church, accompanying the manuscript
*Robust Multi-Sensor Point Cloud Registration for Cultural Heritage
Documentation: A Multi-Population Differential Evolution Approach*.

## Contents

- `MDE_PointCloud_Registration.py`: coarse-to-fine TLS-UAV registration pipeline
  using FPFH + RANSAC for the coarse stage and the Multi-population Differential
  Evolution (MDE) algorithm of Karkinli (2023) for the fine stage.
- `comparison_pso_mde.py`: comparison of Particle Swarm Optimization (PSO) and
  MDE from the same accepted coarse transformation, together with the FPFH
  neighborhood-radius sensitivity test.
- `PointClouds.zip`: TLS and UAV point clouds of the Hasakoy (Sasima) Church.
- `requirements.txt`: Python package dependencies.
- `LICENSE`: MIT License.
- `CITATION.cff`: machine-readable citation metadata.

## Environment

Tested on Windows 11 with Python 3.12, Open3D 0.19, NumPy 2.3, SciPy 1.17, and
laspy 2.7. Install the dependencies with:

```bash
pip install -r requirements.txt
```

Extract `PointClouds.zip` so that the `PointClouds/` folder contains
`TLS_PointCloud.las` and `UAV_PointCloud.las`.

## Running the Main Pipeline

```bash
python MDE_PointCloud_Registration.py \
    --source_tls PointClouds/TLS_PointCloud.las \
    --target_uav PointClouds/UAV_PointCloud.las \
    --results_dir results_registration_pipeline \
    --runs 30
```

Outputs are written to `results_registration_pipeline/` and include the MDE
RMSE summary table, the convergence plot, and the per-run transformation
matrices.

## Running the PSO and MDE Comparison

PSO and MDE share the same 6-DoF search vector, adaptive bounds, trimmed
nearest-neighbor objective, population parameter of 10, maximum budget of 250
iterations or cycles, and early stopping after 30 successive iterations or
cycles without improvement.

```bash
python comparison_pso_mde.py
```

The script reads optional environment variables for reproducibility:

| Variable | Default | Description |
|---|---|---|
| `COMPARISON_RUNS` | 30 | Number of stochastic runs |
| `COMPARISON_SEED` | 100 | Base random seed |
| `COMPARISON_COARSE_SEED` | 1 | Seed for the fixed coarse alignment |
| `COMPARISON_POPULATION_SIZE` | 10 | Population parameter shared by PSO and MDE |
| `COMPARISON_MAX_CYCLES` | 250 | Maximum iterations or cycles |
| `COMPARISON_PATIENCE` | 30 | Early-stopping patience |
| `COMPARISON_JOBS` | 8 | Parallel workers for fitness evaluation |
| `COMPARISON_OUTPUT_DIR` | `../comparison_results` | Output directory |

The comparison outputs are written to the chosen output directory and include
runtime statistics, the 6-DoF correction decomposition, the FPFH radius
sensitivity table, and a per-method summary CSV.

## How to Cite

If you use this code or the accompanying point clouds in academic work, please
cite the associated manuscript:

> Karkınlı, A.E.; Janowski, A.; Kaderli, L.; Hüsrevoğlu, B.G.; Hüsrevoğlu, M.
> *Robust Multi-Sensor Point Cloud Registration for Cultural Heritage
> Documentation: A Multi-Population Differential Evolution Approach.*
> Manuscript under peer review, 2026. Full bibliographic details will be added
> here upon publication.

A machine-readable citation entry is also provided in `CITATION.cff`.

## Algorithm References

- PSO: Kennedy, J.; Eberhart, R. Particle swarm optimization. *Proc. ICNN'95*,
  1995, 1942-1948. https://doi.org/10.1109/ICNN.1995.488968
- MDE: Karkinli, A.E. Detection of object boundary from point cloud by using
  multi-population based differential evolution algorithm. *Neural Computing
  and Applications* 2023, 35, 5193-5206.
  https://doi.org/10.1007/s00521-022-07969-w

## License

The source code in this repository is released under the MIT License (see
`LICENSE`). The accompanying point cloud data in `PointClouds.zip` are made
available for academic and research use; please cite the associated manuscript
when using the data.

## Contact

Code author and maintainer: **Dr. Ahmet Emin Karkınlı**
Department of Geomatics Engineering, Faculty of Engineering,
Nigde Omer Halisdemir University, Nigde 51240, Türkiye
Email: akarkinli@ohu.edu.tr
