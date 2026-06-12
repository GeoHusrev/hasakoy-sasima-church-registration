# TLS-UAV Point Cloud Registration for the Hasakoy (Sasima) Church

Public source code and point cloud data for the multi-sensor registration
workflow of the Hasakoy (Sasima) Church, accompanying the manuscript
*Robust Multi-Sensor Point Cloud Registration for Cultural Heritage
Documentation: A Multi-Population Differential Evolution Approach*.

## Contents

- `MDE_PointCloud_Registration.py`: coarse-to-fine TLS-UAV registration pipeline
  using FPFH + RANSAC for the coarse stage and the Multi-population Differential
  Evolution (MDE) algorithm of Karkinli (2023) for the fine stage.
- `comparison_pso_mde.py`: comparison of Particle Swarm Optimization (PSO),
  Aquila Optimizer (AO), and MDE from the same accepted coarse transformation,
  together with the FPFH neighborhood-radius sensitivity test.
- `PointClouds.zip`: TLS and UAV point clouds of the Hasakoy (Sasima) Church.
- `requirements.txt`: Python package dependencies.
- `LICENSE`: MIT License.
- `CITATION.cff`: machine-readable citation metadata.

## Environment

Tested on Windows 11 with Python 3.12, Open3D 0.19, SciPy 1.17, laspy 2.7,
and MEALPY 3.0.3 for the AO comparator. Install the dependencies with:

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

## Running the Metaheuristic Comparison

PSO, AO, and MDE share the same 6-DoF search vector, adaptive bounds, trimmed
nearest-neighbor objective, population parameter of 10, maximum budget of 250
iterations or cycles, and early stopping after 30 successive iterations or
cycles without improvement. AO is provided as the contemporary MEALPY-based
comparator alongside PSO and MDE.

```bash
python comparison_pso_mde.py
```

To run only the AO comparator from the accepted fixed coarse transformation:

```bash
COMPARISON_METHODS=AO \
COMPARISON_FIXED_COARSE_PATH=transforms/fixed_coarse_centered.txt \
python comparison_pso_mde.py
```

The script reads optional environment variables for reproducibility:

| Variable | Default | Description |
|---|---|---|
| `COMPARISON_RUNS` | 30 | Number of stochastic runs |
| `COMPARISON_SEED` | 100 | Base random seed |
| `COMPARISON_COARSE_SEED` | 1 | Seed for the fixed coarse alignment |
| `COMPARISON_POPULATION_SIZE` | 10 | Population parameter shared by PSO, AO, and MDE |
| `COMPARISON_MAX_CYCLES` | 250 | Maximum iterations or cycles |
| `COMPARISON_PATIENCE` | 30 | Early-stopping patience |
| `COMPARISON_JOBS` | 8 | Parallel workers for fitness evaluation |
| `COMPARISON_OUTPUT_DIR` | `../comparison_results` | Output directory |
| `COMPARISON_METHODS` | `PSO,AO,MDE` | Comma-separated method list: `PSO`, `AO`, `MDE` |
| `COMPARISON_FIXED_COARSE_PATH` | unset | Optional fixed centered coarse-transform matrix |
| `COMPARISON_RUN_FPFH` | 1 | Set to `0` to skip FPFH-radius sensitivity |
| `MEALPY_SOLVE_MODE` | `thread` | MEALPY solve mode for AO when parallel workers are available |

The comparison outputs are written to the chosen output directory and include
runtime statistics, the 6-DoF correction decomposition, the FPFH radius
sensitivity table, and a per-method summary CSV.

## How to Cite

If you use this code or the accompanying point clouds in academic work, please
cite the associated manuscript:

> Karkınlı, A.E.; Janowski, A.; Kaderli, L.; Hüsrevoğlu, B.G.; Hüsrevoğlu, M.
> *Robust Multi-Sensor Point Cloud Registration for Cultural Heritage
> Documentation: A Multi-Population-Based Differential Evolution Approach.*
> Manuscript under peer review, 2026. Full bibliographic details will be added
> here upon publication.

A machine-readable citation entry is also provided in `CITATION.cff`.

## Algorithm References

- PSO: Kennedy, J.; Eberhart, R. Particle swarm optimization. *Proc. ICNN'95*,
  1995, 1942-1948. https://doi.org/10.1109/ICNN.1995.488968
- AO: Abualigah, L.; Yousri, D.; Abd Elaziz, M.; Ewees, A.A.; Al-qaness,
  M.A.A.; Gandomi, A.H. Aquila Optimizer: A novel meta-heuristic optimization
  algorithm. *Computers & Industrial Engineering* 2021, 157, 107250.
  https://doi.org/10.1016/j.cie.2021.107250
- MDE: Karkinli, A.E. Detection of object boundary from point cloud by using
  multi-population based differential evolution algorithm. *Neural Computing
  and Applications* 2023, 35, 5193-5206.
  https://doi.org/10.1007/s00521-022-07969-w

## License

The source code in this repository is released under the MIT License (see
`LICENSE`). The accompanying point cloud data in `PointClouds.zip` are made
available for academic and research use; please cite the associated manuscript
when using the data.

## Code Author

**Ahmet Emin Karkınlı**  
ORCID: 0000-0001-7216-6251  
Department of Geomatics Engineering, Faculty of Engineering, Niğde Ömer Halisdemir University, Niğde 51240, Türkiye  
Email: akarkinli@ohu.edu.tr

## Associated Manuscript Authors

- **Ahmet Emin Karkınlı**  
  ORCID: 0000-0001-7216-6251  
  Department of Geomatics Engineering, Faculty of Engineering, Niğde Ömer Halisdemir University, Niğde 51240, Türkiye  
  Email: akarkinli@ohu.edu.tr

- **Artur Janowski** (corresponding author)  
  ORCID: 0000-0002-5535-408X  
  Institute of Geodesy and Construction, University of Warmia and Mazury in Olsztyn, 10-720 Olsztyn, Poland  
  Email: artur.janowski@uwm.edu.pl

- **Leyla Kaderli**  
  ORCID: 0000-0002-3497-6664  
  Department of Architecture, Faculty of Architecture, Erciyes University, Kayseri 38280, Türkiye  
  Email: leylakaderli@erciyes.edu.tr

- **Betül Gül Hüsrevoğlu**  
  ORCID: 0009-0008-0252-9406  
  Department of Architecture, Graduate School of Natural and Applied Sciences, Erciyes University, Kayseri 38280, Türkiye  
  Email: 4011030096@erciyes.edu.tr

- **Mustafa Hüsrevoğlu**  
  ORCID: 0000-0003-1324-9617  
  Department of Geomatics Engineering, Faculty of Engineering, Niğde Ömer Halisdemir University, Niğde 51240, Türkiye  
  Email: mhusrevoglu@ohu.edu.tr
