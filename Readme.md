# Hasakoy/Sasima Church Registration Pipeline

Python implementation of the TLS-UAV point cloud registration workflow prepared for the manuscript:

**Robust Multi-Sensor Point Cloud Registration for Cultural Heritage Documentation: A Multi-Population Differential Evolution Approach**

## Code Author
**Ahmet Emin Karkınlı**  
ORCID: 0000-0001-7216-6251  
Department of Geomatics Engineering, Faculty of Engineering, Niğde Ömer Halisdemir University, Niğde 51240, Türkiye  
Email: akarkinli@ohu.edu.tr

## Associated Manuscript Authors
- **Artur Janowski**  
  ORCID: 0000-0002-5535-408X  
  Institute of Geodesy and Construction, University of Warmia and Mazury in Olsztyn, 10-720 Olsztyn, Poland  
  Email: artur.janowski@uwm.edu.pl

- **Ahmet Emin Karkınlı**  
  ORCID: 0000-0001-7216-6251  
  Department of Geomatics Engineering, Faculty of Engineering, Niğde Ömer Halisdemir University, Niğde 51240, Türkiye  
  Email: akarkinli@ohu.edu.tr

- **Leyla Kaderli**  
  ORCID: 0000-0002-3497-6664  
  Department of Architecture, Faculty of Architecture, Erciyes University, Kayseri 38280, Türkiye  
  Email: leylakaderli@erciyes.edu.tr

- **Betül Gül Hüsrevoğlu**  
  ORCID: 0009-0008-0252-9406  
  Department of Architecture, Graduate School of Natural and Applied Sciences, Erciyes University, Kayseri 38280, Türkiye  
  Email: betulgulnny@gmail.com

- **Mustafa Hüsrevoğlu**  
  ORCID: 0000-0003-1324-9617  
  Department of Geomatics Engineering, Faculty of Engineering, Niğde Ömer Halisdemir University, Niğde 51240, Türkiye  
  Email: mhusrevoglu@ohu.edu.tr

## Description
This repository contains the Python code used to implement the point cloud registration workflow described in the associated manuscript. The workflow follows a coarse-to-fine strategy based on FPFH feature extraction, RANSAC-based coarse alignment, TR-ICP baseline refinement, and MDE-based fine registration.

## Repository Contents
- `MDE_PointCloud_Registration.py`: main Python implementation of the registration workflow
- `README.md`: repository description and usage notes
- `requirements.txt`: Python package requirements
- Due to file size constraints, the 3D point cloud datasets (TLS and UAV) used in this study are hosted in the Releases section.

* **Download the Dataset:** [Hasakoy/Sasima Dataset (v1.0.0-review)](https://github.com/GeoHusrev/hasakoy-sasima-church-registration/releases/tag/v1.0.0-review)

## Disclaimer
This code was prepared to support the methodology presented in the associated manuscript. It is provided for research and academic use only, without any express or implied warranty. Because numerical optimization and registration workflows may be affected by software versions, hardware, stochastic initialization, and input data characteristics, exact numerical reproduction may vary across environments. Users are responsible for validating the code and all derived results in their own applications.

## Recommended Citation
If you use this repository, please cite the associated manuscript (Robust Multi-Sensor Point Cloud Registration for Cultural Heritage Documentation: A Multi-Population Differential Evolution Approach) and the original study describing the Multi-population Differential Evolution (MDE) algorithm (Detection of object boundary from point cloud by using multi-population based differential evolution algorithm).
