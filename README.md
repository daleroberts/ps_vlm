**PSVLM** generates measurements of vertical land motion using Sentinel-1 Synthetic Aperture Radar (SAR) observations and a persistent scatterer approach.

In other words, this code takes a time series of SAR observations and estimates the vertical land motion by considering differences of phase between pairs of SAR observations. It only does this at locations across the landscape that are considered stable, or persistent, over time.

<img src="https://github.com/daleroberts/ps_vlm/blob/8950c9cedd7af6a360dd6c3176b950aa8fe1039b/docs/cover.png" width="800">

## Overview

This self-contained codebase provides:

- A pipeline for transforming the raw SAR observations into interferograms using the GAMMA software, 
- A pure Python implementation of the *Stanford Method for Persistent Scatterers* (STaMPS) methodology.

The code is designed as a starting point for research and development of persistent scatterer interferometry (PSI) methods.

### Workflow

The workflow for generating the data from raw SAR observations can be found in the `gamma` directory. This workflow is highly flexible and can be adapted to different data sources and processing requirements. The workflow is designed to be run on the supercomputer at the National Computational Infrastructure (NCI) in Australia, as such, some assumptions on the environment are made.

The workflow has the ability to specify an area of interest (AOI) and a time period of interest (TOI) and it will automatically find the relevant SAR observations that intersects the AOI. The workflow will then process the data using the GAMMA software to generate the interferograms and other ancillary data required for the STaMPS methodology. The Copernicus Hub located at the NCI contains data that covers the region of Australia, its surrounding oceans, and Antarctica.

### STaMPS

The STaMPS methodology is implemented in the `psvlm.py` code. The code is written in pure Python and is designed to be easy to understand and modify. The code is designed to be run on a single machine and is not parallelised. The code is designed to be run on the output of the workflow.

The original STaMPS implementation was written in C++ and MATLAB. The C++ code was used to perform the identification of persistent scatterers and extraction of time series from the SAR observations. The MATLAB code was used to perform the estimation of the vertical land motion from the time series. This code repository provides a pure Python implementation of both those steps where the first step is now called "Stage 0" in this current implementation.

The methodology consists of the following stages:

  - Stage 0: Preprocessing and finding candidate PS pixels
  - Stage 1: Load data for the candidate PS pixels
  - Stage 2: Estimate the initial coherence
  - Stage 3: Select stable pixels from candidate pixels
  - Stage 4: Weeding out unstable pixels chosen in stage 3
  - Stage 5: Correct the phase for look angle errors
  - Stage 6: Unwrapping the phase
  - Stage 7: Spatial filtering of the unwrapped phase

