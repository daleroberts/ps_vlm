**PSVLM** generates measurements of vertical land motion using Sentinel-1 Synthetic Aperture Radar (SAR) observations and a persistent scatterer approach.

In other words, this code takes a time series of SAR observations and estimates the vertical land motion by considering differences of phase between pairs of SAR observations. It only does this at locations across the landscape that are considered stable, or persistent, over time.

<img src="https://github.com/daleroberts/ps_vlm/blob/8950c9cedd7af6a360dd6c3176b950aa8fe1039b/docs/cover.png" width="800">

## Overview

This self-contained codebase provides:

- A pipeline for preparing the data using the GAMMA software, 
- A pure Python implementation of the *Stanford Method for Persistent Scatterers* (STaMPS) methodology.

The code is designed as a starting point for research and development of persistent scatterer interferometry (PSI) methods.

### Workflow

The workflow for generating the data from raw SAR observations can be found in the `gamma` directory. This workflow is highly fexible and can be adapted to different data sources and processing requirements. The workflow is designed to be run on the supercomputer at the National Computational Infrastructure (NCI) in Australia, as such, some assumptions on the environment are made.

The workflow has the ability to specify an area of interest (AOI) and a time period of interest (TOI) and it will automatically find the relevant SAR observations. The workflow will then process the data using the GAMMA software to generate the interferograms and other ancillary data required for the STaMPS methodology.

### STaMPS

The STaMPS methodology is implemented in the `psvlm.py` code. The code is written in pure Python and is designed to be easy to understand and modify. The code is designed to be run on a single machine and is not parallelised. The code is designed to be run on the output of the workflow.

