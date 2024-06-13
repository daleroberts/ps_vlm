**PSVLM** generates measurements of vertical land motion using Sentinel-1 Synthetic Aperture Radar (SAR) observations and a persistent scatterer approach.

In other words, this code takes a time series of SAR observations and estimates the vertical land motion by considering differences of phase between pairs of SAR observations. It only does this at locations across the landscape that are considered stable, or persistent, over time.

<img src="https://github.com/daleroberts/ps_vlm/blob/8950c9cedd7af6a360dd6c3176b950aa8fe1039b/docs/cover.png" width="800">

## Overview

This self-contained codebase provides:

- A pipeline for preparing the data using the GAMMA software, 
- A pure Python implementation of the *Stanford Method for Persistent Scatterers* (STaMPS) methodology.

The code is designed as a starting point for research and development of persistent scatterer interferometry (PSI) methods.


