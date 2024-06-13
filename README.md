**PSVLM** generates measurements of vertical land motion using Sentinel-1 Synthetic Aperture Radar (SAR) observations and a persistent scatterer approach. 

In other words, this code takes a time series of SAR observations and estimates the vertical land motion by considering differences of phase between pairs of SAR observations. It only does this at locations across the landscape that are considered stable, or persistent, over time.

This a self-contained codebase that provides a fully working pipeline for preparing the data using the GAMMA software, and a pure Python implementation of the STaMPS methodology for Persistent Scatterer Interferometry (PSI) processing. The code is based on the original MATLAB and C++ code from the Stanford Method for Persistent Scatterers (StaMPS) project.
