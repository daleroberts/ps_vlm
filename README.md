This is a prototype of SAR workflow code called *aludra* that aims to replace the overly-complex [ga_sar_workflow](https://github.com/geoscienceAustralia/ga_sar_workflow) with something much simpler and more flexible. The main GAMMA workflow is contained in the file `run_gamma_workflow` and clocks in at 21 lines of code.

> [!important]  
> Since this is a prototype, this code is not to be shared outside GA's Geodescy group until it has been rewritten to be a more complete solution in the next stage of the project.

## Overview

The projects [ga_sar_workflow](https://github.com/geoscienceAustralia/ga_sar_workflow) + [PyRate](https://github.com/GeoscienceAustralia/PyRate) combine together to perform a *distributed scatterer* approach to InSAR. In this project, we are trying a completely different methodology called the *persistent scatterer* approach.

The first stage of this project was to: (1) prototype a PS approach using GAMMA and StaMPS to see if this is actually what we want to do (e.g., do the outputs look ok? and can we validate them?), (2) determine if we can write a more flexible and simpler GAMMA workflow that allows us to consider any area of interest on-the-fly (including areas outside of Australia), automatically pull-in new observations as they become available, and can scale-up to generate a continental-scale output. We note that ga_sar_workflow is unable to do any of those things.

The second stage of the project will be to: (1) Remove the Matlab / StaMPS dependency and implement our own PS approach in Python, (2) Combine the GAMMA driver and our PS approach into a single code to obtain numerous processing and storage efficiences. This is achievable as the PS approach only works on a sparse number of spatial locations across the landscape.

## Running the workflow

Briefly, the minimal way to run this workflow over an area in Australia is to first create a working directory where the outputs will be generated.
```bash
mkdir prep
cd prep
```
Define an area of interest by specifying a bounding box in "minX minY maxX maxY" (W,S,E,N) format in the file `aoi.txt`.
```bash
echo "149.12747394 -35.39624112 149.15662046 -35.37862325" > aoi.txt
```
Define a date range to consider in the file `daterange.txt`.
```bash
echo "20190101 20230101" > daterange.txt
```
Submit the job (where I have assumed that aludra is in your home directory).
```
qsub ~/aludra/run_gamma_workflow
```
