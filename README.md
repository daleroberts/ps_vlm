This is a prototype of SAR workflow code called *aludra* that aims to replace the overly-complex [ga_sar_workflow](https://github.com/geoscienceAustralia/ga_sar_workflow) with something much simpler and more flexible. The main GAMMA workflow is contained in the file `run_gamma_workflow` and clocks-in at 20 lines of code.

> [!important]  
> Since this is a prototype, this code is not to be shared outside GA's Geodescy group until it has been rewritten to be a more complete solution in the next stage of the project.

## Overview

The projects [ga_sar_workflow](https://github.com/geoscienceAustralia/ga_sar_workflow) + [PyRate](https://github.com/GeoscienceAustralia/PyRate) combine together to perform a *distributed scatterer* approach to InSAR. In this project, we are trying a completely different methodology called the *persistent scatterer* approach.

The first stage of this project was to: (1) prototype a PS approach using GAMMA and StaMPS to see if this is actually what we want to do (e.g., do the outputs look ok? and can we validate them?), (2) determine if we can write a more flexible and simpler GAMMA workflow that allows us to consider any area of interest on-the-fly (including areas outside of Australia), automatically pull-in new observations as they become available, and can scale-up to generate a continental-scale output. We note that ga_sar_workflow is unable to do any of those things.

The second stage of the project will be to: (1) Remove the Matlab / StaMPS dependency and implement our own PS approach in Python, (2) Combine the GAMMA driver and our PS approach into a single code to obtain numerous processing and storage efficiences. This is achievable as the PS approach only works on a sparse number of spatial locations across the landscape.

## Initial setup on NCI

These are just some initial housekeeping things that you might want to check and setup before you get started. If all this is sorted, just skip to the next 'Running the workflow' section.

### Setup SSH link to GitHub and clone repository

You may have the all this setup already but just in case you haven't. First, setup your GitHub access on NCI:
- [Generating a SSH key](https://docs.github.com/authentication/connecting-to-github-with-ssh)
- Adding your authentication key on GitHub [here](https://github.com/settings/keys)

Once that is done, you should be able to do:
```bash
cd ~
git clone git@github.com:daleroberts/aludra.git
```
This will create a path called `aludra` in your home directory with all the code.

### Environment

I have created a self-contained environment with Python, GDAL, GAMMA, etc. under `/g/data/dg9` and it assumes that the `aludra` path is under your home directory and it will add all these commands to your path if you do `source /g/data/dg9/env`. The PBS job command does this automatically so this is only necessary if you want to run the commands individually.

### Data access

You will need access to the following projects on the NCI:

- `dg9` which is the Geodesy InSAR project
- `fj7` which is the CopernicusHub project, this gives you access to the Sentinel-1 data

You can request access [here](https://my.nci.org.au/mancini/login?next=/mancini/).

## Running the workflow

Briefly, the minimal way to run this workflow over an area in Australia is to first create a working directory where the outputs will be generated. You should probably do this under the `/g/data/dg9` directory, e.g.,
```bash
mkdir -p /g/data/dg9/$USER/prep
cd /g/data/dg9/$USER/prep
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
This will generate SLCs (`*.slc`), MLIs in radar coordinates (`*.mli`), and inteferograms (`*.diff`) in radar coordinates. It will also generate the DEM-related heights in radar coordinates. There will also be some images for quick visualisation of outputs (`*.bmp`).
