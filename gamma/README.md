This is a SAR workflow for generating data for the persistent scatter code that aims to replace the overly-complex [ga_sar_workflow](https://github.com/geoscienceAustralia/ga_sar_workflow) with something much simpler and more flexible. The main GAMMA workflow is contained in the file `run_gamma_workflow` and clocks-in at about 20 lines of code.

## Overview

The projects [ga_sar_workflow](https://github.com/geoscienceAustralia/ga_sar_workflow) + [PyRate](https://github.com/GeoscienceAustralia/PyRate) combine together to perform a *distributed scatterer* approach to InSAR. In this project, we are performing a different methodology called the *persistent scatterer* approach.

## Initial setup on NCI

These are just some initial housekeeping things that you might want to check and setup before you get started. If all this is sorted, just skip to the next 'Running the workflow' section.

### Setup SSH link to GitHub and clone repository

You may have the all this setup already but just in case you haven't. First, setup your GitHub access on NCI:
- [Generating a SSH key](https://docs.github.com/authentication/connecting-to-github-with-ssh)
- Adding your authentication key on GitHub [here](https://github.com/settings/keys)

Once that is done, you should be able to do:
```bash
cd ~
git clone git@github.com:daleroberts/ps_vlm.git
```
This will create a path called `ps_vlm` in your home directory with all the code.

### Environment

I have created a self-contained environment with Python, GDAL, GAMMA, etc. under `/g/data/dg9` and it assumes that the `ps_vlm` path is under your home directory and it will add all these commands to your path if you do `source /g/data/dg9/env`. The PBS job command does this automatically so this is only necessary if you want to run the commands individually.

### Data access

You will need access to the following projects on the NCI:

- `dg9` which is the Geodesy InSAR project
- `fj7` which is the CopernicusHub project, this gives you access to the Sentinel-1 data

You can request access [here](https://my.nci.org.au/mancini/login?next=/mancini/).

## Running the workflow (minimal example)

Briefly, the minimal way to run this workflow over an area in Australia is to first create a working directory where the outputs will be generated. You should probably do this under the `/g/data/dg9` directory, e.g.,
```bash
mkdir -p /g/data/dg9/$USER/canberra/prep
cd /g/data/dg9/$USER/canberra/prep
```
This has created a working directory for the Canberra area. You can replace `canberra` with whatever you like. The name `prep` is just a convention to indicate that this is the pre-processing (preparation) step. This is where you will run the workflow and many files will be generated here. For large areas or deep time series, this directory can get quite large.

Now define an area of interest by specifying a bounding box in "minX minY maxX maxY" (W,S,E,N) format in the file `aoi.txt`.
```bash
echo "149.12747394 -35.39624112 149.15662046 -35.37862325" > aoi.txt
```
Define a date range to consider in the file `daterange.txt`.
```bash
echo "20190101 20230101" > daterange.txt
```
Submit the job (where I have assumed that `ps_vlm` is in your home directory) and you are currently in the `prep` directory.
```
qsub ~/ps_vlm/gamma/run_gamma_workflow
```
This will generate SLCs (`*.slc`), MLIs in radar coordinates (`*.mli`), and inteferograms (`*.diff`) in radar coordinates. It will also generate the DEM-related heights in radar coordinates. There will also be some images for quick visualisation of outputs (`*.bmp`).

## Documentation

This workflow is designed to use GAMMA as much as possible and only supplement it where necessary for our purposes. As such, I will assume you are familiar with SAR and using GAMMA, so I will only document the additional programs that are available in this repository.

### `workflow_params`

This file is a program that sets up the parameters for the workflow. It magically defines the following environment variables that can be used in the bash workflow script (`run_gamma_workflow`):

 - `aoi` - the area of interest in "minX minY maxX maxY" format
 - `startdate` - the start date of the time series
 - `enddate` - the end date of the time series
 - `opodspath` - the path to the Sentinel-1 precise orbit files
 - `dem` - the path to the DEM file which can be a very large file covering all of Australia (for example)
 - `passdir` - the pass direction of the Sentinel-1 satellite: `Ascending` or `Descending`
 - `map_srs` - the map spatial reference system to use
 - `aoi_srs` - the spatial reference system of the area of interest
 - `pol` - the polarisation to use: `vv`, `vh`, `hv`, or `hh`
 - `azlks` - the azimuth looks to use
 - `rlks` - the range looks to use
 - `cwd` - the current working directory
 - `workdir` - the working directory where the outputs will be generated

This program is intended to be called at the start of the `run_gamma_workflow` script as `eval $(workflow_params)`. This sets up the environment variables for the workflow.

The parameters above can also be set in your environment, for example:
```bash
export aoi="149.12747394 -35.39624112 149.15662046 -35.37862325"
export startdate="20190101"
export enddate="20230101"
```
If they are set, then the `workflow_params` program will not override them.

The parameters can also be set by creating a file called `{param}.txt` in the working directory. For example, to set the area of interest, you can create a file called `aoi.txt` with the contents:
```
149.12747394 -35.39624112 149.15662046 -35.37862325
```

Certain parameters are required for the workflow to run, such as the area of interest and the date range. If these are not set, the workflow will exit with an error. The following parameters have the following defaults:

```bash
opodspath='/g/data/dg9/orbits'
dem='/g/data/dg9/copernicus-dem/dem-australia.tif'
passdir='Descending'
map_srs=32755
aoi_srs=4326
pol='vv'
azlks=1
rlks=1
cwd=$(pwd)
workdir= '$cwd/workdir'
```

### `find_obs`

This program quickly finds Sentinel-1 observations in a given time range and optionally within a given bounding box. We assume that the data is stored in the NCI's `/g/data/fj7/Copernicus/Sentinel-1/C-SAR/SLC` directory using the Copernicus Hub naming convention. The beauty of this code is that it does not require a database or any other indexing system. It simply reads the directory structure and file names to find the observations and it does so very quickly. This also means that it is not dependent on any particular database or indexing system and any new observations will be found automatically.

The program has the following arguments:

 - `-from` - the start date of the time range
 - `-to` - the end date of the time range
 - `-passdir` - the pass direction of the satellite: `Ascending` or `Descending`
 - `-aoi` - the area of interest in "minX minY maxX maxY" format
 - `-aoifile` - a file containing the area of interests in "minX minY maxX maxY" format, one per line
 - `-bbox` - the bounding box of the area of interest in "minX minY maxX maxY" format
 - `-aoiidx` - the index of the area of interest in the file `aoifile`
 - `-out` - the output file to write the observation locations to
 - `-root` - the root directory of the Sentinel-1 data (default is `/g/data/fj7/Copernicus/Sentinel-1/C-SAR/SLC`)
 - `-workers` - the number of workers to use (default is 10 or `$PBS_NCPUS` if running on NCI)

Note that this program is explicitly designed to be run on the NCI and it will use the PBS job system to run in parallel. The number of workers can be set with the `-workers` argument. If this is not set, it will default to 10 or the number of CPUs available in the PBS job.

### `collect_obs_by_date`

This simple program collects the observations found by `find_obs` and organises them by date. This is useful for the next step in the workflow which is to generate the SLCs, MLIs, and interferograms. The program has the following arguments:

 - `input_obs` - the input file containing the observations
 - `template` - the template to use to organise the observations by date

```bash
collect_obs_by_date input_obs --template "{date}.zip_files"
```
This command will generate a number of files in the current directory with the format `YYYYMMDD.zip_files` where `YYYYMMDD` is the date of the observations. Inside these files are the observations for that date.

### `mosaic_aoi`

This program creates a `SLC_tab` file for a given AOI and a list of S1 zip files.                                              
                                                                                                                            
It takes the following arguments:
 - `srs`: the spatial reference system of the AOI
 - `aoifn`: the filename of the AOI which will be stored as a GeoJSON, the default is 'aoi.geojson'                           
 - `aoi`: the bounding box of the AOI in format (minX, minY, maxX, maxY)  
 - `opods`: the directory with OPOD state vector files                                                                        
 - `outfn`: the filename of the `SLC_tab` file default is `SLC_tab`                                                   
 - `polarisation`: the polarisation of the S1 data, default is `vv`
 - `dtype`: the data type of the S1 data, default is 0                                                                         
 - `zipfiles`: the list of S1 zip files                           
                                                                                                                                  
If zipfiles is a single file with the extension `.zip_files` it is assumed to be a text file with a list of S1 zip files. These files are read into the list zipfiles.                                                                                                                                                                                                    
It achieves this doing the following steps:
 - Unzips the S1 zip files but excludes the tiff files       
 - Creates a GeoJSON file with the AOI          
 - Creates a GeoJSON file with the burst geometries intersecting with the AOI
 - Creates a `SLC_tab` file for each zip file with the bursts intersecting with the AOI

For example, in the workflow script, you might have the following command:
```bash
mosaic_aoi -bbox $(echo $aoi) -must_contain -unique -opods ${opodspath} -pol ${pol} -twd $workdir -template "{date}.{pol}.SLC_tab" *.zip_files
```
