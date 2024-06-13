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


## Installation


## Usage

The code can be run from the command line using the `psvlm.py` script. By default, the code will run all the stages from 0 to 7. Alternatively, the code can be run for a specific stage by specifying the stage number. For example, to run stage 0, then stages 3-5, the following command can be used:
```
./psvlm.py 0 3-5
```
This allows the user to run the code in a step-by-step manner and inspect the output at each stage, or to re-run various stages, or to restart the code from a specific stage.

### Output messages

By default the code is verbose and will output messages to the console. Output can be disabled using the following option:

- `-q`, `--quiet`: Disable verbose outputs

In certain stages of processing, the code displays "fancy" progress bars when processing is run in interactive mode. When not run in interactive mode, for example when the output is piped to a file, the progress bars are switch to a more simple style. The more simple style of progress bar can also be switched on using the following option:

- `--nofancy`: Disable fancy outputs

If you would like more control over the program output, please see the logging section below.


### Logging

By default, the code simply outputs messages to stdout in a simple manner. If you would instead like to use the `logging` module, this can be enabled using the following option:

- `--logging`: Use the `logging` module

A logging configuration file can be specified using the following option:

- `--logconfig LOGCONFIG`: Use `logging` configuration file

For example, the following configuration file can be used to log messages to a file:

```
[loggers]
keys=root

[handlers]
keys=file

[formatters]
keys=formatter

[logger_root]
level=DEBUG
handlers=file

[handler_file]
class=FileHandler
level=DEBUG
formatter=formatter
args=('psvlm.log',)
filename=psvlm.log

[formatter_formatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

We refer the reader to the `logging` module documentation for more information on how to configure logging.

### Configuration

- `-c CONFIG`, `--config CONFIG`: Configuration file in .toml format
- `--params`: Print all parameters

### Debugging

- `-d`, `--debug`: Enable debug outputs

### Testing

- `--test`: Run the tests

### Processor option

- `--processor PROCESSOR`: Processor to use

### Location of dependency executables

- `--triangle TRIANGLE`: Triangle executable
- `--snaphu SNAPHU`: Snaphu executable


### Limiting the maximum memory usage

- `--maxmem MAXMEM`: Maximum memory usage

### Setting the master date

- `--master_date YYYYMMDD`: Master date

### Setting the data directory

- `--datadir DATADIR`: Data directory

### Threshold for persistent scatterers

- `--da_thresh DA_THRESH`: DA threshold

### Range and azimuth patches

- `--rg_patches RG_PATCHES`: Number of range patches
- `--az_patches AZ_PATCHES`: Number of azimuth patches
- `--rg_overlap RG_OVERLAP`: Range overlap
- `--az_overlap AZ_OVERLAP`: Azimuth overlap

### Mask file

- `--maskfile MASKFILE`: Mask file

### Check against MATLAB outputs

- `--check`: Check against MATLAB outputs

