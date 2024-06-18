#!/usr/bin/env python3.12
"""
This is a Python implementation of:

  "Hooper A; Bekaert D; Spaans K; Arikan M (2012), Recent advances in SAR
  interferometry time series analysis for measuring crustal deformation,
  Tectonophysics, 514-517, pp.1-13. doi: 10.1016/j.tecto.2011.10.013".

Otherwise known as the "Stanford Method for Persistent Scatterers" (StaMPS)
methodology which is an InSAR persistent scatterer (PS) method developed to
work even in terrains devoid of man-made structures and/or undergoing
non-steady deformation.

This Python implementation is written by Dale Roberts (github.com/daleroberts),
and is based on the original MATLAB code by Andrew Hooper et al.

The methodology consists of the following stages:

  - Stage 0: Preprocessing and finding candidate PS pixels
  - Stage 1: Load data for the candidate PS pixels
  - Stage 2: Estimate the initial coherence
  - Stage 3: Select stable pixels from candidate pixels
  - Stage 4: Weeding out unstable pixels chosen in stage 3
  - Stage 5: Correct the phase for look angle errors
  - Stage 6: Unwrapping the phase
  - Stage 7: Spatial filtering of the unwrapped phase
  ...

Note that Stage 0 in the original code is a combination of Bash and C++ codes
and triggered by the `mt_prep_????` command. This stage has now been replaced
by Python code and called "Stage 0". Therefore, you will not find any
reference to this stage in the original documentation of StaMPS.

The code has been written to have only minimal dependencies on external
libraries. The only required python libraries are `numpy` and `scipy`. The
code will also need access to the `triangle` and `snaphu` executables, which
are used for various Delaunay triangulations and phase unwrapping, respectively.

Note: At this stage the code is still under development and:
  - written in a single file for ease of (planned) refactoring,
  - not yet fully optimized for speed,
  - only the default code execution path has been tested,
  - is a bug-for-bug compatible implementation of the original MATLAB code,
  - only parses outputs from the GAMMA software suite,
  - does not include any (planned) method improvements beyond the original StaMPS.

"""

import numpy as np
import subprocess
import sys
import os

from scipy.signal import fftconvolve, convolve2d, lfilter, firls
from scipy.signal.windows import gaussian
from scipy.optimize import least_squares
from scipy.fft import fftshift, ifftshift  # FIXME: do we need these? replace by np.fft?
from scipy.spatial import KDTree

from datetime import datetime, timezone, timedelta
from contextlib import ExitStack, chdir
from pathlib import Path

from typing import TextIO, Any, Dict, Tuple, Optional, List, no_type_check
from numpy.typing import NDArray as Array


np.set_printoptions(
    precision=4, suppress=True, linewidth=sys.maxsize, threshold=sys.maxsize
)

# These constants can be overridden using command-line arguments

TRIANGLE: str = os.getenv("TRIANGLE", "triangle")
SNAPHU: str = os.getenv("SNAPHU", "snaphu")
PROCESSOR: str = "gamma"
FANCY_PROGRESS: bool = True
VERBOSE: bool = True
DEBUG: bool = False
OPTIONS: Dict[str, Any] = {}

# Default options for the StaMPS configuration file in .toml format

DEFAULT_OPTIONS: str = """
clap_alpha = 1
clap_beta = 0.3
clap_low_pass_wavelength = 800
clap_win = 32
density_rand = 20
drop_ifg_index = []
filter_grid_size = 50
filter_weighting = 'P-square'
gamma_change_convergence = 0.005
gamma_max_iterations = 3
gamma_stdev_reject = 0
lambda = 0.055465759531382094
max_topo_err = 20
percent_rand = 20
ref_centre_lonlat = 0
ref_lat = -inf
ref_lon = -inf
ref_radius = inf
ref_x = ''
ref_y = ''
sb_scla_drop_index = ''
scla_deramp = 'y'
scla_drop_index = []
scla_method = 'L2'
select_method = 'DENSITY'
slc_osf = 1
small_baseline_flag = 'n'
subtr_tropo = 'n'
tropo_method = 'a_l'
unwrap_gold_alpha = 0.8
unwrap_gold_n_win = 8
unwrap_grid_size = 200
unwrap_hold_good_values = 'n'
unwrap_la_error_flag = 'y'
unwrap_method = '3D'
unwrap_patch_phase = 'n'
unwrap_prefilter_flag = 'y'
unwrap_spatial_cost_func_flag = 'n'
unwrap_time_win = 730
weed_max_noise = inf
weed_neighbours = 'n'
weed_standard_dev = 1
weed_time_win = 730
weed_zero_elevation = 'n'
"""


class dotdict(dict):
    """
    A dictionary that allows access to its keys as attributes.

    This gives MATLAB style dot access to dictionary keys. For example, instead
    of `d['key']` you can use `d.key`. All dict-style data loaded from the
    StaMPS files using `stamps_load` will be stored in this type of dictionary.
    """

    @no_type_check
    def __getattr__(self, x):
        return dict.get(self, x)

    @no_type_check
    def __setattr__(self, x):
        return dict.__setitem__(self, x)

    @no_type_check
    def __getstate__(self) -> dict:
        return self.__dict__

    @no_type_check
    def __setstate__(self, d: dict) -> None:
        self.__dict__.update(d)

    __delattr__ = dict.__delitem__  # type: ignore
    __dir__ = dict.keys  # type: ignore


class PrepareData:
    """
    Base class for "Stage 0" of the StaMPS processing chain.

    This stage takes care of finding candidate pixels for persistent scatterers
    in the SLC data. Once the candidate pixels have been identified, the phase
    time series are extracted for each candidate pixel and all their associated
    data such as locations (lon/lat), heights, etc.

    This replaces all the bash and C++ code in the original StaMPS code with
    Python code. The main goal is to make the code more readable and easier to
    maintain.

    This class should be subclassed for each specific data format (e.g. GAMMA,
    GSAR, SNAP, etc.). However, a lot of the common functionality can be
    implemented in this (base) class.

    Depending on the value of PROCESSOR (e.g. "gamma", "doris", "snap", "gsar"),
    the appropriate data processing class will be (magically) instantiated when
    "Stage 0" is run. If you want to implement the "Foo" processor, you will
    need to subclass this class and name it "FooPrepareData". Then it can be
    used by setting PROCESSOR to "foo" (using the command-line argument).
    """

    def __init__(
        self,
        master_date: str,
        datadir: Path,
        da_thresh: float = 0.4,
        rg_patches: int = 1,
        az_patches: int = 1,
        rg_overlap: int = 50,
        az_overlap: int = 50,
        maskfile: Optional[Path] = None,
    ):
        self.processor = PROCESSOR

        self.master_date = master_date
        self.datadir = datadir.resolve()
        self.da_thresh = da_thresh
        self.rg_patches = rg_patches
        self.az_patches = az_patches
        self.rg_overlap = rg_overlap
        self.az_overlap = az_overlap
        self.maskfile = maskfile
        self.width = 0
        self.length = 0
        self.precision = "f"
        self.workdir = Path(".").resolve()

        log(f"workdir = {self.workdir}")
        log(f"master_date = {self.master_date}")
        log(f"datadir = {self.datadir}")
        log(f"da_thresh = {self.da_thresh}")
        log(f"rg_patches = {self.rg_patches}")
        log(f"az_patches = {self.az_patches}")
        log(f"rg_overlap = {self.rg_overlap}")
        log(f"az_overlap = {self.az_overlap}")
        log(f"maskfile = {self.maskfile}")

        # Find the SLC directory

        self.slcdir = next(self.datadir.glob("*slc"), None)
        assert self.slcdir is not None, f"SLC directory not found in {self.datadir}"

        # Find the RSC file

        self.rscfile = next(
            (self.datadir / self.slcdir).glob(f"{self.master_date}.*slc.par"), None
        )
        assert (
            self.rscfile is not None
        ), f"RSC file not found in {self.datadir / self.slcdir}"
        self.rscfile = self.rscfile.resolve()  # get the full path

        self.width, self.length, self.precision = self.parse_rsc_file(self.rscfile)

        if self.precision == "FCOMPLEX":
            prec = "f"
        else:
            prec = "s"

        log(f"rscfile = {self.rscfile}")
        log(f"width = {self.width}")
        log(f"length = {self.length}")
        log(f"precision = {self.precision} ({prec})")

        # Generate some global configuration files (width, length, processor, etc.)

        self.generate_global_config_files()

        # Generate the dem configuration files. This configuration file is used by
        # the `extract_lonlats`, `extract_heights`, and `extract_phases` methods.

        self.generate_dem_config_file(
            self.workdir / "pscdem.in", self.workdir / "psclonlat.in"
        )

        # Generate the patch configuration files. These are used by the `select_candidate_pixels`
        # method to extract the candidate pixels from the SLC data.

        self.generate_patch_config_files()

        # Find the SLC files

        slcs = list((self.datadir / self.slcdir).glob("*.*slc"))

        log(f"Found {len(slcs)} SLCs to process in {self.datadir / self.slcdir}")

        if len(slcs) == 0:
            raise FileNotFoundError(
                f"No SLC files found in {self.datadir / self.slcdir}"
            )

        # Calibrate the amplitude of the SLC data

        with open(self.workdir / "calamp.in", "w") as f:
            for slc in slcs:
                f.write(f"{slc}\n")

        self.calibrate_amplitude(
            self.workdir / "calamp.in",
            self.width,
            self.workdir / "calamp.out",
            prec,
            maskfile=self.maskfile,
        )

        # Generate the differential interferogram configuration files

        self.generate_diff_config_files(
            self.master_date, self.workdir / "calamp.out", self.workdir / "pscphase.in"
        )

    def parse_rsc_file(self, rscfile: Path) -> Tuple[int, int, str]:
        """Parse the RSC file to get the width, length, and precision."""

        for line in open(rscfile):
            if "range_samples" in line:
                width = int(line.split()[1])
            elif "azimuth_lines" in line:
                length = int(line.split()[1])
            elif "image_format" in line:
                precision = line.split()[1]

        return width, length, precision

    def write_param_to_file(self, x: Any, name: str | Path) -> None:
        """Write a parameter to a basic txt file."""

        if not Path(name).suffix:
            name = Path(f"{name}.txt")

        f = Path(name).resolve()

        with f.open("a") as file:
            file.write(f"{x}\n")

    def generate_global_config_files(self) -> None:
        """Write some global parameters to the appropriate txt files."""

        log("Generating global configuration files")

        self.write_param_to_file(self.processor, "processor")
        self.write_param_to_file(self.width, "width")
        self.write_param_to_file(self.length, "len")
        self.write_param_to_file(str(self.rscfile), "rsc")

    def generate_dem_config_file(self, demfn: Path, lonlatfn: Path) -> None:
        """Write the DEM parameters to `pscdem.in` file."""

        log("Generating DEM configuration files")

        with open(demfn, "w") as f:
            f.write(f"{self.width}\n")
            for dem in self.datadir.glob("*dem.rdc"):
                f.write(f"{dem}\n")

        geopath = self.datadir / "geo"
        lonfn = next(geopath.glob("*.lon"), None)
        latfn = next(geopath.glob("*.lat"), None)

        with open(lonlatfn, "a") as f:
            f.write(f"{self.width}\n")

            if lonfn is not None and latfn is not None:
                f.write(f"{lonfn}\n")
                f.write(f"{latfn}\n")

    def generate_patch_config_files(self) -> None:
        """Generate the patch configuration files."""

        log("Generating patch configuration files and directories")

        # Generate the `selpsc.in` file, this is later used by the
        # `select_candidate_pixels` method

        selfile = Path(self.workdir / "selpsc.in").resolve()

        selfile.unlink(missing_ok=True)

        self.write_param_to_file(self.da_thresh, selfile)
        self.write_param_to_file(self.width, selfile)
        self.write_param_to_file(f"{self.workdir}/calamp.out", selfile)

        width_p = self.width // self.rg_patches
        length_p = self.length // self.az_patches

        log(f"{width_p = }")
        log(f"{length_p = }")

        # Remove the old patch list file, if it exists.
        # Note that we do not remove the patch directories as we
        # prefer to overwrite them. We cannot do the same for the
        # `patch.list` file as we append to it.

        patch_list = (self.workdir / "patch.list").resolve()

        log(f"Generating patch list file `{patch_list}`")

        patch_list.unlink(missing_ok=True)

        # Generate the patch directories and patch list file

        for irg in range(self.rg_patches):
            for iaz in range(self.az_patches):
                start_rg1 = width_p * irg
                start_rg = start_rg1 - self.rg_overlap
                if start_rg < 1:
                    start_rg = 1

                end_rg1 = width_p * (irg + 1)
                end_rg = end_rg1 + self.rg_overlap
                if end_rg > self.width:
                    end_rg = self.width

                start_az1 = length_p * iaz
                start_az = start_az1 - self.az_overlap
                if start_az < 1:
                    start_az = 1

                end_az1 = length_p * (iaz + 1)
                end_az = end_az1 + self.az_overlap
                if end_az > self.length:
                    end_az = self.length

                patch_dir = self.workdir / f"PATCH_{irg * self.az_patches + iaz + 1}"
                patch_dir.mkdir(exist_ok=True)
                patch_dir = patch_dir.resolve()

                log(f"Creating patch directory: {patch_dir}")

                with open(patch_dir / "patch.in", "w") as f:
                    f.write(f"{start_rg}\n")
                    f.write(f"{end_rg}\n")
                    f.write(f"{start_az}\n")
                    f.write(f"{end_az}\n")

                with open(patch_dir / "patch_noover.in", "w") as f:
                    f.write(f"{start_rg1}\n")
                    f.write(f"{end_rg1}\n")
                    f.write(f"{start_az1}\n")
                    f.write(f"{end_az1}\n")

                # Append to the patch list file

                with open(self.workdir / "patch.list", "a") as f:
                    f.write(f"{patch_dir}\n")

    def calibrate_amplitude(
        self,
        infile: Path,
        width: int,
        outfile: Path,
        prec: str,
        byteswap: bool = False,
        maskfile: Optional[Path] = None,
    ) -> None:
        """Calibrate the amplitude of the SLC data. This is equivalent to the
        `calamp` program."""

        log(f"Calibrating amplitude, using config file: `{infile.resolve()}`")

        if prec == "s":
            typestr = ">h"
        else:
            typestr = ">c8"

        # Identify the master observation

        with open("rsc.txt") as fd:
            masterfn = fd.readline().strip()

        log(f"{masterfn = }")

        # read the heading parameter from the master observation

        with open(masterfn) as fd:
            for line in fd.readlines():
                if line.startswith("heading"):
                    master_heading = float(line.split()[1])
                    break

        log(f"{master_heading = }")

        # read the list of files to process from the input file

        fns = []
        with open(infile) as fd:
            for line in fd.readlines():
                fns.append(line.strip())

        log(f"{len(fns)} files to process")

        # open the parameter file associated with each file in 'fns'
        # and then read the heading parameter. If the heading is
        # different to the master heading, then remove the file from
        # the list of files to process

        log("Filtering observations based on heading")

        fns2 = []
        for fn in fns:
            parfn = fn + ".par"
            with open(parfn) as fd:
                for line in fd.readlines():
                    if line.startswith("heading"):
                        heading = float(line.split()[1])
                        break

            # Additional filtering based on heading

            if np.abs(heading - master_heading) > 0.01:
                log(
                    f"{fn} heading: {heading:9.6f}"
                    f"master_heading: {master_heading:9.6f} - skipping"
                )
            else:
                log(
                    f"{fn} heading: {heading:9.6f} master_heading: {master_heading:9.6f}"
                )
                fns2.append(fn)

        log(f"Before: {len(fns)} obs After: {len(fns2)} obs")

        fns = fns2

        mean_amps = np.zeros(len(fns))
        sd_amps = np.zeros(len(fns))

        with open(outfile, "w") as fd:
            for i, fn in enumerate(fns):
                try:
                    data = np.fromfile(fn, dtype=typestr)
                    data.shape = (-1, width)

                    amp = np.absolute(data)
                    amp[amp <= 10e-6] = np.nan

                    mean_amp = np.nanmean(amp)
                    sd_amp = np.nanstd(amp)

                    mean_amps[i] = mean_amp
                    sd_amps[i] = sd_amp

                    fd.write(f"{fn} {mean_amp}\n")

                    log(f"{fn} mean_amp: {mean_amp:.4f}")
                except ValueError as e:
                    log(f"Error processing {fn}: {e}")

        mu = np.nanmean(mean_amps)
        sd = np.nanstd(mean_amps)

        log(
            f"Files with mean_amp outside 2 sigma: {mu:.4f} +/- {sd:.4f} are flagged with a '*'"
        )
        for i, fn in enumerate(fns):
            star = " " if np.abs(mean_amps[i] - mu) < 2 * sd else "*"
            print(
                f"{fn} mean_amp: {mean_amps[i]:+8.4f} sd_amp: {sd_amps[i]:+8.4f} {star}"
            )

    def generate_diff_config_files(
        self, master_date: str, calampfn: Path, pscphasefn: Path
    ) -> None:
        """Generate the differential interferogram configuration file. Basically,
        use the calibrated amplitude file to generate the differential interferograms
        filenames and write them to a file."""

        log("Generating interferogram configuration files")

        with open(calampfn) as fd, open(pscphasefn, "w") as fo:
            for line in fd:
                fn = Path(line.strip().split()[0])
                stem = fn.stem
                if master_date not in stem:
                    difffn = (
                        fn.parent.parent
                        / "diff0"
                        / Path(f"{master_date}_{fn.stem}.diff")
                    )
                    fo.write(f"{difffn}\n")

    def identify_candidates(
        self,
        do_identify: bool = True,
        do_lonlats: bool = True,
        do_heights: bool = True,
        do_phases: bool = True,
        precision: str = "f",
        byteswap: bool = False,
        maskfile: Optional[Path] = None,
        patchlist: Path = Path("patch.list"),
    ) -> None:
        """Extract candidate locations for scatterers from the Single-Look
        Complex (SLC) data. This is roughly equivalent to the
        `mt_extract_cands` program."""

        with open(patchlist) as f:
            patchdirs = [Path(x.strip()).resolve() for x in f.readlines()]

        for patchdir in patchdirs:
            if do_identify:
                self.select_candidate_pixels(
                    self.workdir / "selpsc.in",
                    patchdir / "patch.in",
                    patchdir / "pscands.1.ij",
                    patchdir / "pscands.1.da",
                    patchdir / "mean_amp.flt",
                    patchdir / "Dsq.flt",
                    patchdir / "pos.bin",
                    precision=precision,
                    byteswap=byteswap,
                )

            if do_lonlats:
                self.extract_lonlats(
                    self.workdir / "psclonlat.in",
                    patchdir / "pscands.1.ij",
                    patchdir / "pscands.1.ll",
                )

            if do_heights:
                self.extract_heights(
                    self.workdir / "pscdem.in",
                    patchdir / "pscands.1.ij",
                    patchdir / "pscands.1.hgt",
                )

            if do_phases:
                self.extract_phases(
                    self.workdir / "pscphase.in",
                    patchdir / "pscands.1.ij",
                    patchdir / "pscands.1.ph",
                )

    def select_candidate_pixels(
        self,
        selpscfn: Path,
        patchfn: Path,
        ijfn: Path,
        dafn: Path,
        meanampfn: Path,
        dsqfn: Path,
        posfn: Path,
        precision: str = "f",
        byteswap: bool = False,
    ) -> None:
        """Select candidate pixels from the SLC data. This is equivalent to the
        `selpsc_patch` program."""

        log("Identifying candidate pixels")

        if precision == "s":
            raise NotImplementedError
        else:
            ts = ">c8"

        log(f"dtype: {ts} ({np.dtype(ts).kind} {np.dtype(ts).itemsize * 8}-bit)")

        calib_factor = []
        D_thresh = 0.0
        width = 0
        fns = []

        log(f"Reading threshold, width, and calampfn from `{selpscfn.resolve()}`")

        with open(selpscfn) as fd:
            D_thresh = float(fd.readline())
            width = int(fd.readline())
            calampfn = Path(fd.readline().strip())

        log(f"Reading calibration factors from calampfn=`{calampfn.resolve()}`")

        with open(calampfn) as fd:
            for line in fd:
                c1, c2 = line.split()
                fns.append(Path(c1))
                calib_factor.append(float(c2))

        log(f"Reading patch parameters from `{patchfn.resolve()}`")

        with open(patchfn) as fd:
            rg_start = int(fd.readline())
            rg_end = int(fd.readline())
            az_start = int(fd.readline())
            az_end = int(fd.readline())

        for fn, c in zip(fns, calib_factor):
            azs, rgs = filedim(fn, width, ts)
            log(f"{fn} mean_amp:{c:6.4f} azs:{azs} rgs:{rgs}")

        calib = np.array(calib_factor)
        nlines, width = filedim(fns[0], width, ts)
        nfiles = len(fns)
        D_sq_thresh = D_thresh**2
        rg_start = max(0, rg_start - 1)
        rg_end = min(width, rg_end - 1)
        az_start = max(0, az_start - 1)
        az_end = min(nlines, az_end - 1)
        pscid = 0

        log(f"nfiles = {nfiles}")
        log(f"dispersion threshold = {D_thresh:.4f}")
        log(f"dispersion-squared threshold = {D_sq_thresh:.4f}")
        log(f"width = {width}")
        log(f"nlines = {nlines}")
        log(f"rg_start = {rg_start}")
        log(f"rg_end = {rg_end}")
        log(f"az_start = {az_start}")
        log(f"az_end = {az_end}")

        nskip = 0

        inazrg: Dict[int, List[int]] = {}

        inazrgfn = Path(selpscfn).resolve().parent / "input_azrg"
        if inazrgfn.exists():
            log(f"Found {inazrgfn}. Also adding points from that file.")
            with open(inazrgfn) as fd:
                for line in fd.readlines():
                    az, rg = [int(x) - 1 for x in line.strip().split()]
                    inazrg.setdefault(az, []).append(rg)

        log(f"Calibrating amplitude and calculating dispersions across {nfiles} files")

        with ExitStack() as stack:
            pfd = stack.enter_context(open(posfn, "w"))
            mfd = stack.enter_context(open(meanampfn, "w"))
            Dsqfd = stack.enter_context(open(dsqfn, "w"))
            ijfd = stack.enter_context(open(ijfn, "w"))
            dafd = stack.enter_context(open(dafn, "w"))
            slcfds = [stack.enter_context(open(f)) for f in fns]

            for az in range(nlines):
                arr = np.array(
                    [np.fromfile(fd, dtype=ts, count=width) for fd in slcfds], dtype=ts
                )

                if not (az_start <= az <= az_end):
                    nskip += 1
                    continue

                # arr = arr[:, rg_start:rg_end]

                amp = np.absolute(arr)

                for i in range(len(calib)):
                    amp[i, :] /= calib[i]

                mask = amp < 0.00005
                amp[mask] = np.nan
                mask = np.count_nonzero(mask, axis=0) > 1

                sumamp = np.nansum(amp, axis=0)
                sumampsq = np.nansum(amp**2, axis=0)

                with np.errstate(divide="ignore", invalid="ignore"):
                    D_sq = nfiles * sumampsq / (sumamp * sumamp) - 1  # var / mean^2

                D_sq[mask] = np.nan

                rgloc = list(np.argwhere(D_sq < D_sq_thresh).flatten())

                if az in inazrg:
                    frgs = inazrg[az]
                    rgloc.extend(frgs)

                pos = np.zeros_like(D_sq, dtype=np.ubyte)
                pos[rgloc] = 1

                show_progress(az, nlines)

                for rg in rgloc:
                    if rg_start <= rg <= rg_end:
                        ijfd.write(f"{pscid} {az + 1} {rg + 1}\n")
                        dafd.write(f"{np.sqrt(D_sq[rg]):.4f}\n")
                        pscid += 1

                sumamp[mask] = 0
                D_sq[mask] = 0

                meanamp = sumamp / nfiles

                meanamp.astype(">f4").tofile(mfd)  # 32-bit float big-endian
                D_sq.astype(">f4").tofile(Dsqfd)
                pos.astype(">B").tofile(pfd)

        log(f"{nskip} lines skipped")

        perc = pscid / ((az_end - az_start) * (rg_end - rg_start)) * 100
        log(f"{pscid} PS candidates generated ({perc:.2f}% of patch pixels)")

    def extract_lonlats(self, lonlatfn: Path, ijfn: Path, llfn: Path) -> None:
        """Extract the longitude and latitude of the candidate pixels. This is roughly
        equivalent to the `psclonlat` program."""

        # TODO: This is not a full implementation of the bash script `psclonlat`

        log(f"Reading lon/lat parameters from `{lonlatfn.resolve()}`")

        with open(lonlatfn, "r") as f:
            width = int(f.readline().strip())
            lonfn = f.readline().strip()
            latfn = f.readline().strip()

        log(f"{width = }")
        log(f"{lonfn = }")
        log(f"{latfn = }")

        log(f"Reading longitudes from `{lonfn}`")

        lon = np.fromfile(lonfn, dtype=np.float32)

        log(f"Reading latitudes from `{latfn}`")

        lat = np.fromfile(latfn, dtype=np.float32)

        log(f"Reading ij parameters from `{ijfn.resolve()}`")

        psc_ids = np.loadtxt(ijfn, delimiter=" ", usecols=(0, 1, 2), dtype=int)

        lonlat = np.zeros((len(psc_ids), 2), dtype=np.float32)
        for i, (pscid, x, y) in enumerate(psc_ids):
            try:
                lon_out = lon[y*width+x]
                lat_out = lat[y*width+x]
                lonlat[i,:] = [lon_out, lat_out]
            except IndexError:
                log(f"IndexError at {x} {y}")

        np.savetxt(llfn, lonlat, fmt="%f")

        log(f"{len(lon_out)} lon/lat pairs written to {llfn}")

    def extract_heights(self, demfn: Path, ijfn: Path, hgtfn: Path) -> None:
        """Extract the heights of the candidate pixels. This is roughly equivalent
        to the `pscdem` program."""

        log("Extracting heights of the candidate pixels")

        with open(demfn) as fd:
            width = int(fd.readline().strip())
            sarhgtfn = Path(fd.readline().strip())

        log(f"{width = }")
        log(f"{sarhgtfn = }")

        with open(sarhgtfn) as sarhgtfd, open(ijfn) as ijfd, open(hgtfn, "w") as hgtfd:
            hgt = np.fromfile(sarhgtfd, dtype=">f4")
            log(f"{hgt.shape = }")

            for line in ijfd:
                n, az, rg = [int(x) for x in line.strip().split()]

                h = hgt[(az - 1) * width + (rg - 1)]
                hgtfd.write(f"{az:>8} {rg:>8} {h:>14.4f}\n")

    def extract_phases(self, paramfn: Path, ijfn: Path, phfn: Path) -> None:
        """Extract the phase of the candidate pixels. This is roughly equivalent
        to the `pscphase` program."""

        log("Extracting time series of phases of the candidate pixels")

        typestr = ">c8"

        log(f"Reading parameters from `{paramfn.resolve()}`")

        with open(paramfn) as fd:
            width = int(fd.readline().strip())
            ifgfns = [Path(line.strip()) for line in fd.readlines()]

        nfiles = len(ifgfns)

        log(f"width (aka. range_samples) = {width}")
        log(f"number of interferograms   = {nfiles}")

        for i, fn in enumerate(ifgfns):
            azs, rgs = filedim(Path(fn), width, typestr)
            log(f"{i:3d}: {fn} azs:{azs} rgs:{rgs}")

        mean_abs_phs = np.zeros(nfiles, dtype="float32")

        with open(ijfn) as ijfd:
            ij_lines = ijfd.readlines()
            nijs = len(ij_lines)

        with open(phfn, "w") as phfd:
            ifgmm = [np.memmap(f, dtype=typestr, shape=(azs, rgs)) for f in ifgfns]

            for i, (fn, ifg) in enumerate(zip(ifgfns, ifgmm)):
                phs = np.empty(nijs, dtype=typestr)
                print(f"{i:3d}: {fn}", end="")
                for k, line in enumerate(ij_lines):
                    n, az, rg = [int(x) - 1 for x in line.strip().split()]
                    ph = np.array(ifg[az, rg], dtype=typestr)
                    if np.isnan(np.absolute(ph)):
                        log(f"NaN at {az} {rg}")
                    ph.tofile(phfd)
                    phs[k] = ph[0]
                phs = np.array(phs, dtype=typestr)
                mean_ph = np.mean(phs)
                mean_abs_ph = np.mean(np.absolute(phs))
                mean_abs_phs[i] = mean_abs_ph
                log(
                    f"\tmean_phase: {mean_ph:+8.4f}\tmean_abs_phase: {mean_abs_ph:+8.4f}"
                )

        log(
            f"Phase time series data of shape {(nfiles, nijs)} written to file `{phfn}`"
        )

        mu = np.mean(mean_abs_phs)
        sigma = np.std(mean_abs_phs)

        log("Summary of mean absolute phases:")

        log(f"mean_abs_phs: {mu:+8.4f} +/- {sigma:+8.4f}")

        for i, fn in enumerate(ifgfns):
            star = " " if np.abs(mean_abs_phs[i] - mu) < 2 * sigma else "*"
            log(f"{fn} {mean_abs_phs[i]:+8.4f} {star}")

    def run(self) -> None:
        raise NotImplementedError("Subclass must implement this method")


class GammaPrepareData(PrepareData):
    def extract_lonlat(self, lonlatfn: Path, ijfn: Path, llfn: Path) -> None:
        """Extract the longitude and latitude of the candidate pixels. This is roughly
        equivalent to the `psclonlat` program."""
        raise NotImplementedError

    def run(self) -> None:
        self.identify_candidates()


class DorisPrepareData(PrepareData):
    def run(self) -> None:
        pass


class SnapPrepareData(PrepareData):
    def run(self) -> None:
        pass


class GsarPrepareData(PrepareData):
    def run(self) -> None:
        pass


def log(msg: str) -> None:
    """Prints a message to stderr."""
    if msg.startswith("#"):
        print("\n+" + msg[1:] + "\n")
    else:
        print(msg)


def show_progress(step: int, total: int, title: Optional[str] = None) -> None:
    """Show a progress bar."""

    if not VERBOSE:
        return

    def simple() -> None:
        incr = total // 10
        dots = incr // 3
        perc = step // incr * 10
        if step == 0 and title:
            print(f"{title}: ", end="", flush=True)
        if step == total - 1:
            print("100 - done.", flush=True)
        if step % incr == 0 and perc != 100:
            print(f"{perc}", end="", flush=True)
        elif step % dots == 0 and perc != 100:
            print(".", end="", flush=True)
        else:
            pass

    def fancy() -> None:
        perc = 100 * float(step + 1) / float(total)
        if title:
            bar_width = os.get_terminal_size().columns - len(title) - 20
        else:
            bar_width = os.get_terminal_size().columns - 20
        blocks = "█▏▎▍▌▋▊▉"
        max_ticks = bar_width * 8
        num_ticks = int(round(perc / 100 * max_ticks))
        full_ticks = num_ticks / 8
        part_ticks = num_ticks % 8
        disp = bar = ""
        bar += blocks[0] * int(full_ticks)
        if part_ticks > 0:
            bar += blocks[part_ticks]
        bar += "∙" * int((max_ticks / 8 - float(num_ticks) / 8.0))
        if title:
            disp = f"{title}: "
        disp += bar
        disp += f" {perc:3.0f}%"
        if int(perc) == 100:
            disp += " - done."
            sys.stdout.write("\r" + disp + "\n")
        else:
            sys.stdout.write("\r" + disp)
        sys.stdout.flush()

    try:
        fancy() if FANCY_PROGRESS else simple()
    except OSError:  # Fallback to simple progress bar
        simple()


def tabulate(data: dict[str, list], precision: int = 16) -> None:
    """Pretty prints a table from a dictionary."""

    # Convert the data to strings.
    data = {
        header: [
            f"{value:.{precision}f}" if isinstance(value, float) else str(value)
            for value in values
        ]
        for header, values in data.items()
    }

    # Determine the maximum width for each column
    column_widths = {
        header: max(len(str(header)), max(len(str(value)) for value in values))
        for header, values in data.items()
    }

    # Print column headers
    for header, width in column_widths.items():
        print(f"{header:>{width}}", end="  ")
    print()

    # Print separator line
    for width in column_widths.values():
        print("─" * width, end="  ")
    print()

    # Print column values
    num_rows = max(len(values) for values in data.values())
    for i in range(num_rows):
        for header, width in column_widths.items():
            value = data[header][i] if i < len(data[header]) else ""
            print(f"{value:>{width}}", end="  ")
        print()


def run_triangle_on(fn: Path) -> None:
    """Run the Triangle program on the given file."""
    cmd = [TRIANGLE, "-V", "-e", str(fn)]
    base = fn.stem
    if VERBOSE:
        out = sys.stdout
        subprocess.call(cmd, stdout=out, stderr=out)
    else:
        with open(f"triangle_{base}.log", "w") as out:
            subprocess.call(cmd, stdout=out, stderr=out)


def run_snaphu_on(fn: Path, ncol: int) -> None:
    """Run the Snaphu program on the given file."""
    cmd = [SNAPHU, "-d", "-f", str(fn), str(ncol)]
    if VERBOSE:
        out = sys.stdout
        log(f"Running: {cmd}")
        subprocess.call(cmd, stdout=out, stderr=out)
    else:
        with open("snaphu.log", "w") as out:
            subprocess.call(cmd, stdout=out, stderr=out)


def filedim(fn: Path, width: int, typestr: str) -> Tuple[int, int]:
    """Determine the dimensions of data in a file based on a
    width and numpy variable dtype string (typestr). e.g.,
        typestr=">c8" for complex64
        typestr=">f8" for float64
        typestr=">f4" for float32
    """
    size = fn.stat().st_size
    nlines = size // (width * np.dtype(typestr).itemsize)
    return (nlines, width)


def chop(x: Array, eps: Optional[float] = None) -> Array:
    """Chop off very small values to zero, which is sometimes useful for
    near-zero imaginary values."""
    if eps is None:
        eps = float(10 * np.finfo(x.dtype).eps)
    return np.where(np.abs(x) < eps, 0, x)


def results_equal(
    name: str,
    tol: float = 1e-8,
    equal_nan: bool = True,
    modulo: Optional[float] = None,
) -> bool:
    """Check if the results of the current run match the expected results that
    were obtained with the MATLAB version of the code."""

    p = stamps_load(name)
    m = loadmat(name)

    def allclose(
        p: Array, m: Array, tol: float, equal_nan: bool, modulo: Optional[float]
    ) -> bool:
        diff = np.abs(p - m)
        if modulo is not None:
            mask = diff > tol
            diff[mask] = np.minimum(np.mod(diff[mask] + tol, modulo), diff[mask])
        if equal_nan:
            mask = np.logical_and(np.isnan(p), np.isnan(m))
            diff[mask] = 0
        return bool(np.all(diff <= tol))

    def print_values(p: Any, m: Any) -> None:
        print(f"Python:\n{p}")
        print(f"MATLAB:\n{m}")

    def compare_array(p: Array, m: Array) -> int:
        # print(f"{p.dtype = }\t{p.shape = }\tbyteorder = '{p.dtype.byteorder}'")
        # print(f"{m.dtype = }\t{m.shape = }\tbyteorder = '{m.dtype.byteorder}'")
        diff = np.abs(p - m)
        if modulo is not None:
            moddiff = np.mod(diff + tol, modulo)
            diff = np.minimum(moddiff, diff)
        mask = np.logical_and(np.isnan(p), np.isnan(m))
        diff[mask] = 0
        ndiff = np.nansum(diff > tol)
        print(f"Count of differences: {ndiff} ({ndiff / p.size * 100:.2f}%)")
        ix = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"Location of max difference: {ix}")
        print(f"Max difference: {diff[ix]}")
        if DEBUG and ndiff > 0:
            import pdb

            pdb.set_trace()
        return int(ndiff > 0)

    def compare_value(p: Any, m: Any, key: str) -> int:
        if isinstance(p, np.ndarray):
            if not allclose(p, m, tol=tol, equal_nan=equal_nan, modulo=modulo):
                print(f"\nError: `{key}` does not match at {tol = }")
                return compare_array(p, m)
            else:
                print(f"`{key}` matches with {tol = }")
        elif isinstance(p, (int, float, complex)):
            if np.abs(p - m) > tol:
                print(f"\nError: `{key}` does not match. {type(p) = } {type(m) = }")
                # print_values(p, m)
                return 1
            else:
                print(f"`{key}` matches with {tol = }")
        elif isinstance(p, str):
            if p != m:
                print(f"\nError: `{key}` does not match")
                # print_values(p, m)
                return 1
            else:
                print(f"`{key}` matches")
        else:
            print(f"Unsupported type: {type(p)}")
            return 0
        return 0

    # name  maxdiff  ndiff  type   size  byteorder

    ndiffs = 0
    if isinstance(p, np.ndarray):
        if isinstance(m, dict):
            m = m.get(name, m[list(m.keys())[0]])
        assert isinstance(m, np.ndarray)

        ndiffs += compare_value(p, m, name)

    elif isinstance(p, dict):
        for key in p:
            if any(exclude in key for exclude in ["ix", "ij", "loop", "sort", "bins"]):
                continue

            if key in m:
                ndiffs += compare_value(p[key], m[key], f"{name}.{key}")
            else:
                ndiffs += 1
    else:
        print(f"Unsupported type: {type(p)}")
        ndiffs += 1

    return ndiffs == 0


def check(
    name: str,
    x: dict | Array,
    tol: float = 1e-6,
    modulo: Optional[float] = None,
) -> None:
    """A debug function to check if the results match the results that have
    been saved in MATLAB."""

    # Disable this function if not in debug mode
    if not DEBUG:
        return

    if isinstance(x, dict):
        stamps_save(name, **x)
    else:
        stamps_save(name, x)

    assert results_equal(name, tol=tol, modulo=modulo)


def datenum(
    dates: Array[np.datetime64] | np.datetime64,
) -> Array[np.float64] | float:
    """Converts a list of dates to MATLAB datenum format."""

    if isinstance(dates, np.datetime64):
        dates = np.array([dates], dtype=np.datetime64)

    def fromtimestamp(t: np.floating) -> datetime:
        d = datetime.fromtimestamp(float(t), timezone.utc)
        return d.replace(tzinfo=None)

    def datetime64_to_datetime(dt64: np.datetime64) -> datetime:
        epoch = np.datetime64("1970-01-01")
        return fromtimestamp((dt64 - epoch) / np.timedelta64(1, "s"))

    def datetime_to_datenum(dt: datetime) -> float:
        mdn = dt + timedelta(days=366)
        frac = (dt - datetime(dt.year, dt.month, dt.day)).seconds / (24 * 3600)
        return mdn.toordinal() + frac

    dts = [datetime64_to_datetime(d) for d in dates]
    dns = np.array([datetime_to_datenum(d) for d in dts], dtype=np.float64)

    if len(dns) == 1:
        return float(dns[0])

    return dns


def datestr(
    datenums: int | float | list | np.ndarray,
) -> np.ndarray | np.datetime64:
    """Converts MATLAB datenum format to a np.datetime64 in the form YYYY-MM-DD."""

    if isinstance(datenums, (int, float)):
        datenums = [datenums]

    dns = np.array(datenums)

    def datenum_to_str(d: float) -> str:
        matlab_epoch = datetime(year=1, month=1, day=1)
        days_since_matlab_epoch = d - 367
        date = matlab_epoch + timedelta(days=days_since_matlab_epoch)
        return date.strftime("%Y-%m-%d")

    ds = np.array([datenum_to_str(d) for d in dns], dtype=np.datetime64)

    if len(ds) == 1:
        return np.datetime64(ds[0])

    return ds


def get_psver() -> int:
    """Retrieve the PS version from the 'psver' file."""
    with open("psver", "r") as f:
        return int(f.read().strip())


def set_psver(version: int) -> None:
    """Set the PS version in the 'psver' file."""
    with open("psver", "w") as f:
        f.write(str(version))


def stamps_save(
    fn: str, *args: Optional[Array | dict], **kwargs: Optional[Any]
) -> None:
    """Save a data file with the given name."""

    assert not fn.endswith(".mat")
    assert not fn.endswith(".pkl")

    if fn.endswith(".npz"):
        f = Path(fn)
    else:
        f = Path(f"{fn}.npz")

    if len(args) > 0 and isinstance(args[0], np.ndarray):
        np.savez(f, args[0])
    else:
        np.savez(f, **dotdict(kwargs))


def stamps_load(fn: str, squeeze: bool = True) -> dotdict | Array:
    """Load a data file with the given name."""

    assert not fn.endswith(".mat")
    assert not fn.endswith(".pkl")

    if fn.endswith(".npz"):
        f = Path(fn)
    else:
        f = Path(f"{fn}.npz")

    data = np.load(f, allow_pickle=True)  # FIXME: allow_pickle=False?

    assert hasattr(data, "files")

    if len(data.files) == 1:
        if squeeze:
            arr = data[data.files[0]]
            assert isinstance(arr, np.ndarray)
            return arr
        else:
            dn = "".join(x for x in fn if not x.isdigit())
            return dotdict({dn: data[data.files[0]]})

    dic = {}
    for k in data.files:
        if hasattr(data[k], "shape") and data[k].shape == ():
            dic[k] = data[k].item()
        else:
            dic[k] = data[k]
    return dotdict(dic)


def stamps_exists(fn: str) -> bool:
    """Check if a data file with the given name exists."""

    assert not fn.endswith(".mat")

    if fn.endswith(".npz"):
        f = Path(fn)
    else:
        f = Path(f"{fn}.npz")

    return f.exists()


def loadmat(fname: str) -> dotdict:
    """Loads a .mat file."""
    import scipy.io as sio

    if fname.endswith(".mat"):
        mat = sio.loadmat(fname, squeeze_me=True)
    else:
        mat = sio.loadmat(fname + ".mat", squeeze_me=True)

    kvs = {k: v for k, v in mat.items() if not k.startswith("__")}

    for k in kvs.keys():
        try:
            v = kvs[k]
            if isinstance(v, np.ndarray) and v.size == 1:
                v = v.flat[0]
        except IndexError:
            pass
        kvs[k] = v

    return dotdict(kvs)


def llh2local_alternate(llh: Array, origin: Array) -> Array:
    """
    Converts from longitude and latitude to local coordinates given an origin.
    llh (lon, lat, height) and origin should be in decimal degrees.
    Note that heights are ignored and that xy is in km.
    """

    import pymap3d as pm  # type: ignore

    xy = np.zeros((2, llh.shape[1]))
    for i, (lon, lat) in enumerate(llh.T):
        x, y, z = pm.geodetic2enu(lat, lon, 0, origin[1], origin[0], 0)
        xy[0, i] = x
        xy[1, i] = y
    return xy / 1000


def llh2local(llh: Array, origin: Array) -> Array:
    """
    Converts from longitude and latitude to local coordinates given an origin.
    llh (lon, lat, height) and origin should be in decimal degrees.
    Note that heights are ignored and that xy is in km.
    """

    # Set ellipsoid constants (WGS84)
    a = 6378137.0  # semi-major axis
    e = 0.08209443794970  # eccentricity

    # Convert to radians
    llh = np.deg2rad(llh)
    origin = np.deg2rad(origin)

    # Initialize xy array
    xy = np.zeros((2, llh.shape[1]))

    # Do the projection
    z = llh[1, :] != 0
    dlambda = llh[0, z] - origin[0]

    # calculates the meridional arc length M for each point and the origin M0, considering
    # the ellipsoidal shape of the Earth. This involves a series expansion that accounts
    # for the ellipsoid's eccentricity.

    M = a * (
        (1 - e**2 / 4 - 3 * e**4 / 64 - 5 * e**6 / 256) * llh[1, z]
        - (3 * e**2 / 8 + 3 * e**4 / 32 + 45 * e**6 / 1024) * np.sin(2 * llh[1, z])
        + (15 * e**4 / 256 + 45 * e**6 / 1024) * np.sin(4 * llh[1, z])
        - (35 * e**6 / 3072) * np.sin(6 * llh[1, z])
    )

    M0 = a * (
        (1 - e**2 / 4 - 3 * e**4 / 64 - 5 * e**6 / 256) * origin[1]
        - (3 * e**2 / 8 + 3 * e**4 / 32 + 45 * e**6 / 1024) * np.sin(2 * origin[1])
        + (15 * e**4 / 256 + 45 * e**6 / 1024) * np.sin(4 * origin[1])
        - (35 * e**6 / 3072) * np.sin(6 * origin[1])
    )

    N = a / np.sqrt(1 - e**2 * np.sin(llh[1, z]) ** 2)
    E = dlambda * np.sin(llh[1, z])

    xy[0, z] = N / np.tan(llh[1, z]) * np.sin(E)
    xy[1, z] = M - M0 + N / np.tan(llh[1, z]) * (1 - np.cos(E))

    # Handle special case of latitude = 0
    dlambda = llh[0, ~z] - origin[0]
    xy[0, ~z] = a * dlambda
    xy[1, ~z] = -M0

    # Convert to km
    xy = xy / 1000

    return xy


def gausswin(n: int, alpha: float = 2.5) -> Array[np.float64]:
    """Create a Gaussian window of length `n` with standard deviation `alpha`."""
    # n = np.arange(0, N) - (N - 1) / 2
    # w = np.array(np.exp(-(1 / 2) * (alpha * n / ((N - 1) / 2)) ** 2), dtype=np.float64)
    # return w
    std = (n - 1) / (2 * alpha)
    return np.array(gaussian(n, std), dtype=np.float64)


def interp(
    data: Array[np.floating], r: int, n: int = 4, cutoff: float = 0.5
) -> Array[np.floating]:
    """Resample `data` at a higher rate `r` using lowpass interpolation."""

    # Ensure data is a 2D array

    x = np.array(data, ndmin=2)
    if x.shape[0] == 1:
        x = x.T

    el = len(x)
    rl = r * el
    rn = r * n

    def design(r: int, n: int, alpha: float) -> Array[np.float64]:
        """Design a lowpass filter using the Parks-McClellan algorithm."""
        if alpha == 1:
            M = np.array([r] * 2 + [0] * 2)
            F = np.array([0, 1 / (2 * r), 1 / (2 * r), 0.5])
        else:
            nband = int(np.floor(0.5 * r))
            M = np.array([r] * 2 + [0.0] * (2 * nband))
            a2r = alpha / (2 * r)
            F = np.zeros(2 * nband + 2)
            k = np.arange(1, nband + 1)
            F[1] = a2r
            F[2::2] = k / r - a2r
            F[3::2] = k / r + a2r
            F[-1] = min(F[-1], 0.5)
        return np.array(firls(2 * r * n + 1, 2 * F, M), dtype=np.float64)

    # Design the lowpass filter

    b = design(r, n, cutoff)

    # Apply the lowpass filter

    y = np.zeros((rl, 1))
    y[::r] = x

    od = np.zeros((2 * rn, 1))
    od[::r] = 2 * x[0] - x[1 : 2 * n + 1][::-1]
    _, zi = lfilter(b, 1, od, axis=0, zi=np.ones((len(b) - 1, 1)))
    y, zf = lfilter(b, 1, y.T, zi=zi.T)
    y[:, : (el - n) * r] = y[:, rn:rl]

    od = np.zeros((2 * rn, 1))
    od[::r] = 2 * x[-1] - x[-2 : -2 * n - 2 : -1]
    od, _ = lfilter(b, 1, od.T, zi=zf)
    y[:, rl - rn : rl] = od[:, :rn]

    return y.ravel()


def getparm(parmname: Optional[str] = None, verbose: bool = False) -> str:
    """Retrieves a parameter value from parameter files."""

    if parmname in OPTIONS:
        return str(OPTIONS[parmname])
    else:
        log(f"Parameter {parmname} not found in OPTIONS")

    # TODO: Remove the following old code:

    def pprint(k: str, v: Any) -> None:
        if isinstance(v, str):
            log(f"{k} = '{v}'")
        else:
            log(f"{k} = {v}")

    # Load global parameters

    parmfile = Path("parms.mat").resolve()
    if parmfile.exists():
        parms = loadmat(str(parmfile))
    elif (parmfile.parent.parent / parmfile.name).exists():
        parmfile = parmfile.parent.parent / parmfile.name
        parms = loadmat(str(parmfile))
    else:
        raise FileNotFoundError(f"`{parmfile}` not found")

    # Load local parameters, if available

    localparmfile = Path("localparms.mat")
    if localparmfile.exists():
        localparms = loadmat(str(localparmfile))
    else:
        # Placeholder for creation date
        localparms = dotdict({"Created": "today"})

    # Retrieve the parameter value

    value = None
    if parmname is None:
        for k, v in parms.items():
            pprint(k, v)
        if len(localparms) > 1:
            log(str(localparms))
    else:
        # Find the parameter in global or local parameters
        parmnames = [k for k in parms if not k.startswith("__")]
        matches = [pn for pn in parmnames if pn.startswith(parmname)]
        if len(matches) > 1:
            raise ValueError(f"Parameter {parmname}* is not unique")
        elif not matches:
            return ""
        else:
            parmname = matches[0]
            value = localparms.get(parmname, parms.get(parmname))

        if verbose and parmname is not None:
            pprint(parmname, value)

    return str(value)


def setparm(parmname: str, value: Any) -> None:
    """Sets a parameter value in the local parameters file."""

    OPTIONS[parmname] = value

    return

    # FIXME: Write to disk as toml?

    import scipy.io as sio

    parmfile = Path("parms.mat").absolute()
    if parmfile.exists():
        parms = loadmat(str(parmfile))
    elif (parmfile.parent.parent / parmfile.name).exists():
        parmfile = parmfile.parent.parent / parmfile.name
        parms = loadmat(str(parmfile))
    else:
        raise FileNotFoundError(f"`{parmfile}` not found")

    parms[parmname] = value
    sio.savemat(str(parmfile), parms)


def read_param(fname: Path, parm: str) -> str:
    """Reads a single parameter value from a GAMMA parameter file.

    This function returns a string and we let the user convert it to the
    appropriate type. This is more type-safe.
    """

    with fname.open("r") as f:
        lines = f.readlines()

    result = ""
    for line in lines:
        if line.startswith(parm):
            result = line.split()[1]

    return result


def read_params(fname: Path, parm: str, numval: int) -> List[str]:
    """Reads multiple parameter values from a GAMMA parameter file.

    This function returns a list of strings and we let the user convert it to
    the appropriate type. This is more type-safe.
    """

    with fname.open("r") as f:
        lines = f.readlines()

    values = []
    for line in lines:
        if line.startswith(parm):
            values = [x for x in line.split()[1:]]
            values = values[:numval]
            break

    return values


def patchdirs() -> List[Path]:
    """Get the patch directories."""

    dirs = []

    patchlist = Path("patch.list")
    if patchlist.exists():
        log(f"Reading directory names from `{patchlist}`")
        with patchlist.open("r") as f:
            dirs = [Path(line.strip()).resolve() for line in f.readlines()]
            log(f"Found {len(dirs)} directories in `patch.list`: {dirs}")
    else:
        # if patch.list does not exist, find all directories with PATCH_ in the name
        dirs = [d for d in Path(".").iterdir() if d.is_dir() and "PATCH_" in d.name]

    if len(dirs) == 0:
        # patch directories not found, use current directory
        dirs.append(Path("."))

    return dirs


def clap_filter(
    ph_in: Array[np.complexfloating],
    alpha: float = 0.5,
    beta: float = 0.1,
    n_win: int = 32,
    n_pad: int = 0,
    low_pass: Optional[Array] = None,
) -> Array[np.complexfloating]:
    """
    Combined Low-pass Adaptive Phase (CLAP) filtering.
    """

    # To be safe, make a copy of the input phase array
    ph = ph_in.copy()

    if low_pass is None:
        low_pass = np.zeros((n_win + n_pad, n_win + n_pad))

    ph_out = np.zeros_like(ph, dtype=ph.dtype)
    n_i, n_j = ph.shape

    n_inc = n_win // 4
    n_win_i = -(-n_i // n_inc) - 3  # Ceiling division
    n_win_j = -(-n_j // n_inc) - 3

    # Create a window function
    x = np.arange(n_win // 2)
    wind_func = np.pad(
        np.add.outer(x, x), ((0, n_win // 2), (0, n_win // 2)), mode="symmetric"
    )

    def adjust_window(wf: Array, shift: int, axis: int) -> Array:
        if axis == 0:  # Adjust rows
            wf[:shift, :] = 0
        else:  # Adjust columns
            wf[:, :shift] = 0
        return wf

    # Replace NaNs with zeros
    ph = np.nan_to_num(ph)

    # Gaussian smoothing kernel
    B = np.outer(gausswin(7, 2.5), gausswin(7, 2.5))

    ph_bit = np.zeros((n_win + n_pad, n_win + n_pad), dtype=ph.dtype)

    for ix1 in range(n_win_i):
        i1 = ix1 * n_inc
        i2 = min(i1 + n_win, n_i + 1)
        wf = wind_func.copy()

        # Adjust the window function for the edge cases
        if i2 > n_i:
            i_shift = i2 - n_i
            i2 = n_i
            i1 = n_i - n_win
            wf = np.vstack((np.zeros((i_shift, n_win)), wf[: n_win - i_shift, :]))

        for ix2 in range(n_win_j):
            j1 = ix2 * n_inc
            j2 = min(j1 + n_win, n_j + 1)
            wf2 = wf.copy()

            # Adjust the window function for the edge cases
            if j2 > n_j:
                j_shift = j2 - n_j
                j2 = n_j
                j1 = n_j - n_win
                wf2 = np.concatenate(
                    (np.zeros((n_win, j_shift)), wf2[:, : n_win - j_shift]),
                    axis=1,
                )

            ph_bit[: i2 - i1, : j2 - j1] = ph[i1:i2, j1:j2]

            # Smooth the magnitude response
            ph_fft = np.fft.fft2(ph_bit)
            H = np.abs(ph_fft)

            H = ifftshift(convolve2d(fftshift(H), B, mode="same"))

            medianH = np.median(H)

            if medianH == 0:
                medianH = 1
            H = (H / medianH) ** alpha

            H = H - 1
            H[H < 0] = 0

            G = H * beta + low_pass

            ph_filt = np.fft.ifft2(ph_fft * G)

            ph_out[i1:i2, j1:j2] = (
                ph_out[i1:i2, j1:j2] + ph_filt[: i2 - i1, : j2 - j1] * wf2
            )

    return ph_out


def clap_filter_patch(
    ph_in: Array[np.complexfloating],
    alpha: float = 0.5,
    beta: float = 0.1,
    low_pass: Optional[Array] = None,
) -> Array[np.complexfloating]:
    """
    Combined Low-pass Adaptive Phase filtering on 1 patch.
    """

    # To be safe, make a copy of the input phase array
    ph = ph_in.copy()

    if low_pass is None:
        low_pass = np.zeros_like(ph)

    # Replace NaNs with 0
    ph = np.nan_to_num(ph)

    # Create Gaussian window
    wl = 7  # Window length
    wf = 2.5  # Width factor
    std = (wl - 1) / (2 * wf)
    gauss_win = gaussian(wl, std)
    B = np.outer(gauss_win, gauss_win)

    # Compute FFT of the patch
    ph_fft = np.fft.fft2(ph)

    # Compute magnitude response and smooth it
    H = np.abs(ph_fft)
    H = fftshift(fftconvolve(ifftshift(H), B, mode="same"))

    # Normalize and apply power law
    medianH = np.median(H)
    if medianH == 0:
        medianH = 1
    H = (H / medianH) ** alpha
    H = H - 1
    H[H < 0] = 0

    # Combine with low_pass using adaptive factor
    G = H * beta + low_pass

    # Inverse FFT to get filtered patch
    ph_out = np.fft.ifft2(ph_fft * G)

    return ph_out


def goldstein_filter(
    ph_in: Array[np.complexfloating],
    n_win: int,
    alpha: float,
    n_pad: Optional[int] = None,
) -> Array[np.complexfloating]:
    """Goldstein's adaptive phase filtering applied to a phase image."""

    # Make a copy of the input phase array
    ph = ph_in.copy()

    if n_pad is None:
        n_pad = int(round(n_win * 0.25))

    # Create the Gaussian smoothing kernel
    gauss_kernel = np.outer(gaussian(7, 7 / 3), gaussian(7, 7 / 3))

    # Initialize the output phase array
    ph_out = np.zeros_like(ph)

    # Replace NaNs with zeros
    ph = np.nan_to_num(ph)

    # Compute increments and the number of windows
    n_inc = n_win // 4
    n_win_i, n_win_j = (np.ceil(np.array(ph.shape) / n_inc) - 3).astype(int)

    # Generate the window function using a vectorized approach
    x = np.arange(1, n_win // 2 + 1)
    wind_func = np.pad(x[:, None] + x, ((0, n_win // 2), (0, n_win // 2)), "symmetric")

    def win_bounds(
        ix: int, n_inc: int, n_win: int, max_bound: int, wind_func: Array
    ) -> Tuple[int, int, Array]:
        i1 = ix * n_inc
        i2 = min(i1 + n_win, max_bound)
        wf = wind_func[
            : i2 - i1, : i2 - i1
        ]  # Adjust the window function for window size
        return i1, i2, wf

    # Loop over each window
    for ix1 in range(n_win_i):
        for ix2 in range(n_win_j):
            # Calculate window bounds
            i1, i2, wf_i = win_bounds(ix1, n_inc, n_win, ph.shape[0], wind_func)
            j1, j2, wf_j = win_bounds(ix2, n_inc, n_win, ph.shape[1], wind_func)

            # Extract the current window and apply FFT
            ph_bit = np.zeros((n_win + n_pad, n_win + n_pad))
            ph_bit[: i2 - i1, : j2 - j1] = ph[i1:i2, j1:j2]
            # NOTE: changed scipy.fft.ifft2 to numpy.fft.ifft2
            ph_fft = np.fft.fft2(ph_bit)

            # Apply the adaptive filter in the frequency domain
            H = ifftshift(
                convolve2d(fftshift(np.abs(ph_fft)), gauss_kernel, mode="same")
            )
            meanH = np.median(H)
            H = ((H / meanH) ** alpha if meanH != 0 else H**alpha) * (
                n_win + n_pad
            ) ** 2

            # Inverse FFT and update the output array
            # NOTE: changed scipy.fft.ifft2 to numpy.fft.ifft2
            ph_filt = np.fft.ifft2(ph_fft * H).real[: i2 - i1, : j2 - j1] * (
                wf_i[:, None] * wf_j
            )
            ph_out[i1:i2, j1:j2] += ph_filt

    return ph_out


def gradient_filter(
    ph: np.ndarray, n_win: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Determine 2-D gradient through FFT over windows of size n_win.
    """

    raise NotImplementedError("This function has not been verified yet.")

    # Initialize variables
    n_i, n_j = ph.shape
    n_inc = n_win // 4
    n_win_i = -(-n_i // n_inc) - 3  # Ceiling division
    n_win_j = -(-n_j // n_inc) - 3

    # Replace NaNs with zeros
    ph = np.nan_to_num(ph)

    # Initialize output arrays
    Hmag = np.full((n_win_i, n_win_j), np.nan)
    ifreq = Hmag.copy()
    jfreq = Hmag.copy()
    ij = np.full((n_win_i * n_win_j, 2), np.nan)

    def calc_bounds(
        ix: int, n_inc: int, n_win: int, max_dim: int
    ) -> Tuple[int, int, int, int]:
        i1 = ix * n_inc
        i2 = min(i1 + n_win, max_dim)
        return max(i2 - n_win, 0), i2

    def calc_freq(I1: Array, I2: Array, n_win: int) -> Tuple[Array, Array]:
        I1 = (I1 + n_win // 2) % n_win
        I2 = (I2 + n_win // 2) % n_win
        return (
            (I1 - n_win // 2) * 2 * np.pi / n_win,
            (I2 - n_win // 2) * -2 * np.pi / n_win,
        )

    idx = 0
    for ix1 in range(n_win_i):
        for ix2 in range(n_win_j):
            # Calculate window bounds
            i1, i2, j1, j2 = calc_bounds(ix1, ix2, n_inc, n_win, n_i, n_j)

            # Extract phase window and apply FFT
            ph_bit = ph[i1:i2, j1:j2]
            if np.count_nonzero(ph_bit) < 6:  # Check for enough non-zero points
                continue

            ph_fft = np.fft.fft2(ph_bit)
            H = np.abs(ph_fft)

            # Find the index of the maximum magnitude
            hidx = np.argmax(H)
            Hmag_this = H.flat[hidx] / H.mean()

            # Calculate frequencies
            hidx1, hidx2 = np.unravel_index(hidx, (n_win, n_win))
            ifreq_val, jfreq_val = calc_freq(hidx1, hidx2, n_win)

            # Update output arrays
            Hmag[ix1, ix2] = Hmag_this
            ifreq[ix1, ix2] = ifreq_val
            jfreq[ix1, ix2] = jfreq_val
            ij[idx] = [(i1 + i2) / 2, (j1 + j2) / 2]
            idx += 1

    return ifreq.T, jfreq.T, ij, Hmag.T


def topofit(
    cpxphase: Array, bperp: Array, n_trial_wraps: float, asym: int = 0
) -> Tuple[float, float, float, Array]:
    """
    Finds the best-fitting range error for complex phase observations.

    Parameters:
    cpxphase : 1D numpy array of complex phase observations.
    bperp : 1D numpy array of perpendicular baseline values corresponding to cpxphase.
    n_trial_wraps : float, the number of trial wraps to consider.
    asym : int, controls the search range for K; -1 for only negative, +1 for only positive, 0 for both.

    Returns:
    K0 : float, estimated range error.
    C0 : float, estimated phase offset.
    coh0 : float, coherence of the fit.
    phase_residual : 1D numpy array, residual phase after removing the fitted model.
    """

    # Ensure cpxphase is a 1D array
    cpxphase = cpxphase.flatten()

    # Filter out zero phase observations
    ix = cpxphase != 0
    cpxphase = cpxphase[ix]
    bperp = bperp[ix]

    # Calculate bperp range
    bperp_range = np.max(bperp) - np.min(bperp)

    # Define trial multipliers for range error search
    trial_mult = (
        np.arange(
            -int(np.ceil(8 * n_trial_wraps)),
            int(np.ceil(8 * n_trial_wraps)) + 1,
        )
        + asym * 8 * n_trial_wraps
    )

    # n_trials = len(trial_mult)

    # Calculate trial phase and trial phase matrix for fitting
    trial_phase = bperp / bperp_range * np.pi / 4
    trial_phase_mat = np.exp(-1j * np.outer(trial_phase, trial_mult))

    # Compute phase responses for each trial
    phaser = trial_phase_mat * cpxphase[:, None]
    phaser_sum = np.sum(phaser, axis=0)

    # Calculate trial coherences and offsets
    C_trial = np.angle(phaser_sum)
    coh_trial = np.abs(phaser_sum) / np.sum(np.abs(cpxphase))

    # Find the trial with the highest coherence
    coh_high_max_ix = np.argmax(coh_trial)

    # Estimate range error, phase offset, and coherence
    K0 = np.pi / 4 / bperp_range * trial_mult[coh_high_max_ix]
    C0 = C_trial[coh_high_max_ix]
    coh0 = coh_trial[coh_high_max_ix]

    # Linearise and solve for residual phase
    resphase = cpxphase * np.exp(-1j * (K0 * bperp))
    offset_phase = np.sum(resphase)
    resphase = np.angle(resphase * np.conj(offset_phase))

    # Weighted least squares fit for residual phase
    weighting = np.abs(cpxphase)
    mopt = np.linalg.lstsq(
        weighting[:, None] * bperp[:, None], weighting * resphase, rcond=None
    )[0]
    K0 += mopt[0]

    # Calculate phase residuals
    phase_residual = cpxphase * np.exp(-1j * (K0 * bperp))
    mean_phase_residual = np.sum(phase_residual)
    C0 = np.angle(mean_phase_residual)  # Updated static offset
    coh0 = np.abs(mean_phase_residual) / np.sum(
        np.abs(phase_residual)
    )  # Updated coherence

    return K0, C0, coh0, phase_residual


def stage0_preprocess(opts: dotdict = dotdict()) -> None:
    """Preprocess the data for the first stage of the InSAR processing. This
    includes the identification of candidate PS pixels."""

    log("# Stage 0: Identifying candidate PS pixels and preprocessing data")

    # Magically identify the processor class

    try:
        processor_class = {
            cls.__name__.replace("PrepareData", "").lower(): cls
            for cls in PrepareData.__subclasses__()
        }[PROCESSOR]
    except KeyError:
        raise RuntimeError(f"Processor `{PROCESSOR}` not found")

    if len(opts.master_date) == 0:
        # Find the master date from the intereferogram base files
        first_basefile = next((opts.datadir / "diff0").glob("*.base"), None)
        if first_basefile:
            master_date = first_basefile.stem[:8]
            log(
                f"Master date automatically set to {master_date} based on data in `{first_basefile.parent}`"
            )
        else:
            raise RuntimeError("Master date not found")

    # Instantiate and run the processor

    processor_class(
        master_date,
        opts.datadir,
        opts.da_thresh,
        opts.rg_patches,
        opts.az_patches,
        opts.rg_overlap,
        opts.az_overlap,
        opts.maskfile,
    ).run()


def stage1_load_data(endian: str = "b", opts: dotdict = dotdict()) -> None:
    """Load all the data we need from GAMMA outputs, process it, and
    save it into our own data format."""

    log("# Stage 1: Load initial data from GAMMA outputs")

    # File names assume we are in a PATCH_ directory
    assert Path(".").resolve().name.startswith("PATCH_")

    phname = Path("./pscands.1.ph")  # phase data
    ijname = Path("./pscands.1.ij")  # pixel location data
    llname = Path("./pscands.1.ll")  # latitude, longitude data
    # xyname = Path("./pscands.1.xy")  # local coordinates
    hgtname = Path("./pscands.1.hgt")  # height data
    daname = Path("./pscands.1.da")  # dispersion data
    rscname = Path("../rsc.txt")  # config with master rslc.par file location
    pscname = Path("../pscphase.in")  # config with width and diff phase file locataions

    # Read master day from rsc file
    with rscname.open() as f:
        rslcpar = Path(f.readline().strip())

    log(f"{rslcpar = }")

    log(f"Reading inteferogram dates from `{pscname.resolve()}`")

    # Read interferogram dates
    with pscname.open() as f:
        f.readline()  # skip first line
        ifgs = sorted([Path(line.strip()) for line in f.readlines()])

    print(f"{ifgs = } {len(ifgs) = }")

    datestr = f"{rslcpar.name[0:4]}-{rslcpar.name[4:6]}-{rslcpar.name[6:8]}"
    master_day = datenum(np.datetime64(datestr))
    log(f"{master_day = } ({datestr})")

    ifgdts = np.array(
        [f"{ifg.name[9:13]}-{ifg.name[13:15]}-{ifg.name[15:17]}" for ifg in ifgs],
        dtype="datetime64",
    )
    day = np.array(datenum(ifgdts), dtype=np.float64)

    n_image = len(day)
    n_ifg = len(ifgs)

    master_ix = np.sum(day < master_day)
    if day[master_ix] != master_day:
        log(f"Master {rslcpar.name[:8]} not found, inserting at index {master_ix}")
        day = np.insert(day, master_ix, master_day)

    log(f"{day = }")
    log(f"{master_ix = }")
    log(f"{n_ifg = }")

    # Set and save heading parameter
    heading = float(read_param(rslcpar, "heading"))
    setparm("heading", heading)

    freq = float(read_param(rslcpar, "radar_frequency"))
    lam = 299792458 / freq
    setparm("lambda", lam)

    sensor = read_param(rslcpar, "sensor")
    platform = sensor  # S1 case
    setparm("platform", platform)

    rps = float(read_param(rslcpar, "range_pixel_spacing"))
    rgn = float(read_param(rslcpar, "near_range_slc"))
    se = float(read_param(rslcpar, "sar_to_earth_center"))
    re = float(read_param(rslcpar, "earth_radius_below_sensor"))
    rgc = float(read_param(rslcpar, "center_range_slc"))
    naz = int(read_param(rslcpar, "azimuth_lines"))
    prf = float(read_param(rslcpar, "prf"))

    mean_az = naz / 2.0 - 0.5

    # Processing of the id, azimuth, range data
    with ijname.open("rb") as f:
        ij = np.loadtxt(f, converters=int).astype(np.uint16)

    stamps_save("ij", ij)

    n_ps = len(ij)

    # Processing of the longitude and latitude data
    with llname.open("rb") as f:
        lonlat = np.fromfile(f, dtype=">f4").reshape((-1, 2)).astype(np.float64)

    # Processing of the Height data
    if hgtname.exists():
        with hgtname.open("rb") as f:
            hgt = np.fromfile(f, dtype=np.float64)
    else:
        log(f"{hgtname} does not exist, proceeding without height data.")

    # Calculate range
    rg = rgn + ij[:, 2] * rps

    # Calculate satellite look angles
    look = np.arccos((se**2 + rg**2 - re**2) / (2 * se * rg))

    # Initialize the perpendicular baseline matrix
    bperp_mat = np.zeros((n_ps, n_ifg))
    for i in range(n_ifg):
        basename = ifgs[i].with_suffix(".base")
        B_TCN = np.array(
            [float(x) for x in read_params(basename, "initial_baseline(TCN)", 3)]
        )
        BR_TCN = np.array(
            [float(x) for x in read_params(basename, "initial_baseline_rate", 3)]
        )
        bc = B_TCN[1] + BR_TCN[1] * (ij[:, 1] - mean_az) / prf
        bn = B_TCN[2] + BR_TCN[2] * (ij[:, 1] - mean_az) / prf
        # Convert baselines from (T)CN to perpendicular-parallel coordinates
        bperp_mat[:, i] = bc * np.cos(look) - bn * np.sin(look)

    # Calculate mean perpendicular baselines
    bperp = np.mean(bperp_mat, axis=0)

    log("Mean perpendicular baseline for each interferogram:")
    for i in range(n_ifg):
        log(f"{ifgs[i]}\tmean(bperp) = {bperp[i]:+.3f}")

    # Calculate incidence angles
    inci = np.arccos((se**2 - re**2 - rg**2) / (2 * re * rg))

    # Calculate mean incidence angle
    mean_incidence = np.mean(inci)

    # Mean range is given by the center range distance
    mean_range = rgc

    # Processing of the phase data
    with phname.open("rb") as f:
        log(f"Loading phase time series data from `{phname.resolve()}`")
        try:
            ph = np.fromfile(f, dtype=">c8").reshape((n_ifg, n_ps)).T
        except ValueError:
            f_n_ifg, f_n_ps = filedim(phname, n_ps, ">c8")
            raise RuntimeError(
                f"Wrong dimensions for `{phname}`. File shape {f_n_ps}x{f_n_ifg} and expected is {n_ps}x{n_ifg}"
            )

    # Calculate mean phases
    mu = np.mean(ph, axis=0)
    for i in range(n_ifg):
        log(f"{ifgs[i]}\tmean(phase) = {mu[i]:+.3f}")
    log(f"{ph.shape = }")

    log(f"{bperp.shape = }")

    # Inserting a column of 1's for master image
    ph = np.insert(ph, master_ix, np.ones(n_ps), axis=1)
    bperp = np.insert(bperp, master_ix, 0)
    n_ifg += 1
    n_image += 1

    log(f"{bperp.shape = }")

    # Find center longitude and latitude
    ll0 = (np.max(lonlat, axis=0) + np.min(lonlat, axis=0)) / 2

    log(f"{ll0 = } (center longitude and latitude in degrees)")

    # Convert to local coordinates and scale to meters
    xy = llh2local(lonlat.T, ll0).T * 1000

    # Sort coordinates by x and y
    sort_x = xy[np.argsort(xy[:, 0])]
    sort_y = xy[np.argsort(xy[:, 1])]

    # Determine corners based on a small percentage of points
    n_pc = int(np.round(n_ps * 0.001))
    bl = np.mean(sort_x[:n_pc], axis=0)  # bottom left
    tr = np.mean(sort_x[-n_pc:], axis=0)  # top right
    br = np.mean(sort_y[:n_pc], axis=0)  # bottom right
    tl = np.mean(sort_y[-n_pc:], axis=0)  # top left

    log(f"{bl = }\n{tl = }\n{br = }\n{tr = } (patch corners in meters)")

    # Calculate rotation angle
    theta = (180 - heading) * np.pi / 180
    if theta > np.pi:
        theta -= 2 * np.pi

    log(f"{theta = } (rotation angle in radians)")

    # Rotation matrix
    rotm = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

    log("Rotation matrix:")
    np.savetxt(sys.stderr, rotm, fmt="%.2f", delimiter=" ")

    # Rotate coordinates
    xy = xy.T
    xynew = rotm @ xy

    # Check if rotation improves alignment and apply if it does
    if (np.max(xynew[0]) - np.min(xynew[0]) < np.max(xy[0]) - np.min(xy[0])) and (
        np.max(xynew[1]) - np.min(xynew[1]) < np.max(xy[1]) - np.min(xy[1])
    ):
        log(
            f"Rotation improved alignment, applying rotation {theta * 180 / np.pi:.2f}°"
        )
        xy = xynew.T

    # Sort local coords `xy` in ascending y, then x order
    sort_ix = np.lexsort((xy[:, 0], xy[:, 1]))

    stamps_save("sort_ix", sort_ix)

    # Sort all data based on the sorted indices
    xy = xy[sort_ix]
    ph = ph[sort_ix]
    lonlat = lonlat[sort_ix]
    bperp_mat = bperp_mat[sort_ix, :]
    la = inci[sort_ix]
    ij = ij[sort_ix]

    # As `ij` is now sorted, we update the point ids (1 to n_ps)
    ij[:, 0] = np.arange(1, n_ps + 1)

    # Round local coords `xy` to nearest mm
    xy = np.insert(xy, 0, np.arange(1, n_ps + 1), axis=1)
    xy[:, 1:] = np.round(xy[:, 1:] * 1000) / 1000

    # Load the dispersion (fano factor) for each candidate PS and sort it
    with daname.open("rb") as f:
        D_A = np.loadtxt(f)
    D_A = D_A[sort_ix]

    # Load the height data and sort it
    with hgtname.open("rb") as f:
        hgt = np.loadtxt(f, usecols=2)
    hgt = hgt[sort_ix]

    set_psver(1)
    psver = get_psver()

    # Save all the data in our own format

    stamps_save(
        f"ps{psver}",
        ij=ij,
        lonlat=lonlat,
        xy=xy,
        bperp=bperp,
        day=day,
        master_day=master_day,
        master_ix=master_ix,
        n_ifg=n_ifg,
        n_image=n_image,
        n_ps=n_ps,
        sort_ix=sort_ix,
        ll0=ll0,
        mean_incidence=mean_incidence,
        mean_range=mean_range,
    )

    stamps_save(f"ph{psver}", ph)  # phase data (n_ps, n_ifg)  - complex
    stamps_save(
        f"bp{psver}", bperp_mat
    )  # perpendicular baselines (n_ps, n_ifg) - meters
    stamps_save(f"la{psver}", la)  # incidence angles (n_ps,) - radians
    stamps_save(f"da{psver}", D_A)  # dispersion (fano factor) (n_ps,) - unitless
    stamps_save(f"hgt{psver}", hgt)  # height data (n_ps,) - meters


def stage2_estimate_noise(max_iters: int = 1000, opts: dotdict = dotdict()) -> None:
    """For each persistent scatterer candidate, estimate the noise in the phase data.

    This is an iterative process that uses the Goldstein adaptive phase filtering
    to estimate the noise in the phase data for each candidate. The process is
    repeated until the noise estimate converges or the maximum number of iterations
    is reached (`max_iters`).
    """

    log("# Stage 2: Estimate noise for each persistent scatterer candidate")

    # Load data
    ps = stamps_load("ps1")
    ph = stamps_load("ph1")
    la = stamps_load("la1")
    da = stamps_load("da1")
    bperp_mat = stamps_load("bp1")

    assert isinstance(ps, dotdict)
    assert isinstance(ph, np.ndarray)
    assert isinstance(la, np.ndarray)
    assert isinstance(da, np.ndarray)
    assert isinstance(bperp_mat, np.ndarray)

    bperp = ps["bperp"]
    n_ifg = ps["n_ifg"]
    n_image = ps["n_image"]
    n_ps = ps["n_ps"]
    xy = ps["xy"]
    master_ix = ps["master_ix"]

    # StaMPS hardcoded value (approx 3 deg)
    inc_mean = ps["mean_incidence"] + 0.052
    # FIXME: This might be better?
    # inc_mean = ps.mean_incidence + np.deg2rad(3)

    # CLAP filter parameters

    grid_size = int(getparm("filter_grid_size"))
    filter_weighting = getparm("filter_weighting")  # string
    n_win = int(getparm("clap_win"))
    low_pass_wavelength = float(getparm("clap_low_pass_wavelength"))
    clap_alpha = float(getparm("clap_alpha"))
    clap_beta = float(getparm("clap_beta"))

    log(f"{grid_size = } (filter grid size)")
    log(f"{filter_weighting = } (filter weighting)")
    log(f"{n_win = } (CLAP filter window size)")
    log(f"{low_pass_wavelength = } (low-pass wavelength in meters)")
    log(f"{clap_alpha = } (CLAP filter alpha)")
    log(f"{clap_beta = } (CLAP filter beta)")

    # Parameters for maximum baseline length (max_K) calculation

    max_topo_err = float(getparm("max_topo_err"))
    lambda_ = float(getparm("lambda"))

    log(f"{max_topo_err = } (maximum topographic error in meters)")
    log(f"{lambda_ = } (wavelength in meters)")

    gamma_change_convergence = float(getparm("gamma_change_convergence"))
    gamma_max_iterations = int(getparm("gamma_max_iterations"))
    small_baseline_flag = getparm("small_baseline_flag")  # string

    log(f"{gamma_change_convergence = } (convergence threshold)")
    log(f"{gamma_max_iterations = } (maximum iterations)")
    log(f"{small_baseline_flag = } (small baseline flag)")

    rho = 830000  # mean range - need only be approximately correct
    n_rand = 300000  # number of simulated random phase pixels
    low_coh_thresh = 31  # equivalent to 15/100 in GAMMA

    # Construct a two-dimensional low-pass filter using a
    # Butterworth filter design in the frequency domain used
    # to attenuate high-frequency components in the observations,
    # effectively smoothing them or reducing noise

    freq0 = 1 / low_pass_wavelength
    freq_i = np.arange(
        -(n_win) / grid_size / n_win / 2,
        (n_win - 1) / grid_size / n_win / 2,
        1 / grid_size / n_win,
    )
    butter_i = 1 / (1 + (freq_i / freq0) ** (2 * 5))
    low_pass = np.outer(butter_i, butter_i)
    low_pass = np.fft.fftshift(low_pass)

    # check("low_pass", low_pass)

    null_i, null_j = np.where(ph == 0)
    null_i = np.unique(null_i)
    good_ix = np.ones(n_ps, dtype=bool)
    good_ix[null_i] = False

    if small_baseline_flag.lower() == "n":
        keep_ix = list(range(master_ix)) + list(range(master_ix + 1, n_ifg))
        ph = ph[:, keep_ix]
        bperp = bperp[keep_ix]
        n_ifg = n_ifg - 1
        n_image = n_image - 1

    # Normalizes the complex phase values to have unit magnitude
    A = np.abs(ph)
    A[A == 0] = 1  # Avoid divide by zero
    ph /= A

    # check("nph", ph, atol=1e-6, rtol=1e-6)

    # Calculate the maximum baseline length (max_K) that can be tolerated for a
    # given topographic error (max_topo_err) considering the wavelength
    # (lambda_) of the radar, the spatial baseline
    # decorrelation coefficient (rho), and the mean incidence angle (inc_mean) of radar signal

    max_K = max_topo_err / (lambda_ * rho * np.sin(inc_mean) / (4 * np.pi))
    log(f"{max_K = } (maximum baseline length tolerated; in radians)")

    bperp_range = np.max(bperp) - np.min(bperp)
    n_trial_wraps = bperp_range * max_K / (2 * np.pi)
    log(f"{n_trial_wraps = } (number of trial wraps)")

    # check(
    #    "ec1",
    #    {
    #        "max_K": max_K,
    #        "max_topo_err": max_topo_err,
    #        "lambda": lambda_,
    #        "rho": rho,
    #        "inc_mean": inc_mean,
    #        "bperp_range": bperp_range,
    #        "n_trial_wraps": n_trial_wraps,
    #    },
    # )

    log("Generating random interferograms")

    if small_baseline_flag.lower() == "y":
        ifgday_ix = ps["ifgday_ix"]

        # Generate random phase values for each pixel in each image, scaled by 2*pi
        # This simulates the random phase component in each SAR image
        rand_image = 2 * np.pi * np.random.rand(n_rand, n_image)

        rand_ifg = np.zeros((n_rand, n_ifg))

        for i in range(n_ifg):
            # Calculate the random phase difference for the interferogram
            # This is done by subtracting the random phase of one image from the other,
            # based on the unique day indices for each interferogram (ifgday_ix)
            # This simulates the random phase component in each interferogram
            rand_ifg[:, i] = (
                rand_image[:, ifgday_ix[i, 1]] - rand_image[:, ifgday_ix[i, 0]]
            )
    else:
        # If not using small baseline flag, generate random interferogram phase values directly
        # This simulates random phase components in interferograms without considering individual images
        rand_ifg = 2 * np.pi * np.random.rand(n_rand, n_ifg)

    # DEBUG: Load pre-generated random interferograms from file
    # rand_ifg = loadmat("rand_ifg.mat")["rand_ifg"]
    # check("rand_ifg", rand_ifg)

    # Pre-compute complex exponential of random interferograms to avoid
    # recalculating in each iteration
    exp_rand_ifg = np.exp(1j * rand_ifg)

    log(f"Fitting topographic phase models to {n_rand:,} random interferogram")

    # Iterate through random phase points in reverse order
    coh_rand = np.zeros(n_rand)
    for i in reversed(range(n_rand)):
        # Fit a topographic phase model to each random phase point
        K_r, C_r, coh_r, res_r = topofit(exp_rand_ifg[i, :], bperp, n_trial_wraps)
        show_progress(n_rand - i, n_rand)

        # Store the first coherence value for each random point
        coh_rand[i] = coh_r

    # check("coh_rand", coh_rand)

    coh_bins = np.arange(0.0, 1.01, 0.01)  # old matlab hist uses bin centers

    log(f"Generating histogram of {n_rand:,} coherences using {len(coh_bins)} bins")

    Nr, _ = np.histogram(coh_rand, bins=coh_bins)
    Nr = Nr.astype(np.float64)  # Fix type - StaMPS error

    # check("Nr", Nr)

    # Find the last non-zero bin index using np.max and np.nonzero
    Nr_max_nz_ix = np.max(np.nonzero(Nr)[0])

    K_ps, C_ps, coh_ps, coh_ps_save, N_opt = (np.zeros(n_ps) for _ in range(5))
    ph_res = np.zeros((n_ps, n_ifg), dtype=np.float32)
    ph_patch = np.zeros_like(ph, dtype=ph.dtype)

    # Calculate grid indices for the third column of 'xy'
    grid_ij = np.zeros((xy.shape[0], 2), dtype=int)
    grid_ij[:, 0] = np.ceil((xy[:, 2] - np.min(xy[:, 2]) + 1e-6) / grid_size).astype(
        int
    )

    # Adjust indices to ensure they are within bounds for the first column
    grid_ij[grid_ij[:, 0] == np.max(grid_ij[:, 0]), 0] = np.max(grid_ij[:, 0]) - 1

    # Calculate grid indices for the second column of 'xy'
    grid_ij[:, 1] = np.ceil((xy[:, 1] - np.min(xy[:, 1]) + 1e-6) / grid_size).astype(
        int
    )

    # Adjust indices to ensure they are within bounds for the second column
    grid_ij[grid_ij[:, 1] == np.max(grid_ij[:, 1]), 1] = np.max(grid_ij[:, 1]) - 1

    grid_ij -= 1  # 0-based indexing

    n_i = np.max(grid_ij[:, 0]) + 1
    n_j = np.max(grid_ij[:, 1]) + 1

    # check("grid_ij", grid_ij+1)

    weighting = 1.0 / da
    gamma_change_save = 0

    log(f"Processing {n_ps} PS candidates")

    for iter in range(1, max_iters + 1):
        log(f"* Iteration {iter}")

        # Initialize phase grids for raw phases, filtered phases, and weighted phases
        ph_grid = np.zeros((n_i, n_j, n_ifg), dtype=np.complex64)
        ph_filt = np.copy(ph_grid)  # Copy ph_grid structure for filtered phases

        # Calculate weighted phases, adjusting for baseline and applying weights
        ph_weight = ph * np.exp(-1j * bperp_mat * K_ps[:, None]) * weighting[:, None]

        # Accumulate weighted phases into grid cells
        for i in range(n_ps):
            ph_grid[grid_ij[i, 0], grid_ij[i, 1], :] = (
                ph_grid[grid_ij[i, 0], grid_ij[i, 1], :] + ph_weight[i, :]
            )

        log("Filtering/smoothing each interferogram in the grid")
        for i in range(n_ifg):
            # Apply a CLAP filter (an edge-preserving smoothing filter) to the phase grid
            ph_filt[:, :, i] = clap_filter(
                ph_grid[:, :, i],
                clap_alpha,
                clap_beta,
                int(n_win * 0.75),
                int(n_win * 0.25),
                low_pass,
            )
            show_progress(i, n_ifg)

        # Extract filtered patch phases for each point
        for i in range(n_ps):
            ph_patch[i, :n_ifg] = ph_filt[grid_ij[i, 0], grid_ij[i, 1], :]

        # check(f"ph_patch_{iter}", ph_patch, atol=1e-3, rtol=1e-3)

        # Clear the filtered phase grid to free memory
        del ph_filt

        # Normalize non-zero phase patch values to unit magnitude
        ix = ph_patch != 0
        ph_patch[ix] = ph_patch[ix] / np.abs(ph_patch[ix])

        log("Estimating topographic phase error")

        K_ps = np.full(n_ps, np.nan)
        C_ps = np.zeros(n_ps)
        coh_ps = np.zeros(n_ps)
        N_opt = np.zeros(n_ps, dtype=int)
        ph_res = np.zeros((n_ps, n_ifg), dtype=np.float32)

        for i in range(n_ps):
            # Calculate phase difference between observed and filtered phase
            psdph = ph[i, :] * np.conj(ph_patch[i, :])

            # Check if there's a non-null value in every interferogram
            if np.all(psdph != 0):
                # Fit the topographic phase model to the phase difference
                Kopt, Copt, cohopt, ph_residual = topofit(
                    psdph, bperp_mat[i, :], n_trial_wraps
                )

                # Store the results
                K_ps[i] = Kopt
                C_ps[i] = Copt
                coh_ps[i] = cohopt
                # N_opt[i] = len(Kopt)
                N_opt[i] = 1
                ph_res[i, :] = np.angle(ph_residual)
            else:
                # Assign default values in case of null values
                K_ps[i] = np.nan
                coh_ps[i] = 0

            show_progress(i, n_ps)

        # Replace NaNs in coherence with zeros
        coh_ps[np.isnan(coh_ps)] = 0

        # Calculate the RMS change in coherence values, ignoring NaNs
        gamma_change_rms = np.sqrt(np.nanmean((coh_ps - coh_ps_save) ** 2))

        log(f"RMS change in coherence: {gamma_change_rms:.6f}")

        # Calculate the change in gamma_change_rms from the previous iteration
        gamma_change_change = np.abs(gamma_change_rms - gamma_change_save)

        # Log the change in gamma_change_rms
        log(f"Change since last iteration: {gamma_change_change:.6f}")

        # Save the current values for comparison in the next iteration
        gamma_change_save = gamma_change_rms
        coh_ps_save = coh_ps.copy()

        if gamma_change_rms < gamma_change_convergence:
            log("Convergence reached, breaking loop")
            break

        # Update the weighting for the next iteration

        if filter_weighting.lower() == "p-square":
            log("Updating weighting using p-square approach...")
            Na, _ = np.histogram(coh_ps, bins=coh_bins)
            Nr *= np.sum(Na[:low_coh_thresh]) / np.sum(
                Nr[:low_coh_thresh]
            )  # scale random distribution
            Na[Na == 0] = 1  # avoid division by zero

            # Calculate the random phase density
            Prand = Nr / Na
            Prand[:low_coh_thresh] = 1
            Prand[Nr_max_nz_ix + 1 :] = 0
            Prand[Prand > 1] = 1

            # Interpolate the random phase density to the phase coherence values
            gauss_filter = gausswin(7)
            Prand = lfilter(
                gauss_filter, 1, np.concatenate((np.ones(7), Prand))
            ) / np.sum(gauss_filter)
            Prand = Prand[7:]  # remove padding
            Prand = interp(np.insert(Prand, 0, 1), 10)[:-9]
            Prand_ps = Prand[np.round(coh_ps * 1000).astype(int)]

            # Calculate the weighting
            weighting = (1 - Prand_ps) ** 2
        else:
            log("Updating weighting using p-linear approach...")

            # Calculate the random phase density
            g = np.mean(A * np.cos(ph_res), axis=1)  # signal
            sigma_n = np.sqrt(0.5 * (np.mean(A**2, axis=1) - g**2))

            # Calculate the weighting
            weighting = np.zeros_like(sigma_n)
            weighting[sigma_n != 0] = g[sigma_n != 0] / sigma_n[sigma_n != 0]  # snr

    stamps_save(
        "pm1",
        ph_patch=ph_patch,  # phase data (n_ps, n_ifg) - complex
        K_ps=K_ps,  # topographic phase model coefficients (n_ps,) - radians
        C_ps=C_ps,  # static phase offset (n_ps,) - radians
        coh_ps=coh_ps,  # coherence values (n_ps,) - unitless
        N_opt=N_opt,  # number of interferograms used in fitting (n_ps,) - unitless
        ph_res=ph_res,  # phase residuals (n_ps, n_ifg) - radians
        ph_grid=ph_grid,  # raw phase data (n_i, n_j, n_ifg) - complex
        n_trial_wraps=n_trial_wraps,  # number of trial wraps (scalar) - unitless
        grid_ij=grid_ij,  # grid indices (n_ps, 2) - unitless
        grid_size=grid_size,  # filter grid size (scalar) - unitless
        low_pass=low_pass,  # low-pass filter (n_win, n_win) - unitless
        i_loop=iter,  # iteration number (scalar) - unitless
        coh_bins=coh_bins,  # coherence bins (n_bins,) - unitless
        Nr=Nr,  # histogram of random coherences (n_bins,) - unitless
        coh_thresh=low_coh_thresh,  # low coherence threshold (scalar) - unitless
        # step_number=step_number,
        # ph_weight=ph_weight,
        # Nr_max_nz_ix=Nr_max_nz_ix,
        # coh_ps_save=coh_ps_save,
        # gamma_change_save=gamma_change_save,
    )


def stage3_select_ps(reest_flag: int = 0, opts: dotdict = dotdict()) -> None:
    """
    Select persistent scatterers based on coherence and phase stability. This
    stage is an iterative process that selects stable-phase pixels based on
    the coherence threshold and the phase stability threshold. The process
    is repeated until the phase stability threshold converges or the maximum
    number of iterations is reached.

    This stage also estimates the percentage of random (non-PS) pixels in the
    in a patch from which the density per square kilometer can be calculated.
    """

    log("# Stage 3: selecting PS based on coherence and phase stability")

    # Reading candidate coherence threshold from file
    with open("../selpsc.in", "r") as fd:
        cand_coh_thresh = float(fd.readline().strip())
    log(f"{cand_coh_thresh = :.3f} (candidate coherence threshold)")

    psver = get_psver()
    if psver > 1:
        set_psver(1)
        psver = 1

    # Retrieve parameters
    slc_osf = float(getparm("slc_osf"))
    clap_alpha = float(getparm("clap_alpha"))
    clap_beta = float(getparm("clap_beta"))
    n_win = int(getparm("clap_win"))
    select_method = getparm("select_method")

    max_percent_rand = float(getparm("percent_rand"))
    max_density_rand = float(getparm("density_rand"))

    gamma_stdev_reject = float(getparm("gamma_stdev_reject"))
    small_baseline_flag = getparm("small_baseline_flag")  # string
    drop_ifg_index = np.fromstring(
        getparm("drop_ifg_index"), sep=" "
    )  # FIXME: This could be a list!

    # Setting low coherence threshold based on small_baseline_flag

    if small_baseline_flag == "y":
        low_coh_thresh = 15
    else:
        low_coh_thresh = 31

    log(f"{select_method = } (selection method)")
    log(f"{slc_osf = } (SLC oversampling factor)")
    log(f"{clap_alpha = } (CLAP alpha)")
    log(f"{clap_beta = } (CLAP beta)")
    log(f"{n_win = } (CLAP window size)")
    log(f"{max_percent_rand = } (maximum percent random)")
    log(f"{max_density_rand = } (maximum density random)")
    log(f"{gamma_stdev_reject = } (gamma standard deviation reject)")
    log(f"{small_baseline_flag = } (small baseline flag)")
    log(f"{drop_ifg_index = } (interferogram indices to drop)")
    log(f"{low_coh_thresh = } (low coherence threshold)")

    # Load data
    ps = stamps_load(f"ps{psver}")
    assert isinstance(ps, dotdict)

    if stamps_exists(f"ph{psver}"):
        ph = stamps_load(f"ph{psver}")
    else:
        ph = ps["ph"]

    assert isinstance(ph, np.ndarray)

    bperp = ps["bperp"]
    n_ifg = int(ps["n_ifg"])

    ifg_index = np.setdiff1d(np.arange(1, ps["n_ifg"] + 1), drop_ifg_index)

    # Adjust ifg_index based on small_baseline_flag
    if not small_baseline_flag == "y":
        master_ix = np.sum(ps["master_day"] > ps["day"]) + 1
        no_master_ix = np.array(
            [i for i in range(1, ps["n_ifg"] + 1) if i not in [master_ix]]
        )
        ifg_index = np.array([i for i in ifg_index if i not in [master_ix]])
        ifg_index = np.array([i - 1 if i > master_ix else i for i in ifg_index])
        no_master_ix -= 1  # Correct for Python indexing
        ifg_index -= 1  # Correct for Python indexing
        ph = ph[:, no_master_ix]
        bperp = bperp[no_master_ix]
        n_ifg = len(no_master_ix)

    n_ps = ps["n_ps"]
    xy = ps["xy"]

    log(f"{n_ps} PS candidates at start")

    # Load pm data
    pm = stamps_load(f"pm{psver}")

    assert isinstance(pm, dict)
    assert "coh_ps" in pm

    # Check if da.mat exists and load D_A, otherwise set D_A to an empty array
    if stamps_exists(f"da{psver}"):
        log("Loading PS candidate dispersions from file")
        D_A = stamps_load(f"da{psver}")
    else:
        log("No PS candidate dispersions found")
        D_A = np.array([])

    assert isinstance(D_A, np.ndarray)

    # Chunk up PSC if D_A is not empty and has a significant number of elements
    if D_A.size >= 10000:
        D_A_sort = np.sort(D_A)
        bs = 10000 if D_A.size >= 50000 else 2000
        D_A_max = np.vstack(
            (np.array([0]), D_A_sort[bs:-bs:bs, np.newaxis], D_A_sort[-1])
        )
    else:
        D_A_max = np.array([0, 1])
        D_A = np.ones_like(pm["coh_ps"])

    assert isinstance(D_A, np.ndarray)

    # check("D_A_sort", D_A_sort)
    # check("D_A_max", D_A_max)

    # Calculate max_percent_rand based on select_method
    if select_method.lower() != "percent":
        patch_area = (
            np.prod(np.max(xy[:, 1:3], axis=0) - np.min(xy[:, 1:3], axis=0)) / 1e6
        )  # Convert to km^2
        max_percent_rand = max_density_rand * patch_area / (len(D_A_max) - 1)

        log(f"{max_density_rand = } (maximum density random)")
        log(f"{patch_area = } (patch area in km^2)")

    # Initialize min_coh and D_A_mean arrays
    min_coh = np.zeros(len(D_A_max) - 1)
    D_A_mean = np.zeros(len(D_A_max) - 1)
    Nr_dist = pm["Nr"]  # Assuming Nr is in pm

    if reest_flag == 3:
        coh_thresh = np.array([0])
        coh_thresh_coeffs = np.array([])

    else:
        for i in range(len(D_A_max) - 1):
            coh_chunk = pm["coh_ps"][(D_A > D_A_max[i]) & (D_A <= D_A_max[i + 1])]
            D_A_mean[i] = D_A[(D_A > D_A_max[i]) & (D_A <= D_A_max[i + 1])].mean()
            coh_chunk = coh_chunk[
                coh_chunk != 0
            ]  # discard PSC for which coherence was not calculated

            Na = np.histogram(coh_chunk, bins=pm["coh_bins"])[0]
            Nr = (
                Nr_dist
                * Na[1 : low_coh_thresh + 1].sum()
                / Nr_dist[1 : low_coh_thresh + 1].sum()
            )

            # check(f"Na_{i+1}", Na)

            Na[Na == 0] = 1  # avoid divide by zero

            if select_method.lower() == "percent":
                percent_rand = np.flip(
                    np.cumsum(np.flip(Nr)) / np.cumsum(np.flip(Na)) * 100
                )
            else:
                percent_rand = np.flip(np.cumsum(np.flip(Nr)))  # absolute number

            ok_ix = np.where(percent_rand < max_percent_rand)[0]
            if ok_ix.size == 0:
                min_coh[i] = 1  # no threshold meets criteria
            else:
                min_fit_ix = ok_ix.min() - 3
                if min_fit_ix <= 0:
                    min_coh[i] = np.nan
                else:
                    max_fit_ix = min(
                        ok_ix.max() + 2, 99
                    )  # ensure not higher than length of percent_rand
                    p, _ = np.polyfit(
                        percent_rand[min_fit_ix : max_fit_ix + 1],
                        np.arange(min_fit_ix, max_fit_ix + 1) * 0.01,
                        3,
                    )
                    p = np.array(p)
                    min_coh[i] = np.polyval(p, max_percent_rand)

        # check("min_coh", min_coh)
        # check("D_A_mean", D_A_mean)

        nonnanix = ~np.isnan(min_coh)
        if nonnanix.sum() < 1:
            log("Not enough random phase pixels to set gamma threshold")
            coh_thresh = np.array([0.3])
            coh_thresh_coeffs = np.array([])
            log(f"Using default gamma threshold: {coh_thresh[0]:.3f}")
        else:
            min_coh = min_coh[nonnanix]
            D_A_mean = D_A_mean[nonnanix]
            if min_coh.size > 1:
                coh_thresh_coeffs = np.polyfit(D_A_mean, min_coh, 1)

                if coh_thresh_coeffs[0] > 0:  # positive slope
                    coh_thresh = np.polyval(coh_thresh_coeffs, D_A)

                else:  # unable to ascertain correct slope
                    coh_thresh = np.polyval(
                        coh_thresh_coeffs, 0.35
                    )  # set an average threshold for all D_A
                    coh_thresh_coeffs = np.array([])

            else:
                coh_thresh = np.array([min_coh])
                coh_thresh_coeffs = np.array([])
            log(f"Using calculated gamma threshold: {coh_thresh[0]:.3f}")

    coh_thresh[coh_thresh < 0] = 0  # Ensures pixels with coh=0 are rejected

    log(f"{min_coh = }")

    log(
        f"Initial gamma threshold: {min(coh_thresh):.3f} at D_A={min(D_A):.2f}"
        f" to {max(coh_thresh):.3f} at D_A={max(D_A):.2f}"
    )

    ix = np.where(pm["coh_ps"] > coh_thresh)[0]  # Select those above threshold
    n_ps = len(ix)
    log(f"{n_ps} PS selected initially")

    if gamma_stdev_reject > 0:
        log("Performing gamma_stdev_reject")

        ph_res_cpx = np.exp(1j * pm["ph_res"][:, ifg_index])
        coh_std = np.zeros(len(ix))

        for i, idx in enumerate(ix):
            # Simplified version: Compute standard deviation of the absolute values directly
            coh_std[i] = np.std(np.abs(ph_res_cpx[idx, ifg_index]) / len(ifg_index))
            # FIXME with scipy.stats.bootstrap ?

        ix = ix[coh_std < gamma_stdev_reject]
        n_ps = len(ix)

        log(f"{n_ps} PS left after pps rejection")

    if reest_flag != 1:
        if reest_flag != 2:
            for i in drop_ifg_index:
                if small_baseline_flag.lower() == "y":
                    md, od = [datestr(x) for x in ps["ifgday"][i]]
                    log(f"{md}-{od} is dropped from noise re-estimation")
                else:
                    breakpoint()
                    dropped = ps.day[i]
                    log(f"{datestr(dropped)} is dropped from noise re-estimation")

            del pm["ph_res"], pm["ph_patch"]
            ph_patch2 = np.zeros((n_ps, n_ifg), dtype=np.complex64)
            ph_res2 = np.zeros((n_ps, n_ifg), dtype=np.float32)
            ph = ph[ix, :]

            if len(coh_thresh) > 1:
                coh_thresh = coh_thresh[ix]

            n_i, n_j = np.max(pm["grid_ij"], axis=0)
            K_ps2 = np.zeros(n_ps)
            C_ps2 = np.zeros(n_ps)
            coh_ps2 = np.zeros(n_ps)
            ph_filt = np.zeros((n_win, n_win, n_ifg), dtype=np.complex64)

            ij_idxs = np.zeros((n_ps, 6), dtype=int)

            log("Re-estimating PS coherences and phases:")

            for i in range(n_ps):
                ps_ij = pm["grid_ij"][ix[i]]  # grid_ij is 0-based

                i_min = max(ps_ij[0] - n_win // 2, 0)
                i_max = i_min + n_win - 1
                j_min = max(ps_ij[1] - n_win // 2, 0)
                j_max = j_min + n_win - 1
                if i_max > n_i:
                    i_min = i_min - i_max + n_i
                    i_max = n_i
                if j_max > n_j:
                    j_min = j_min - j_max + n_j
                    j_max = n_j

                ij_idxs[i] = [ps_ij[0], ps_ij[1], i_min, i_max, j_min, j_max]

                if j_min < 0 or i_min < 0:
                    ph_patch2[i, :] = 0
                else:
                    # Remove the pixel for which the smoothing is computed
                    ps_bit_i = ps_ij[0] - i_min
                    ps_bit_j = ps_ij[1] - j_min
                    ph_bit = pm["ph_grid"][
                        i_min : (i_max + 1), j_min : (j_max + 1), :
                    ].copy()
                    ph_bit[ps_bit_i, ps_bit_j, :] = 0

                    # Oversample update for PS removal + general usage update
                    ix_i = np.arange(
                        max(ps_bit_i - (slc_osf - 1), 0),
                        min(ps_bit_i + slc_osf, ph_bit.shape[0]),
                        dtype=int,
                    )

                    ix_j = np.arange(
                        max(ps_bit_j - (slc_osf - 1), 0),
                        min(ps_bit_j + slc_osf, ph_bit.shape[1]),
                        dtype=int,
                    )

                    # Set the oversampled region to 0
                    ph_bit[np.ix_(ix_i, ix_j)] = 0

                    for ifg in range(n_ifg):
                        ph_filt[:, :, ifg] = clap_filter_patch(
                            ph_bit[:, :, ifg],
                            clap_alpha,
                            clap_beta,
                            pm["low_pass"],
                        )

                    ph_patch2[i] = ph_filt[ps_bit_i, ps_bit_j, :]

                show_progress(i, n_ps)

            del pm["ph_grid"]
            bp = stamps_load(f"bp{psver}")
            bperp_mat = bp[ix, :]

            log("Performing a topographic phase model fit to the PS candidates:")

            for i in range(n_ps):
                psdph = ph[i] * np.conj(ph_patch2[i])
                if not np.any(
                    psdph == 0
                ):  # Ensure there's a non-null value in every interferogram
                    psdph /= np.abs(psdph)
                    Kopt, Copt, cohopt, ph_residual = topofit(
                        psdph[ifg_index],
                        bperp_mat[i, ifg_index],
                        pm["n_trial_wraps"],
                        False,
                    )
                    K_ps2[i] = Kopt
                    C_ps2[i] = Copt
                    coh_ps2[i] = cohopt
                    ph_res2[i, ifg_index] = np.angle(ph_residual)
                else:
                    K_ps2[i] = np.nan
                    coh_ps2[i] = np.nan

                show_progress(i, n_ps)

            # check("K_ps2", K_ps2)
            # check("C_ps2", C_ps2, atol=1e-3, rtol=1e-3)
            # check("coh_ps2", coh_ps2, atol=1e-3, rtol=1e-3)

        else:  # reest_flag == 2, use previously recalculated coh
            sl = stamps_load(f"select{psver}")
            ix = sl["ix"].flatten()
            coh_ps2 = sl["coh_ps2"].flatten()
            K_ps2 = sl["K_ps2"].flatten()
            C_ps2 = sl["C_ps2"].flatten()
            ph_res2 = sl["ph_res2"]
            ph_patch2 = sl["ph_patch2"]

        pm["coh_ps"][ix] = coh_ps2

        for i in range(len(D_A_max) - 1):
            coh_chunk = pm["coh_ps"][(D_A > D_A_max[i]) & (D_A <= D_A_max[i + 1])]
            D_A_mean[i] = D_A[(D_A > D_A_max[i]) & (D_A <= D_A_max[i + 1])].mean()
            coh_chunk = coh_chunk[
                coh_chunk != 0
            ]  # Discard PSC for which coherence was not calculated

            Na, _ = np.histogram(coh_chunk, bins=pm["coh_bins"])
            Nr = (
                Nr_dist
                * Na[: low_coh_thresh + 1].sum()
                / Nr_dist[: low_coh_thresh + 1].sum()
            )

            Na[Na == 0] = 1  # Avoid divide by zero

            if select_method.lower() == "percent":
                percent_rand = np.flip(
                    np.cumsum(np.flip(Nr)) / np.cumsum(np.flip(Na)) * 100
                )
            else:
                percent_rand = np.flip(np.cumsum(np.flip(Nr)))  # Absolute number

            ok_ix = np.where(percent_rand < max_percent_rand)[0]
            if ok_ix.size == 0:
                min_coh[i] = 1
            else:
                min_fit_ix = min(ok_ix) - 3
                if min_fit_ix <= 0:
                    min_coh[i] = np.nan
                else:
                    max_fit_ix = min(ok_ix) + 2

                    # Ensure max_fit_ix does not exceed the length of percent_rand
                    max_fit_ix = np.minimum(max_fit_ix, len(percent_rand))

                    # Fit a polynomial of degree 3 to the data
                    p = np.polyfit(
                        np.arange(min_fit_ix, max_fit_ix + 1) * 0.01,
                        percent_rand[min_fit_ix : max_fit_ix + 1],
                        3,
                    )

                    # Evaluate the polynomial at max_percent_rand
                    min_coh[i] = np.polyval(p, max_percent_rand)

        # check("min_coh2", min_coh)

        nonnanix = ~np.isnan(min_coh)
        if nonnanix.sum() < 1:
            coh_thresh = np.array([0.3])
            coh_thresh_coeffs = np.array([])
        else:
            min_coh = min_coh[nonnanix]
            D_A_mean = D_A_mean[nonnanix]
            if min_coh.size > 1:
                coh_thresh_coeffs = np.polyfit(D_A_mean, min_coh, 1)
                if coh_thresh_coeffs[0] > 0:
                    coh_thresh = np.polyval(coh_thresh_coeffs, D_A[ix])
                else:
                    coh_thresh = np.polyval(coh_thresh_coeffs, 0.35)
                    coh_thresh_coeffs = np.array([])
            else:
                coh_thresh = np.array([min_coh])
                coh_thresh_coeffs = np.array([])

        coh_thresh[coh_thresh < 0] = 0

        log(
            f"Reestimation of threshold: {min(coh_thresh):.3f} at D_A={min(D_A):.2f}"
            f"to {max(coh_thresh):.3f} at D_A={max(D_A):.2f}"
        )

        bperp_range = max(bperp) - min(bperp)
        keep_ix = (coh_ps2 > coh_thresh) & (
            np.abs(pm["K_ps"][ix] - K_ps2) < 2 * np.pi / bperp_range
        )

        log(f"{keep_ix.sum()} PS selected after re-estimation of coherence")

    else:
        del pm["ph_grid"]
        ph_patch2 = pm["ph_patch"][ix]
        ph_res2 = pm["ph_res"][ix]
        K_ps2 = pm["K_ps"][ix]
        C_ps2 = pm["C_ps"][ix]
        coh_ps2 = pm["coh_ps"][ix]
        keep_ix = np.ones_like(ix, dtype=bool)

    if stamps_exists("no_ps_info"):
        stamps_step_no_ps = stamps_load("no_ps_info")
        stamps_step_no_ps[2:] = 0
    else:
        stamps_step_no_ps = np.zeros(5)

    if keep_ix.sum() == 0:
        log("***No PS points left. Updating the stamps log for this***")
        stamps_step_no_ps[2] = 1

    stamps_save("no_ps_info", stamps_step_no_ps=stamps_step_no_ps)

    log(f"{min_coh = }")
    log(f"{coh_thresh_coeffs = }")

    stamps_save(
        f"select{psver}",
        ix=ix,
        keep_ix=keep_ix,
        ph_patch2=ph_patch2,  # phase data (n_ps, n_ifg) - complex
        ph_res2=ph_res2,  # phase residuals (n_ps, n_ifg) - radians
        K_ps2=K_ps2,  # topographic phase model coefficients (n_ps,) - radians
        C_ps2=C_ps2,  # static phase offset (n_ps,) - radians
        coh_ps2=coh_ps2,  # coherence values (n_ps,) - unitless
        coh_thresh=coh_thresh,  # coherence threshold (n_ps,) - unitless
        coh_thresh_coeffs=coh_thresh_coeffs,  # coherence threshold coefficients (2,) - unitless
        clap_alpha=clap_alpha,  # CLAP alpha (scalar) - unitless
        clap_beta=clap_beta,  # CLAP beta (scalar) - unitless
        n_win=n_win,  # CLAP window size (scalar) - unitless
        max_percent_rand=round(
            max_percent_rand, 0
        ),  # maximum percent random (scalar) - unitless
        gamma_stdev_reject=gamma_stdev_reject,  # gamma standard deviation reject (scalar) - unitless
        small_baseline_flag=small_baseline_flag,  # small baseline flag (string) - unitless
        ifg_index=ifg_index + 1,  # 1-based indexing to match MATLAB
    )


def stage4_weed_ps(
    all_da_flag: bool = False,
    no_weed_adjacent: bool = False,
    no_weed_noisy: bool = False,
    opts: dotdict = dotdict(),
) -> None:
    """
    PS selected in the previous stage are possibly dropped if they are adjacent
    to another PS or if they have a high standard deviation in the phase noise.
    """

    log("# Stage 4: weeding out PS based on adjacency and phase noise")

    # Load hardcoded pixels
    input_azrg = Path("../input_azrg")
    if input_azrg.exists():
        forced_ij = np.loadtxt(str(input_azrg), dtype=int)
        log(f"loaded {forced_ij.shape[0]} hardcoded pixels from `input_azrg`")
    else:
        forced_ij = np.zeros((0, 2), dtype=int)
        log("no hardcoded pixels loaded")

    time_win = float(getparm("weed_time_win"))
    weed_standard_dev = float(getparm("weed_standard_dev"))
    weed_max_noise = float(getparm("weed_max_noise"))
    weed_zero_elevation = getparm("weed_zero_elevation")  # string
    weed_neighbours = getparm("weed_neighbours")  # string
    small_baseline_flag = getparm("small_baseline_flag")  # string

    drop_ifg_index = eval(getparm("drop_ifg_index"))  # FIXME?
    assert isinstance(drop_ifg_index, list)

    log(f"{time_win = } (time window)")
    log(f"{weed_standard_dev = } (standard deviation)")
    log(f"{weed_max_noise = } (max noise)")
    log(f"{weed_zero_elevation = } (zero elevation)")
    log(f"{weed_neighbours = } (neighbours)")
    log(f"{drop_ifg_index = } (interferogram indices to drop)")
    log(f"{small_baseline_flag = } (small baseline flag)")

    if not no_weed_adjacent:
        if weed_neighbours.lower() == "y":
            no_weed_adjacent = False
        else:
            no_weed_adjacent = True

    if not no_weed_noisy:
        if weed_standard_dev >= np.pi and weed_max_noise >= np.pi:
            no_weed_noisy = True
        else:
            no_weed_noisy = False

    log(f"{no_weed_adjacent = }")
    log(f"{no_weed_noisy = }")

    psver = get_psver()
    log(f"{psver = }")

    ps = stamps_load(f"ps{psver}")
    ifg_index = np.array(
        np.setdiff1d(np.arange(1, ps["n_ifg"] + 1), drop_ifg_index) - 1
    )  # 0-based indexing

    sl = stamps_load(f"select{psver}")

    if stamps_exists(f"ph{psver}"):
        ph = stamps_load(f"ph{psver}")
    else:
        ph = ps["ph"]

    day = ps["day"]
    bperp = ps["bperp"]
    master_day = ps["master_day"]

    forced_ij_idx = np.where(np.all(ps["ij"][:, 1:] == forced_ij[:, None], axis=2))[1]
    forced_ix = ps["ij"][forced_ij_idx, 0]

    if "keep_ix" in sl:
        forced_ix2 = np.where(np.isin(sl["ix"], forced_ix))
        sl["keep_ix"][forced_ix2] = 1

        keep_ix = sl["keep_ix"].astype(bool)  # FIXME: 1-based indexing?

        ix2 = sl["ix"][keep_ix]
        K_ps2 = sl["K_ps2"][keep_ix]
        C_ps2 = sl["C_ps2"][keep_ix]
        coh_ps2 = sl["coh_ps2"][keep_ix]
    else:
        ix2 = sl["ix2"]
        K_ps2 = sl["K_ps2"]
        C_ps2 = sl["C_ps2"]
        coh_ps2 = sl["coh_ps2"]

    log(f"{ix2.shape[0]} pts")

    ij2 = ps["ij"][ix2, :]
    xy2 = ps["xy"][ix2, :]
    ph2 = ph[ix2, :]
    lonlat2 = ps["lonlat"][ix2, :]

    # log(np.where(np.all(ij2[:, 1:] == forced_ij[:, None], axis=2)))

    pm = stamps_load(f"pm{psver}")
    ph_patch2 = pm["ph_patch"][ix2, :]  # use original patch phase, with PS left in

    if "ph_res2" in sl:
        ph_res2 = sl["ph_res2"][keep_ix, :]
    else:
        ph_res2 = []

    assert isinstance(ps, dict)

    if "ph" in ps:
        del ps["ph"]

    ps.pop("xy")
    ps.pop("ij")
    ps.pop("lonlat")
    ps.pop("sort_ix")

    if all_da_flag:
        pso = stamps_load("ps_other")
        slo = stamps_load("select_other")

        ix_other = slo["ix_other"].astype(bool)
        n_ps_other = np.sum(ix_other)
        K_ps_other2 = pso["K_ps_other"][ix_other]
        C_ps_other2 = pso["C_ps_other"][ix_other]
        coh_ps_other2 = pso["coh_ps_other"][ix_other]
        ph_res_other2 = pso["ph_res_other"][ix_other, :]

        ij2 = np.vstack((ij2, pso["ij_other"][ix_other, :]))
        xy2 = np.vstack((xy2, pso["xy_other"][ix_other, :]))
        ph2 = np.vstack((ph2, pso["ph_other"][ix_other, :]))
        lonlat2 = np.vstack((lonlat2, pso["lonlat_other"][ix_other, :]))

        pmo = stamps_load("pm_other")
        ph_patch_other2 = pmo["ph_patch_other"][ix_other, :]

        K_ps2 = np.hstack((K_ps2, K_ps_other2))
        C_ps2 = np.hstack((C_ps2, C_ps_other2))
        coh_ps2 = np.hstack((coh_ps2, coh_ps_other2))
        ph_patch2 = np.vstack((ph_patch2, ph_patch_other2))
        ph_res2 = np.vstack((ph_res2, ph_res_other2))
    else:
        n_ps_other = 0

    log(f"n_ps_other = {n_ps_other}")

    if stamps_exists(f"hgt{psver}"):
        hgt = stamps_load(f"hgt{psver}")[ix2]
        if all_da_flag:
            hgt_other = stamps_load("hgt_other")[ix_other]
            hgt = np.hstack((hgt, hgt_other))

    n_ps_low_D_A = ix2.shape[0]
    n_ps = n_ps_low_D_A + n_ps_other
    ix_weed = np.ones(n_ps, dtype=bool)

    log(f"{n_ps_low_D_A} low D_A PS, {n_ps_other} high D_A PS")

    if not no_weed_adjacent:  # Weeding adjacent pixels
        log("Initializing neighbour matrix")

        ij_shift = ij2[:, 1:] + np.tile([2, 2] - np.min(ij2[:, 1:], axis=0), (n_ps, 1))
        neigh_ix = np.zeros(
            (np.max(ij_shift[:, 0]) + 1, np.max(ij_shift[:, 1]) + 1), dtype=int
        )
        miss_middle = np.ones((3, 3), dtype=bool)
        miss_middle[1, 1] = False

        for i in range(n_ps):
            neigh_this = neigh_ix[
                ij_shift[i, 0] - 1 : ij_shift[i, 0] + 2,
                ij_shift[i, 1] - 1 : ij_shift[i, 1] + 2,
            ]
            neigh_this[neigh_this == 0] = i + 1
            neigh_ix[
                ij_shift[i, 0] - 1 : ij_shift[i, 0] + 2,
                ij_shift[i, 1] - 1 : ij_shift[i, 1] + 2,
            ] = neigh_this

            if (i + 1) % 100000 == 0:
                log(f"{i + 1} PS processed")

        # check("neigh_ix", neigh_ix + 1, exit=True)

        log("Finding neighbours")

        neigh_ps: List[Array] = [np.array([]) for _ in range(n_ps)]
        for i in range(n_ps):
            my_neigh_ix = neigh_ix[ij_shift[i, 0], ij_shift[i, 1]]
            if my_neigh_ix != 0:
                neigh_ps[my_neigh_ix - 1] = np.append(neigh_ps[my_neigh_ix - 1], i + 1)

            if (i + 1) % 100000 == 0:
                log(f"{i + 1} PS processed")

        log("Selecting best PS from each group")

        for i in range(n_ps):
            if neigh_ps[i]:
                same_ps = np.array([i + 1])
                i2 = 0
                while i2 < len(same_ps):
                    ps_i = same_ps[i2] - 1
                    same_ps = np.append(same_ps, neigh_ps[ps_i])
                    neigh_ps[ps_i] = []
                    i2 += 1
                same_ps = np.unique(same_ps)
                high_coh = np.argmax(coh_ps2[same_ps - 1])
                low_coh_ix = np.ones(same_ps.shape, dtype=bool)
                low_coh_ix[high_coh] = False
                ix_weed[same_ps[low_coh_ix] - 1] = False

            if (i + 1) % 100000 == 0:
                log(f"{i + 1} PS processed")

        log(f"{np.sum(ix_weed)} PS kept after dropping adjacent pixels")

    # output how many PS are left after weeding zero elevations out
    if weed_zero_elevation.lower() == "y" and "hgt" in locals():
        sea_ix = hgt < 1e-6
        ix_weed[sea_ix] = False
        log(f"{np.sum(ix_weed)} PS kept after weeding zero elevation")

    xy_weed = xy2[ix_weed, :]
    n_ps = np.sum(ix_weed)

    log("Removing duplicated points")

    ix_weed_num = np.where(ix_weed)[0]
    _, unique_indices = np.unique(xy_weed[:, 1:], axis=0, return_index=True)
    dups = np.setdiff1d(
        np.arange(np.sum(ix_weed)), unique_indices
    )  # pixels with duplicate lon/lat

    for i in range(len(dups)):
        dups_ix_weed = np.where(
            (xy_weed[:, 1] == xy_weed[dups[i], 1])
            & (xy_weed[:, 2] == xy_weed[dups[i], 2])
        )[0]
        dups_ix = ix_weed_num[dups_ix_weed]
        max_coh_ix = np.argmax(coh_ps2[dups_ix])
        ix_weed[dups_ix[np.arange(len(dups_ix)) != max_coh_ix]] = (
            False  # drop dups with lowest coh
        )

    if len(dups) > 0:
        xy_weed = xy2[ix_weed, :]
        log(f"{len(dups)} PS with duplicate lon/lat dropped")
    else:
        log("No PS with duplicate lon/lat")

    n_ps = np.sum(ix_weed)
    ix_weed2 = np.ones(n_ps, dtype=bool)

    # check("xy_weed", xy_weed)

    log("Weeding noisy pixels")

    ps_std = np.zeros(n_ps)
    ps_max = np.zeros(n_ps)

    if n_ps > 0 and not no_weed_noisy:
        if Path(TRIANGLE).exists():
            nodename = "psweed.1.node"
            with open(nodename, "w") as fid:
                fid.write(f"{n_ps} 2 0 0\n")
                for i in range(n_ps):
                    fid.write(f"{i+1} {xy_weed[i, 1]} {xy_weed[i, 2]}\n")

            if DEBUG:
                subprocess.call([TRIANGLE, "-e", "psweed.1.node"], stdout=sys.stdout)
            else:
                subprocess.call(
                    [TRIANGLE, "-e", "psweed.1.node"],
                    stdout=open("triangle_weed.log", "w"),
                )

            with open("psweed.2.edge", "r") as fid:
                header = np.fromstring(fid.readline().strip(), sep=" ")
                n = int(header[0])
                log(f"{n} edges found")
                edgs = np.zeros((n, 4), dtype=int)
                for i in range(n):
                    edgs[i, :] = np.fromstring(fid.readline().strip(), sep=" ")
                edgs = edgs[:, 1:3] - 1  # 0-based indexing

        else:
            # use Delaunay triangulation from scipy
            from scipy.spatial import Delaunay

            xy_weed = xy_weed.astype(float)
            tri = Delaunay(xy_weed[:, 1:3])
            edgs = tri.simplices.copy()

        n_edge = edgs.shape[0]

        # check("edgs", edgs + 1)  # 1-based indexing

        # Subtract range error and add master noise if applicable
        ph_weed = ph2[ix_weed, :] * np.exp(-1j * (K_ps2[ix_weed][:, None] * bperp))
        ph_weed = ph_weed / np.abs(ph_weed)

        if small_baseline_flag.lower() != "y":  # add master noise
            ph_weed[:, ps["master_ix"]] = np.exp(1j * C_ps2[ix_weed])

        # Noise estimation for edges
        edge_std = np.zeros(n_edge)
        edge_max = np.zeros(n_edge)
        dph_space = ph_weed[edgs[:, 1], :] * np.conj(ph_weed[edgs[:, 0], :])
        dph_space = dph_space[:, ifg_index]

        n_use = len(ifg_index)
        for i in drop_ifg_index:
            if small_baseline_flag.lower() == "y":
                ds = datetime.strptime(str(ps["ifgday"][i, 1]), "%Y%m%d").strftime(
                    "%Y-%m-%d"
                )
                log(f"{ds}-{ds} dropped from noise estimation")
            else:
                ds = datetime.strptime(str(day[i]), "%Y%m%d").strftime("%Y-%m-%d")
                log(f"{ds} dropped from noise estimation")

        if not small_baseline_flag.lower() == "y":
            log(f"Estimating noise for {n_use} arcs:")

            # This section performs noise estimation for all edges in a set of
            # interferograms by smoothing the differential phase values and then
            # calculating noise as the difference between the original and
            # smoothed values. It uses weighted linear regression to adjust for
            # temporal changes and estimates DEM error to further refine the
            # noise estimates.

            # Initialize arrays to store smoothed differential phase values
            dph_smooth = np.zeros((n_edge, n_use), dtype=np.complex64)
            dph_smooth2 = np.zeros((n_edge, n_use), dtype=np.complex64)

            for i1 in range(n_use):
                # Calculate time differences between the current interferogram
                # and all others
                time_diff = day[ifg_index[i1]] - day[ifg_index]

                # Calculate weighting factors based on time differences, using a
                # Gaussian function
                weight_factor = np.exp(-(time_diff**2) / (2 * time_win**2))
                weight_factor /= np.sum(weight_factor)  # Normalize

                # Compute the mean differential phase for each edge, weighted by
                # time proximity
                dph_mean = np.sum(dph_space * weight_factor[None, :], axis=1)

                # Adjust the mean differential phase by subtracting the weighted
                # mean from each value
                dph_mean_adj = np.angle(dph_space * np.conj(dph_mean[:, None]))

                # Prepare a design matrix for linear regression, including a
                # constant term and time differences
                G = np.vstack([np.ones(n_use), time_diff]).T

                # Perform weighted linear least squares to fit the adjusted mean
                # differential phase
                m = lscov(G, dph_mean_adj.T, weight_factor)

                # Update the adjusted mean differential phase by subtracting the
                # linear fit
                dph_mean_adj = np.angle(np.exp(1j * (dph_mean_adj - (G @ m).T)))

                # Perform a second round of weighted linear least squares on the
                # updated adjusted mean differential phase
                m2 = lscov(G, dph_mean_adj.T, weight_factor)

                # Combine the original mean phase with the corrections from both
                # rounds of linear regression
                dph_smooth[:, i1] = dph_mean * np.exp(1j * (m[0] + m2[0]))

                # Zero out the weight factor for the current interferogram to
                # exclude it from its own smoothing
                weight_factor[i1] = 0

                # Recalculate the smoothed differential phase without the
                # current interferogram
                dph_smooth2[:, i1] = np.sum(dph_space * weight_factor[None, :], axis=1)

                show_progress(i1, n_use)

            # Calculate the noise by subtracting the smoothed phase from the
            # original differential phase
            dph_noise = np.angle(dph_space * np.conj(dph_smooth))

            # Repeat the noise calculation using the second set of smoothed phases
            dph_noise2 = np.angle(dph_space * np.conj(dph_smooth2))

            # Calculate the variance of the second set of noise estimates,
            # ignoring NaN values
            ifg_var = np.var(dph_noise2, axis=0, ddof=1, where=~np.isnan(dph_noise2))

            # Estimate the DEM error for each arc using a weighted linear fit
            # with weights based on the noise variance
            K = lscov(bperp[ifg_index, np.newaxis], dph_noise.T, 1 / ifg_var)

            # Adjust the noise estimates by subtracting the estimated DEM error
            dph_noise -= K.T @ bperp[ifg_index].reshape(1, -1)

            # Calculate the standard deviation of the adjusted noise estimates
            # for each edge
            edge_std = np.std(dph_noise, axis=1, ddof=1)

            # Find the maximum absolute noise estimate for each edge
            edge_max = np.max(np.abs(dph_noise), axis=1)

        if small_baseline_flag.lower() == "y":
            # Variance of the differential phase space along the interferograms
            # axis
            ifg_var = np.var(dph_space, axis=1, ddof=1)

            # Least squares fitting to estimate arc DEM error
            K = lscov(bperp[ifg_index], dph_space, 1 / ifg_var)

            # Adjust dph_space based on the estimated arc DEM error
            dph_space_adjusted = dph_space - (K.T @ bperp[ifg_index].reshape(1, -1)).T

            # Calculate the standard deviation and maximum of the phase angles
            # of the adjusted differential phase space
            edge_std = np.std(np.angle(dph_space_adjusted), axis=1, ddof=1)
            edge_max = np.max(np.abs(np.angle(dph_space_adjusted)), axis=1)

            # Save memory
            del dph_space

        # check("edge_std", edge_std, atol=1e-3, rtol=1e-3)
        # check("edge_max", edge_max, atol=1e-3, rtol=1e-3)

        # We now remove points with excessive noise. We calculate the standard
        # deviation and maximum noise level for each pixel based on noise
        # estimates for edges connecting the pixels. Then, we apply thresholds
        # to identify and keep only the pixels with noise levels below these
        # thresholds (`weed_standard_dev` and `weed_max_noise`), effectively
        # weeding out noisy pixels from the dataset.

        log("Estimating max noise for all pixels")

        ps_std = np.full(n_ps, np.inf, dtype=np.float32)
        ps_max = np.full(n_ps, np.inf, dtype=np.float32)
        for i in range(n_edge):
            ps_std[edgs[i, :]] = np.minimum(
                ps_std[edgs[i, :]], [edge_std[i], edge_std[i], edge_std[i]]
            )
            ps_max[edgs[i, :]] = np.minimum(
                ps_max[edgs[i, :]], [edge_max[i], edge_max[i], edge_max[i]]
            )

        ix_weed2 = (ps_std < weed_standard_dev) & (ps_max < weed_max_noise)
        ix_weed[ix_weed] = ix_weed2
        n_ps = np.sum(ix_weed)

        # check("ps_std", ps_std, atol=1e-3, rtol=1e-3)
        # check("ps_max", ps_max, atol=1e-3, rtol=1e-3)

        log(f"{n_ps} PS kept after dropping noisy pixels")

    if n_ps == 0:
        log("Error: No PS left after weeding")
        sys.exit(1)

    # Remove hard-coded points from ix_weed based on forced_ij coordinates
    forced_ij_mask = np.isin(ij2[:, 1:3], forced_ij).all(axis=1)
    n_remain = ix_weed[forced_ij_mask].sum()
    log(f"{n_remain} fixed locations remaining")

    ix_weed[forced_ij_mask] = True
    n_remain = ix_weed[forced_ij_mask].sum()
    log(f"{n_remain} fixed locations remaining after forcing")

    stamps_save(
        f"weed{psver}",
        ix_weed=ix_weed,
        ix_weed2=ix_weed2,
        ps_std=ps_std,
        ps_max=ps_max,
        ifg_index=ifg_index + 1,  # Back to 1-based indexing
    )

    # Save selected pixel metrics
    coh_ps = coh_ps2[ix_weed]
    K_ps = K_ps2[ix_weed]
    C_ps = C_ps2[ix_weed]
    ph_patch = ph_patch2[ix_weed, :]
    ph_res = ph_res2[ix_weed, :] if ph_res2.size else ph_res2

    stamps_save(
        f"pm{psver + 1}",
        ph_patch=ph_patch,  # phase data (n_ps, n_ifg) - complex
        ph_res=ph_res,  # phase residuals (n_ps, n_ifg) - radians
        coh_ps=coh_ps,  # coherence values (n_ps,) - unitless
        K_ps=K_ps,  # topographic phase model coefficients (n_ps,) - radians
        C_ps=C_ps,  # static phase offset (n_ps,) - radians
    )

    # Prepare phase data for saving
    ph2 = ph2[ix_weed, :]
    stamps_save(f"ph{psver+1}", ph2)  # phase data (n_ps, n_ifg) - complex

    # Update PS information with weed results
    xy2 = xy2[ix_weed, :]
    ij2 = ij2[ix_weed, :]
    lonlat2 = lonlat2[ix_weed, :]

    ps.update({"xy": xy2, "ij": ij2, "lonlat": lonlat2, "n_ps": ph2.shape[0]})
    psname = f"ps{psver + 1}"
    stamps_save(psname, ps)  # phase data (n_ps, n_ifg) - complex

    # Process and save height data if available
    if stamps_exists(f"hgt{psver}"):
        hgt = hgt[ix_weed]
        stamps_save(f"hgt{psver + 1}", hgt)

    # Process and save look angle data if available
    if stamps_exists(f"la{psver}"):
        la = stamps_load(f"la{psver}")
        la = la[ix2]
        if all_da_flag:
            la_other = la[ix_other]
            la = np.concatenate([la, la_other])
        la = la[ix_weed]
        stamps_save(f"la{psver + 1}", la)  # look angle data (n_ps,) - radians

    # Process and save incidence angle data if available
    if stamps_exists(f"inc{psver}"):
        inc = stamps_load(f"inc{psver}")
        inc = inc[ix2]
        if all_da_flag:
            inc_other = inc[ix_other]
            inc = np.concatenate([inc, inc_other])
        inc = inc[ix_weed]
        stamps_save(f"inc{psver + 1}", inc)  # incidence angle data (n_ps,) - radians

    # Process and save baseline data if available
    if stamps_exists(f"bp{psver}"):
        bperp_mat = stamps_load(f"bp{psver}")
        bperp_mat = bperp_mat[ix2, :]
        if all_da_flag:
            bperp_other = bperp[ix_other, :]
            bperp_mat = np.concatenate([bperp_mat, bperp_other])
        bperp_mat = bperp_mat[ix_weed, :]
        stamps_save(f"bp{psver + 1}", bperp_mat)  # baseline data (n_ps, n_ifg) - meters


def stage5_correct_phases(opts: dotdict = dotdict()) -> None:
    """
    Correct the wrapped phases of the selected PS for spatially-uncorrelated
    look angle (DEM) error. This is done by subtracting the range error and
    master noise from the phase data. The corrected phase data is saved for
    further processing.
    """

    log("# Stage 5: Correcting phase for look angle (DEM) error")

    small_baseline_flag = getparm("small_baseline_flag")

    psver = get_psver()
    log(f"{psver = }")

    ps = stamps_load(f"ps{psver}")
    pm = stamps_load(f"pm{psver}")
    bp = stamps_load(f"bp{psver}")

    if stamps_exists(f"ph{psver}"):
        ph = stamps_load(f"ph{psver}")
    else:
        ph = ps["ph"]

    n_ifg = ps["n_ifg"]
    n_ps = ps["n_ps"]
    master_ix = int(np.sum(ps["master_day"] > ps["day"]))

    del ps

    K_ps = pm["K_ps"].astype(np.float32)
    C_ps = pm["C_ps"].astype(np.float32)
    ph_patch = pm["ph_patch"]

    del pm

    if small_baseline_flag.lower() == "y":
        # Perform phase correction by subtracting range error
        ph_rc = ph * np.exp(-1j * (K_ps[:, np.newaxis] * bp))

        # Save the corrected phase
        stamps_save(f"rc{psver}", ph_rc=ph_rc)

    else:  # not small baseline
        bperp_mat = np.hstack(
            (
                bp[:, :master_ix],
                np.zeros((n_ps, 1)),
                bp[:, master_ix:],
            )
        )

        del bp

        # Perform phase correction by subtracting range error and master noise
        ph_rc = ph * np.exp(
            1j
            * (
                -K_ps[:, np.newaxis] * bperp_mat - C_ps[:, np.newaxis] * np.ones(n_ifg)
            )  # - range error  # - master noise
        )

        ph_reref = np.hstack(
            (
                ph_patch[:, :master_ix],
                np.ones((n_ps, 1)),
                ph_patch[:, master_ix:],
            )
        )

        # Save the corrected phase and ph_reref
        stamps_save(
            f"rc{psver}",
            ph_rc=ph_rc,  # phase data (n_ps, n_ifg) - complex
            ph_reref=ph_reref,  # phase reference (n_ps, n_ifg) - complex
        )


def ps_calc_ifg_std() -> None:
    """Calculate the standard deviation of the interferograms."""

    small_baseline_flag = getparm("small_baseline_flag")
    psver = get_psver()

    ps = stamps_load(f"ps{psver}")
    pm = stamps_load(f"pm{psver}")
    bp = stamps_load(f"bp{psver}")
    ph = stamps_load(f"ph{psver}")

    assert isinstance(ps, dict)
    assert isinstance(pm, dict)
    assert isinstance(bp, np.ndarray)
    assert isinstance(ph, np.ndarray)

    # n_ifg = ps["n_ifg"]
    n_ps = ps["n_ps"]
    master_ix = np.sum(ps["master_day"] > ps["day"])

    log("Estimating noise standard deviation (in degrees)")

    if small_baseline_flag == "y":
        ph_diff = np.angle(
            ph * np.conj(pm.ph_patch) * np.exp(-1j * (pm.K_ps[:, np.newaxis] * bp))
        )
    else:
        bperp_mat = np.hstack(
            (
                bp[:, :master_ix],
                np.zeros((n_ps, 1)),
                bp[:, master_ix:],
            )
        )

        ph_patch = np.hstack(
            (
                pm.ph_patch[:, :master_ix],
                np.ones((n_ps, 1)),
                pm.ph_patch[:, master_ix:],
            )
        )

        ph_diff = np.angle(
            ph
            * np.conj(ph_patch)
            * np.exp(
                1j
                * (
                    -pm.K_ps[:, np.newaxis] * bperp_mat
                    - pm.C_ps[:, np.newaxis] * np.ones_like(bperp_mat)
                )
            )
        )

    ifg_mean = np.degrees(np.nanmean(ph_diff, axis=0))
    ifg_std = np.degrees(np.nanstd(ph_diff, axis=0))

    if small_baseline_flag == "y":
        ifgday = ps.ifgday
        for i in range(ps["n_ifg"]):
            log(
                f"{i+1:3d}\t{datestr(ifgday[i, 0])}_{datestr(ifgday[i, 1])}\t{ifg_std[i]:>3.2f}"
            )
    else:
        day = ps.day
        log("INDEX    IFG_DATE       MEAN    STD_DEV")
        for i in range(ps["n_ifg"]):
            log(
                f"{i+1:5d}  {datestr(day[i]):10s}    {ifg_mean[i]:>6.2f}°    {ifg_std[i]:>6.2f}°"
            )

    stamps_save(f"ifgstd{psver}", ifg_std=ifg_std)


def stage6_unwrap_phases(opts: dotdict = dotdict()) -> None:
    """Unwrap the corrected phases of the selected PS using a space-time approach.

    This stage unwraps the corrected phases of the selected PS by applying
    a space-time unwrapping algorithm. The unwrapped phases are saved for
    further processing.
    """

    log("# Stage 6: Unwrapping phases of selected PS")

    small_baseline_flag = getparm("small_baseline_flag")
    unwrap_patch_phase = getparm("unwrap_patch_phase")
    scla_deramp = getparm("scla_deramp")
    subtr_tropo = getparm("subtr_tropo")
    aps_name = getparm("tropo_method")

    psver = get_psver()
    psname = f"ps{psver}"
    rcname = f"rc{psver}"
    pmname = f"pm{psver}"
    bpname = f"bp{psver}"
    goodname = f"phuw_good{psver}"

    if small_baseline_flag != "y":
        sclaname = f"scla_smooth{psver}"
        apsname = f"tca{psver}"
        phuwname = f"phuw{psver}"
    else:
        sclaname = f"scla_smooth_sb{psver}"
        apsname = f"tca_sb{psver}"
        phuwname = f"phuw_sb{psver}"

    ps = stamps_load(psname)

    assert isinstance(ps, dotdict)

    drop_ifg_index = getparm("drop_ifg_index")
    unwrap_ifg_index = np.setdiff1d(np.arange(ps.n_ifg), drop_ifg_index)

    if stamps_exists(bpname):
        bp = stamps_load(bpname)
    else:
        bperp = ps.bperp
        if small_baseline_flag != "y":
            bperp = np.delete(bperp, ps.master_ix)
        # bp = {"bperp_mat": np.tile(bperp, (ps.n_ps, 1))}
        bp = bperp

    assert isinstance(bp, np.ndarray)

    if small_baseline_flag != "y":
        bperp_mat = np.insert(bp, ps.master_ix, 0, axis=1)
    else:
        bperp_mat = bp

    if unwrap_patch_phase == "y":
        pm = stamps_load(pmname)
        assert isinstance(pm, dotdict)
        ph_w = pm.ph_patch / np.abs(pm.ph_patch)
        if small_baseline_flag != "y":
            ph_w = np.insert(ph_w, ps.master_ix, 1, axis=1)
    else:
        rc = stamps_load(rcname)
        assert isinstance(rc, dotdict)
        ph_w = rc.ph_rc
        if stamps_exists(pmname):
            pm = stamps_load(pmname)
            assert isinstance(pm, dotdict)
            if "K_ps" in pm and pm.K_ps is not None:
                ph_w *= np.exp(1j * (pm.K_ps[:, np.newaxis] * bperp_mat))

    ix = np.isfinite(ph_w) & (np.abs(ph_w) > 0)
    ph_w[ix] /= np.abs(ph_w[ix])  # normalize

    scla_subtracted_sw = 0
    ramp_subtracted_sw = 0

    options = {"master_day": ps.master_day}

    unwrap_hold_good_values = getparm("unwrap_hold_good_values")

    if small_baseline_flag != "y" or not stamps_exists(phuwname):
        unwrap_hold_good_values = "n"
        log("Code to hold good values skipped")

    if unwrap_hold_good_values == "y":
        sb_identify_good_pixels()
        options["ph_uw_predef"] = np.full(ph_w.shape, np.nan, dtype=np.float32)

        uw = stamps_load(phuwname)
        good = stamps_load(goodname)

        assert isinstance(uw, dotdict)
        assert isinstance(good, dotdict)

        if ps.n_ps == good.good_pixels.shape[0] and ps.n_ps == uw.ph_uw.shape[0]:
            options["ph_uw_predef"][good.good_pixels] = uw.ph_uw[good.good_pixels]
        else:
            log("Wrong number of PS in keep good pixels - skipped")

    if small_baseline_flag != "y" and stamps_exists(sclaname):  # PS
        log("subtracting scla and master aoe")

        scla = stamps_load(sclaname)
        assert isinstance(scla, dotdict)

        if scla.K_ps_uw.shape[0] == ps.n_ps:
            scla_subtracted_sw = 1  # FIXME: Change to bool

            # Subtract spatially correlated look angle error
            ph_w *= np.exp(-1j * scla.K_ps_uw[:, np.newaxis] * bperp_mat)

            # Subtract master APS
            ph_w *= np.exp(-1j * scla.C_ps_uw[:, np.newaxis] * np.ones_like(bperp_mat))

            if (
                scla_deramp == "y"
                and "ph_ramp" in scla
                and scla.ph_ramp.shape[0] == ps.n_ps
            ):
                ramp_subtracted_sw = 1  # FIXME: Change to bool

                # Subtract orbital ramps
                ph_w *= np.exp(-1j * scla.ph_ramp)
            else:
                log("   wrong number of PS in scla - subtraction skipped...")
                os.remove(sclaname + ".mat")  # FIXME: Check if this is correct

    if small_baseline_flag == "y" and os.path.exists(
        sclaname + ".mat"
    ):  # Small baselines
        log("   subtracting scla...")

        scla = stamps_load(sclaname)
        assert isinstance(scla, dotdict)

        if scla.K_ps_uw.shape[0] == ps.n_ps:
            scla_subtracted_sw = 1  # FIXME: Change to bool

            # Subtract spatially correlated look angle error
            ph_w *= np.exp(-1j * scla.K_ps_uw[:, np.newaxis] * bperp_mat)

            # Subtract spatially correlated look angle error
            if unwrap_hold_good_values == "y":
                options["ph_uw_predef"] -= scla.K_ps_uw[:, np.newaxis] * bperp_mat

            if (
                scla_deramp == "y"
                and "ph_ramp" in scla
                and scla.ph_ramp.shape[0] == ps.n_ps
            ):
                ramp_subtracted_sw = 1  # FIXME: Change to bool

                # Subtract orbital ramps
                ph_w *= np.exp(-1j * scla.ph_ramp)

                if unwrap_hold_good_values == "y":
                    options["ph_uw_predef"] -= scla.ph_ramp
        else:
            log("   wrong number of PS in scla - subtraction skipped...")
            os.remove(sclaname + ".mat")  # FIXME: Check if this is correct

    if stamps_exists(apsname) and subtr_tropo == "y":
        log("   subtracting slave aps...")

        aps = stamps_load(apsname)
        aps_corr, fig_name_tca, aps_flag = ps_plot_tca(aps, aps_name)

        ph_w *= np.exp(-1j * aps_corr)

        if unwrap_hold_good_values == "y":
            options["ph_uw_predef"] -= aps_corr

    options["time_win"] = int(getparm("unwrap_time_win"))
    options["grid_size"] = int(getparm("unwrap_grid_size"))
    options["prefilt_win"] = int(getparm("unwrap_gold_n_win"))

    options["unwrap_method"] = getparm("unwrap_method")
    options["goldfilt_flag"] = getparm("unwrap_prefilter_flag")
    options["la_flag"] = getparm("unwrap_la_error_flag")
    options["scf_flag"] = getparm("unwrap_spatial_cost_func_flag")

    options["gold_alpha"] = float(getparm("unwrap_gold_alpha"))

    max_topo_err = float(getparm("max_topo_err"))
    lambda_ = float(getparm("lambda"))

    rho = 830000  # mean range - need only be approximately correct

    if "mean_incidence" in ps:
        inc_mean = ps.mean_incidence
    else:
        laname = f"./la{psver}"
        if stamps_exists(laname):
            la = stamps_load(laname)
            assert isinstance(la, dotdict)
            # incidence angle approx equals look angle + 3 deg
            inc_mean = np.mean(la.la) + 0.052
        else:
            inc_mean = 21 * np.pi / 180  # guess the incidence angle

    max_K = max_topo_err / (lambda_ * rho * np.sin(inc_mean) / 4 / np.pi)

    bperp_range = np.max(ps.bperp) - np.min(ps.bperp)

    options["n_trial_wraps"] = int(bperp_range * max_K / (2 * np.pi))

    log(f'n_trial_wraps={options["n_trial_wraps"]}')

    if small_baseline_flag == "y":
        options["lowfilt_flag"] = "n"
        ifgday_ix = ps.ifgday_ix
        day = ps.day - ps.master_day
    else:
        options["lowfilt_flag"] = "n"
        mix = int(ps.master_ix)
        ifgday_ix = np.column_stack(
            (np.ones(ps.n_ifg) * mix, np.arange(ps.n_ifg))
        ).astype(np.int32)
        master_ix = np.sum(ps.master_day > ps.day)
        unwrap_ifg_index = np.setdiff1d(
            unwrap_ifg_index, master_ix
        )  # leave master ifg (which is only noise) out
        day = ps.day - ps.master_day

    if unwrap_hold_good_values == "y":
        options["ph_uw_predef"] = options["ph_uw_predef"][:, unwrap_ifg_index]

    if sys.platform.startswith("win"):
        log(
            "Windows detected: using old unwrapping code without statistical cost processing"
        )
        ph_uw_some = uw_nosnaphu(ph_w[:, unwrap_ifg_index], ps.xy, day, options)
    else:
        ph_uw_some, msd_some = uw_3d(
            ph_w[:, unwrap_ifg_index],
            ps.xy,
            day,
            ifgday_ix[unwrap_ifg_index, :],
            ps.bperp[unwrap_ifg_index],
            options,
        )

    ph_uw = np.zeros((ps.n_ps, ps.n_ifg), dtype=np.float32)
    msd = np.zeros((ps.n_ifg,), dtype=np.float32)

    ph_uw[:, unwrap_ifg_index] = ph_uw_some

    if "msd_some" in locals():  # FIXME?
        msd[unwrap_ifg_index] = msd_some

    if scla_subtracted_sw and small_baseline_flag != "y":
        log("Adding back SCLA and master AOE...")
        scla = stamps_load(sclaname)
        assert isinstance(scla, dotdict)

        # Add back spatially correlated look angle error
        ph_uw += scla.K_ps_uw[:, np.newaxis] * bperp_mat

        # Add back master APS
        ph_uw += scla.C_ps_uw[:, np.newaxis] * np.ones_like(bperp_mat)

        if ramp_subtracted_sw:
            # Add back orbital ramps
            ph_uw += scla.ph_ramp

    if scla_subtracted_sw and small_baseline_flag == "y":
        log("Adding back SCLA..")
        scla = stamps_load(sclaname)
        assert isinstance(scla, dotdict)

        # Add back spatially correlated look angle error
        ph_uw += scla.K_ps_uw[:, np.newaxis] * bperp_mat

        if ramp_subtracted_sw:
            # Add back orbital ramps
            ph_uw += scla.ph_ramp

    if stamps_exists(apsname) and subtr_tropo == "y":
        log("Adding back slave APS...")
        aps = stamps_load(apsname)
        aps_corr, fig_name_tca, aps_flag = ps_plot_tca(aps, aps_name)
        ph_uw += aps_corr

    if unwrap_patch_phase == "y":
        pm = stamps_load(pmname)
        rc = stamps_load(rcname)

        assert isinstance(pm, dotdict)
        assert isinstance(rc, dotdict)

        ph_w = pm.ph_patch / np.abs(pm.ph_patch)

        if small_baseline_flag != "y":
            ph_w = np.insert(ph_w, ps.master_ix - 1, 0, axis=1)

        ph_uw += np.angle(rc.ph_rc * np.conj(ph_w))

    ph_uw[:, np.setdiff1d(np.arange(ps.n_ifg), unwrap_ifg_index)] = 0

    stamps_save(
        phuwname,
        ph_uw=ph_uw,  # unwrapped phase data (n_ps, n_ifg) - radians
        msd=msd,  # mean squared differences (n_ifg,) - radians
    )


def sb_identify_good_pixels() -> None:
    raise NotImplementedError


def ps_plot_tca(aps, aps_name) -> Tuple[np.ndarray, str, str]:  # type: ignore
    raise NotImplementedError


def uw_nosnaphu(
    ph_w: Array, xy: Array, day: Array, options: Optional[dict] = None
) -> Array:
    raise NotImplementedError


def uw_3d(
    ph: Array,
    xy: Array,
    day: Array,
    ifgday_ix: Array,
    bperp: Optional[Array] = None,
    options: Optional[dict] = None,
) -> Tuple[Array, Array]:
    """
    Unwrap phase time series (single or multiple master).

    :param ph: N x M matrix of wrapped phase values (real phase or complex phasor)
               where N is number of pixels and M is number of interferograms.
    :param xy: N x 2 matrix of coordinates in meters.
    :param day: Vector of image acquisition dates in days, relative to master.
    :param ifgday_ix: M x 2 matrix giving index to master and slave date in DAY for each interferogram.
    :param bperp: M x 1 vector giving perpendicular baselines.
    :param options: Dictionary containing optional parameters.
    :return: Unwrapped phase and mean squared differences.
    """

    # Check for necessary parameters and set defaults
    if bperp is None:
        bperp = np.array([])

    if options is None:
        options = {}

    # Set default option values if not provided
    options.setdefault("master_day", 0)
    options.setdefault("grid_size", 5)
    options.setdefault("prefilt_win", 16)
    options.setdefault("time_win", 365)
    options.setdefault("goldfilt_flag", "n")
    options.setdefault("lowfilt_flag", "n")
    options.setdefault("gold_alpha", 0.8)
    options.setdefault("n_trial_wraps", 6)
    options.setdefault("la_flag", "y")
    options.setdefault("scf_flag", "y")
    options.setdefault("temp", None)
    options.setdefault("n_temp_wraps", 2)
    options.setdefault("max_bperp_for_temp_est", 100)
    options.setdefault("variance", [])
    options.setdefault("ph_uw_predef", None)

    # FIXME: This is horrible logic around unwrap_method in matlab code

    single_master_flag = len(np.unique(ifgday_ix[:, 0])) == 1

    if single_master_flag:
        options.setdefault("unwrap_method", "3D")
    else:
        options.setdefault("unwrap_method", "3D_FULL")

    if options["unwrap_method"] in ["3D", "3D_NEW"]:
        if single_master_flag:
            options["unwrap_method"] = "3D_FULL"
        else:
            options["lowfilt_flag"] = "y"

    for k, v in options.items():
        log(f"{k} = {v}")

    # Convert options to dotdict

    options = dotdict(options)

    # Ensure correct shape for input arrays
    if xy.shape[1] == 2:
        xy = np.hstack((np.arange(1, xy.shape[0] + 1).reshape(-1, 1), xy))

    day = np.array(day).flatten()

    # DEBUG: ph matches to 2 decimal places at this point

    uw_grid_wrapped(
        ph,
        xy,
        options["grid_size"],
        options["prefilt_win"],
        options["goldfilt_flag"],
        options["lowfilt_flag"],
        options["gold_alpha"],
        options["ph_uw_predef"],
    )

    del ph

    # if DEBUG:
    #    uw = stamps_load("uw_grid")
    #    check("uw_grid", uw, tol=1e-1)
    #    del uw

    uw_interp()

    uw_sb_unwrap_space_time(
        day,
        ifgday_ix,
        options.unwrap_method,
        options.time_win,
        options.la_flag,
        bperp,
        options.n_trial_wraps,
        options.prefilt_win,
        options.scf_flag,
        options.temp,
        options.n_temp_wraps,
        options.max_bperp_for_temp_est,
    )

    # check("uw_space_time", stamps_load("uw_space_time"), tol=1e-5)

    uw_stat_costs(options.unwrap_method, options.variance)

    ph_uw, msd = uw_unwrap_from_grid(xy, options.grid_size)

    return ph_uw, msd


def uw_stat_costs(
    unwrap_method: str = "3D",
    variance: Optional[Array] = None,
    subset_ifg_index: Optional[Array] = None,
) -> None:
    """Find unwrapped solutions using MAP cost functions"""

    costscale = 100
    nshortcycle = 200
    maxshort = 32000

    log("Unwrapping in space")

    log(f"{unwrap_method = }")
    log(f"{variance = }")
    log(f"{subset_ifg_index = }")

    uw = stamps_load("uw_grid")
    ui = stamps_load("uw_interp")
    ut = stamps_load("uw_space_time")

    assert isinstance(uw, dotdict)
    assert isinstance(ui, dotdict)
    assert isinstance(ut, dotdict)

    if variance is None:
        variance = np.array([], dtype=np.float32)

    if subset_ifg_index is None:
        subset_ifg_index = np.arange(uw.ph.shape[1], dtype=np.int32)

    predef_flag = "n"
    if hasattr(ut, "predef_ix"):
        predef_flag = "y"

    nrow, ncol = uw.nzix.shape

    y, x = np.where(uw.nzix)
    # nzix = np.flatnonzero(uw.nzix)
    # z = np.arange(uw.n_ps)

    Z = ui.Z

    grid_edges = np.concatenate(
        [ui.colix[np.abs(ui.colix) > 0], ui.rowix[np.abs(ui.rowix) > 0]]
    )
    # matlab `hist` uses bin centers, but np.histogram uses bin edges
    n_edges = np.histogram(np.abs(grid_edges), bins=np.arange(1, ui.n_edge + 2) - 0.5)[
        0
    ]

    check("n_edges", n_edges)

    if unwrap_method.upper() == "2D":
        raise NotImplementedError("This option has not been verified for accuracy")
        edge_length = np.sqrt(
            np.diff(x[ui.edgs[:, 1:3]], axis=1) ** 2
            + np.diff(y[ui.edgs[:, 1:3]], axis=1) ** 2
        )
        if uw.pix_size == 0:
            pix_size = 5  # if we don't know resolution
        else:
            pix_size = uw.pix_size
        if not variance:
            sigsq_noise = np.zeros(edge_length.shape)
        else:
            sigsq_noise = variance[ui.edgs[:, 1]] + variance[ui.edgs[:, 2]]
        sigsq_aps = (2 * np.pi) ** 2  # fixed for now as one fringe
        aps_range = 20000  # fixed for now as 20 km
        sigsq_noise = sigsq_noise + sigsq_aps * (
            1 - np.exp(-edge_length * pix_size * 3 / aps_range)
        )
        sigsq_noise = sigsq_noise / 10  # scale it to keep in reasonable range
        dph_smooth = ut.dph_space_uw

    else:
        # DEBUG: default path
        sigsq_noise = (np.std(ut.dph_noise, ddof=1, axis=1) / (2 * np.pi)) ** 2
        dph_smooth = ut.dph_space_uw - ut.dph_noise

    ut = dotdict({k: v for k, v in ut.items() if k != "dph_noise"})

    # Strange way of doing things, but this is what the matlab code does
    colix = ui.colix.astype(np.float32)  # uses 1-based indexing
    rowix = ui.rowix.astype(np.float32)  # uses 1-based indexing

    nostats_ix = np.flatnonzero(np.isnan(sigsq_noise))
    for i in nostats_ix:
        rowix[np.abs(rowix) == i + 1] = np.nan
        colix[np.abs(colix) == i + 1] = np.nan

    with np.errstate(invalid="ignore"):  # handle nans
        sigsq = np.round(
            (sigsq_noise * nshortcycle**2) / costscale * n_edges, 0
        ).astype(np.int16)
    sigsq[sigsq < 1] = 1  # zero causes snaphu to crash

    # check("sigsq_noise", sigsq_noise)
    # check("n_edges", n_edges)
    # check("sigsq", sigsq, tol=1e-6)

    rowcost = np.zeros(((nrow - 1), ncol * 4), dtype=np.int16)
    colcost = np.zeros((nrow, (ncol - 1) * 4), dtype=np.int16)

    nzrowix = np.abs(rowix) > 0
    rowstdgrid = np.ones(rowix.shape, dtype=np.int16)

    nzcolix = np.abs(colix) > 0
    colstdgrid = np.ones(colix.shape, dtype=np.int16)

    # check("nzrowix", nzrowix)
    # check("nzcolix", nzcolix)

    rowcost[:, 2::4] = maxshort
    colcost[:, 2::4] = maxshort

    stats_ix = ~np.isnan(rowix)
    rowcost[:, 3::4] = stats_ix * (-1 - maxshort) + 1

    stats_ix = ~np.isnan(colix)
    colcost[:, 3::4] = stats_ix * (-1 - maxshort) + 1

    ph_uw = np.zeros((uw.n_ps, uw.n_ifg), dtype=np.float64)
    ifguw = np.zeros((nrow, ncol))
    msd = np.zeros((uw.n_ifg,), dtype=np.float64)

    with open("snaphu.conf", "w") as fid:
        fid.write("INFILE  snaphu.in\n")
        fid.write("OUTFILE snaphu.out\n")
        fid.write("COSTINFILE snaphu.costinfile\n")
        fid.write("STATCOSTMODE  DEFO\n")
        fid.write("INFILEFORMAT  COMPLEX_DATA\n")
        fid.write("OUTFILEFORMAT FLOAT_DATA\n")

    for i1 in subset_ifg_index:
        log(f"Processing IFG {i1+1} of {len(subset_ifg_index)}")

        spread = np.ravel(ut.spread[:, i1].todense())
        spread = ((np.abs(spread) * nshortcycle**2) / 6 / costscale * n_edges).astype(
            np.int16
        )

        sigsqtot = sigsq + spread

        if ut.predef_ix is not None:
            sigsqtot[ut.predef_ix[:, i1]] = 1

        ix = np.abs(rowix[nzrowix]).astype(np.int32) - 1  # 0-based indexing
        rowstdgrid[nzrowix] = sigsqtot[ix]
        rowcost[:, 1::4] = rowstdgrid

        ix = np.abs(colix[nzcolix]).astype(np.int32) - 1  # 0-based indexing
        colstdgrid[nzcolix] = sigsqtot[ix]
        colcost[:, 1::4] = colstdgrid

        offset_cycle = (
            np.angle(np.exp(1j * ut.dph_space_uw[:, i1])) - dph_smooth[:, i1]
        ) / (2 * np.pi)

        ix = np.abs(rowix[nzrowix]).astype(np.int32) - 1  # 0-based indexing
        offgrid = np.zeros(rowix.shape, dtype=np.int16)
        offgrid[nzrowix] = np.round(
            offset_cycle[ix] * np.sign(rowix[nzrowix]) * nshortcycle, 0
        )
        rowcost[:, ::4] = -offgrid

        ix = np.abs(colix[nzcolix]).astype(np.int32) - 1  # 0-based indexing
        offgrid = np.zeros(colix.shape, dtype=np.int16)
        offgrid[nzcolix] = np.round(
            offset_cycle[ix] * np.sign(colix[nzcolix]) * nshortcycle, 0
        )
        colcost[:, ::4] = offgrid

        # check("offset_cycle", offset_cycle)
        # check("rowix", rowix)
        # check("colix", colix)
        # check("offgrid", offgrid)

        with open("snaphu.costinfile", "wb") as fid:
            rowcost.tofile(fid)
            colcost.tofile(fid)

        ifgw = uw.ph[Z - 1, i1].reshape(nrow, ncol)
        mask = np.isfinite(ifgw)
        ifgw[~mask] = 0

        # writecpx(Path("snaphu.in"), ifgw)
        ifgw.astype(np.complex64).tofile("snaphu.in")

        run_snaphu_on(Path("snaphu.conf"), ncol)

        with open("snaphu.out", "rb") as fid:
            ifguw = (
                np.fromfile(fid, dtype=np.float32)
                .astype(np.float64)
                .reshape(nrow, ncol)
            )

        check(f"ifguw_{i1+1}", ifguw, tol=1e-2)

        # FIXME: Still something wrong with the precision here when '!= 0' is used

        ifg_diff1 = np.ravel(ifguw[:-1, :] - ifguw[1:, :], "F")
        ifg_diff1 = ifg_diff1[ifg_diff1 != 0]

        ifg_diff2 = np.ravel(ifguw[:, :-1] - ifguw[:, 1:], "F")
        ifg_diff2 = ifg_diff2[ifg_diff2 != 0]

        msd[i1] = (np.sum(ifg_diff1**2) + np.sum(ifg_diff2**2)) / (
            len(ifg_diff1) + len(ifg_diff2)
        )

        ifguw_ur = np.ravel(ifguw, "F")
        nzix_ur = np.ravel(uw.nzix, "F")

        ph_uw[:, i1] = ifguw_ur[nzix_ur]

    # check("ph_uw", ph_uw, tol=1e-2)
    # check("msd", msd, tol=1e-1)

    stamps_save("uw_phaseuw", ph_uw=ph_uw, msd=msd)


def uw_unwrap_from_grid(xy: Array, pix_size: int) -> Tuple[Array, Array]:
    """Unwrap PS from unwrapped gridded ifgs"""

    log("Unwrapping from grid")

    uw = stamps_load("uw_grid")
    uu = stamps_load("uw_phaseuw")

    assert isinstance(uw, dotdict)
    assert isinstance(uu, dotdict)

    n_ps, n_ifg = uw.ph_in.shape

    gridix = np.zeros_like(uw.nzix, dtype=np.int32)
    gridix.T[uw.nzix.T] = np.arange(1, uw.n_ps + 1)

    ph_uw = np.zeros((n_ps, n_ifg), dtype=np.float32)

    for i in range(n_ps):
        ix = gridix[uw.grid_ij[i, 0] - 1, uw.grid_ij[i, 1] - 1]

        if ix == 0:
            # wrapped phase values were zero
            ph_uw[i, :] = np.nan
        else:
            ph_uw_pix = uu.ph_uw[int(ix) - 1, :]

            if np.isrealobj(uw.ph_in):
                ph_uw[i, :] = ph_uw_pix + np.angle(
                    np.exp(1j * (uw.ph_in[i, :] - ph_uw_pix))
                )
            else:
                ph_uw[i, :] = ph_uw_pix + np.angle(
                    uw.ph_in[i, :] * np.exp(-1j * ph_uw_pix)
                )

    if uw.ph_in_predef is not None and len(uw.ph_in_predef) > 0:
        predef_ix = ~np.isnan(uw.ph_in_predef)

        meandiff = np.nanmean(ph_uw - uw.ph_in_predef, axis=0)
        meandiff = 2 * np.pi * np.round(meandiff / (2 * np.pi))

        uw["ph_in_predef"] = uw.ph_in_predef + meandiff[np.newaxis, :]

        ph_uw[predef_ix] = uw.ph_in_predef[predef_ix]

    msd = uu.msd

    # check("ph_uw", ph_uw, tol=1e-2)
    # check("msd", msd, tol=1e-1)

    return ph_uw, msd


def uw_sb_unwrap_space_time(
    day: Array,
    ifgday_ix: Array,
    unwrap_method: str,
    time_win: float,
    la_flag: str,
    bperp: Array,
    n_trial_wraps: float,
    prefilt_win: int,
    scf_flag: str,
    temp: Optional[Array] = None,
    n_temp_wraps: Optional[float] = None,
    max_bperp_for_temp_est: Optional[float] = None,
) -> None:
    """
    Smooth and unwrap phase differences between neighboring data points in time.

    Parameters:
        day (array): Array of days.
        ifgday_ix (array): Array of interferogram day indices.
        unwrap_method (str): Unwrapping method.
        time_win (float): Time window for smoothing.
        la_flag (str): Flag for estimating look angle error.
        bperp (array): Array of perpendicular baselines.
        n_trial_wraps (float): Number of trial wraps.
        prefilt_win (int): Prefilter window size.
        scf_flag (str): Flag for using spatial cost function.
        temp (array, optional): Array of temperatures.
        n_temp_wraps (float, optional): Number of temperature wraps.
        max_bperp_for_temp_est (float, optional): Maximum perpendicular baseline for temperature estimation.
    """

    from scipy.sparse import csr_matrix
    from scipy.linalg import cholesky, inv, lstsq
    from scipy.interpolate import griddata

    log("Unwrapping in time-space...")

    uw = stamps_load("uw_grid")
    ui = stamps_load("uw_interp")

    if DEBUG:
        # uw.ph is not sufficient accuracy for comparison
        # so we need to load the original data
        uw = loadmat("uw_grid")
        ui = loadmat("uw_interp")

    assert isinstance(uw, dotdict)
    assert isinstance(ui, dotdict)

    n_image = day.shape[0]
    n_ifg = uw.n_ifg
    n_ps = uw.n_ps
    nzix = uw.nzix
    ij = uw.ij

    if uw.ph_uw_predef is None or uw.ph_uw_predef.size == 0:
        predef_flag = "n"
    else:
        predef_flag = "y"

    n_image = day.shape[0]
    nrow, ncol = ui.Z.shape

    # master_ix = np.where(day == 0)[0]  # FIXME: Can be removed?

    # day_pos_ix = np.where(day > 0)[0]
    # I = np.min(day[day_pos_ix])

    dph_space = uw.ph[ui.edgs[:, 2] - 1, :] * np.conj(uw.ph[ui.edgs[:, 1] - 1, :])

    # check("edgs", ui.edgs, tol=1e-6)
    # check("uw_ph", uw.ph, tol=1e-6)
    # check("dph_space", uw.ph[ui.edgs[:, 2] - 1, :], tol=1e-6)

    if predef_flag == "y":
        dph_space_uw = (
            uw.ph_uw_predef[ui.edgs[:, 2] - 1, :]
            - uw.ph_uw_predef[ui.edgs[:, 1] - 1, :]
        )
        predef_ix = ~np.isnan(dph_space_uw)
        dph_space_uw = dph_space_uw[predef_ix]
    else:
        predef_ix = np.array([])

    scale = np.abs(dph_space)
    good = scale > 0
    dph_space[good] = dph_space[good] / scale[good]

    # ifreq_ij = []
    # jfreq_ij = []

    G = np.zeros((n_ifg, n_image))
    for i in range(n_ifg):
        G[i, ifgday_ix[i, 0]] = -1
        G[i, ifgday_ix[i, 1]] = 1

    nzc_ix = np.sum(np.abs(G), axis=0) != 0
    day = day[nzc_ix]

    G = G[:, nzc_ix]

    zc_ix = np.where(nzc_ix == 0)[0]
    zc_ix = np.sort(zc_ix)[::-1]
    for i in range(len(zc_ix)):
        ifgday_ix[ifgday_ix > zc_ix[i]] = ifgday_ix[ifgday_ix > zc_ix[i]] - 1

    n = G.shape[1]

    if temp is not None:
        temp_flag = "y"
    else:
        temp_flag = "n"

    if temp_flag == "y":
        raise NotImplementedError("Not checked for correctness")

        log("Estimating temperature correlation")
        ix = np.abs(bperp) < max_bperp_for_temp_est
        temp_sub = temp[ix]
        temp_range = np.max(temp) - np.min(temp)
        temp_range_sub = np.max(temp_sub) - np.min(temp_sub)
        dph_sub = dph_space[:, ix]
        n_temp_wraps = n_temp_wraps * (temp_range_sub / temp_range)

        trial_mult = np.arange(
            -np.ceil(8 * n_temp_wraps), np.ceil(8 * n_temp_wraps) + 1
        )
        n_trials = len(trial_mult)
        trial_phase = temp_sub / temp_range_sub * np.pi / 4
        trial_phase_mat = np.exp(-1j * trial_phase[:, None] * trial_mult)

        Kt = np.zeros((ui.n_edge, 1), dtype=np.float32)
        coh = np.zeros((ui.n_edge, 1), dtype=np.float32)

        for i in range(ui.n_edge):
            cpxphase = dph_sub[i, :].flatten()
            cpxphase_mat = np.tile(cpxphase, (n_trials, 1)).T

            phaser = trial_phase_mat * cpxphase_mat
            phaser_sum = np.sum(phaser, axis=1)

            coh_trial = np.abs(phaser_sum) / np.sum(np.abs(cpxphase))

            coh_max_ix = np.argmax(coh_trial)
            coh_max = coh_trial[coh_max_ix]

            falling_ix = np.where(np.diff(coh_trial[: coh_max_ix + 1]) < 0)[0]
            if falling_ix.size > 0:
                peak_start_ix = falling_ix[-1] + 1
            else:
                peak_start_ix = 0

            rising_ix = np.where(np.diff(coh_trial[coh_max_ix:]) > 0)[0]
            if rising_ix.size > 0:
                peak_end_ix = rising_ix[0] + coh_max_ix
            else:
                peak_end_ix = n_trials - 1

            coh_trial[peak_start_ix : peak_end_ix + 1] = 0

            if coh_max - np.max(coh_trial) > 0.1:
                K0 = np.pi / 4 / temp_range_sub * trial_mult[coh_max_ix]
                resphase = cpxphase * np.exp(-1j * (K0 * temp_sub))
                offset_phase = np.sum(resphase)
                resphase = np.angle(resphase * np.conj(offset_phase))
                weighting = np.abs(cpxphase)
                mopt = lstsq(
                    weighting[:, None] * temp_sub[:, None], weighting * resphase
                )[0]
                Kt[i] = K0 + mopt
                phase_residual = cpxphase * np.exp(-1j * (Kt[i] * temp_sub))
                mean_phase_residual = np.nansum(phase_residual)
                coh[i] = np.abs(mean_phase_residual) / np.sum(np.abs(phase_residual))

        Kt[coh < 0.31] = 0  # FIXME: hardcoded value

        dph_space = dph_space * np.exp(-1j * Kt * temp[:, None])

        if predef_flag == "y":
            dph_temp = Kt * temp[:, None]
            dph_space_uw = dph_space_uw - dph_temp[predef_ix]

        dph_sub = dph_sub * np.exp(-1j * Kt * temp_sub[:, None])

    if la_flag == "y":
        # DEBUG: default case

        log("Estimating look angle error")

        bperp_range = np.max(bperp) - np.min(bperp)

        ix = np.where(np.abs(np.diff(ifgday_ix, axis=1)).flatten() == 1)[0]

        if len(ix) >= n_image - 1:
            raise NotImplementedError("Not checked for correctness")

            log("using sequential daisy chain of interferograms")

            dph_sub = dph_space[:, ix]
            bperp_sub = bperp[ix]
            bperp_range_sub = np.max(bperp_sub) - np.min(bperp_sub)
            n_trial_wraps = n_trial_wraps * (bperp_range_sub / bperp_range)

        else:
            # DEBUG: default case

            ifgs_per_image = np.sum(np.abs(G), axis=0).astype(np.int32)
            max_ifgs_per_image = np.max(ifgs_per_image)
            max_ix = np.argmax(ifgs_per_image)

            log(f"{max_ifgs_per_image = }")

            if max_ifgs_per_image >= n_image - 2:
                # DEBUG: default case
                log("Using sequential daisy chain of interferograms")

                # Find the interferograms that are connected to the master
                ix = G[:, max_ix] != 0

                gsub = G[ix, max_ix]
                sign_ix = -np.sign(gsub.flatten()).astype(np.int32)

                # Only use interferograms that are connected to the master
                dph_sub = dph_space[:, ix]

                bperp_sub = bperp[ix]
                bperp_sub[sign_ix == -1] = -bperp_sub[sign_ix == -1]
                bperp_sub = np.concatenate([bperp_sub, [0]])

                sign_ix = np.tile(sign_ix, (ui.n_edge, 1))

                # Flip the sign if necessary to make the ith ifg the master
                dph_sub[sign_ix == -1] = np.conj(dph_sub[sign_ix == -1])

                # Add zero phase master
                dph_sub = np.hstack(
                    (dph_sub, np.nanmean(np.abs(dph_sub), axis=1, keepdims=True))
                )

                slave_ix = np.sum(ifgday_ix[ix, :], axis=1) - max_ix

                # Extract the days for the subset
                day_sub = day[np.concatenate((slave_ix, [max_ix]))]

                # Sort the interferograms by day
                sort_ix = np.argsort(day_sub)
                day_sub = day_sub[sort_ix]
                dph_sub = dph_sub[:, sort_ix]
                bperp_sub = bperp_sub[sort_ix]
                bperp_sub = np.diff(bperp_sub)
                bperp_range_sub = np.max(bperp_sub) - np.min(bperp_sub)

                n_trial_wraps = n_trial_wraps * (bperp_range_sub / bperp_range)
                n_sub = len(day_sub)

                dph_sub = dph_sub[:, 1:] * np.conj(dph_sub[:, :-1])

                nbad = np.count_nonzero(np.isnan(dph_sub))
                log(f"{nbad = } values in {n_sub} interferograms")

                # Normalize the phase at valid pixels
                scale = np.abs(dph_sub)
                good = scale > 0
                dph_sub[good] = dph_sub[good] / scale[good]

            else:
                raise NotImplementedError("Not checked for correctness")

                dph_sub = dph_space
                bperp_sub = bperp
                bperp_range_sub = bperp_range

        # DEBUG: default case

        trial_mult = np.arange(
            -np.ceil(8 * n_trial_wraps), np.ceil(8 * n_trial_wraps) + 1
        )
        n_trials = len(trial_mult)

        trial_phase = bperp_sub / bperp_range_sub * np.pi / 4
        trial_phase_mat = np.exp(-1j * trial_phase[:, None] * trial_mult)

        K = np.zeros((ui.n_edge, 1), dtype=np.float32)
        coh = np.zeros((ui.n_edge, 1), dtype=np.float32)

        for i in range(ui.n_edge):
            cpxphase = dph_sub[i, :].flatten()
            cpxphase_mat = np.tile(cpxphase, (n_trials, 1)).T

            phaser = trial_phase_mat * cpxphase_mat
            phaser_sum = np.sum(phaser, axis=1)

            with np.errstate(divide="ignore", invalid="ignore"):
                coh_trial = np.abs(phaser_sum) / np.sum(np.abs(cpxphase))

            coh_max_ix = np.argmax(coh_trial)
            coh_max = coh_trial[coh_max_ix]

            falling_ix = np.where(np.diff(coh_trial[: coh_max_ix + 1]) < 0)[0]
            if falling_ix.size > 0:
                peak_start_ix = falling_ix[-1] + 1
            else:
                peak_start_ix = 0

            rising_ix = np.where(np.diff(coh_trial[coh_max_ix:]) > 0)[0]
            if rising_ix.size > 0:
                peak_end_ix = rising_ix[0] + coh_max_ix
            else:
                peak_end_ix = n_trials - 1

            coh_trial[peak_start_ix : peak_end_ix + 1] = 0

            if coh_max - np.max(coh_trial) > 0.1:  # FIXME: hardcoded value
                K0 = np.pi / 4 / bperp_range_sub * trial_mult[coh_max_ix]

                resphase = cpxphase * np.exp(-1j * (K0 * bperp_sub))
                offset_phase = np.sum(resphase)
                resphase = np.angle(resphase * np.conj(offset_phase))

                weighting = np.abs(cpxphase)
                mopt = lstsq(
                    weighting[:, None] * bperp_sub[:, None], weighting * resphase
                )[0]

                K[i] = K0 + mopt

                phase_residual = cpxphase * np.exp(-1j * (K[i] * bperp_sub))
                mean_phase_residual = np.nansum(phase_residual)

                coh[i] = np.abs(mean_phase_residual) / np.sum(np.abs(phase_residual))

        K[coh < 0.31] = 0  # FIXME: hardcoded value

        if temp_flag == "y":
            raise NotImplementedError("Not checked for correctness")

            dph_space[K == 0, :] = dph_space[K == 0, :] * np.exp(
                1j * Kt[K == 0] * temp[:, None]
            )
            Kt[K == 0] = 0
            K[Kt == 0] = 0

        dph_space = dph_space * np.exp(-1j * K @ bperp[:, np.newaxis].T)

        if predef_flag == "y":
            raise NotImplementedError("Not checked for correctness")

            dph_scla = K * bperp[:, None]
            dph_space_uw = dph_space_uw - dph_scla[predef_ix]

    spread = csr_matrix((ui.n_edge, n_ifg), dtype=np.float32)

    if unwrap_method == "2D":
        raise NotImplementedError("Not checked for correctness")

        dph_space_uw = np.angle(dph_space)

        if la_flag == "y":
            dph_space_uw = dph_space_uw + K * bperp[:, None]

        if temp_flag == "y":
            dph_space_uw = dph_space

        dph_noise = []

        stamps_save(
            "uw_space_time",
            dph_space_uw=dph_space_uw,
            spread=spread,
            dph_noise=dph_noise,
        )

    elif unwrap_method == "3D_NO_DEF":
        raise NotImplementedError("Not checked for correctness")

        dph_noise = np.angle(dph_space)
        dph_space_uw = np.angle(dph_space)

        if la_flag == "y":
            dph_space_uw = dph_space_uw + K * bperp[:, None]

        if temp_flag == "y":
            dph_space_uw = dph_space_uw + Kt * temp[:, None]

        stamps_save(
            "uw_space_time",
            dph_space_uw=dph_space_uw,
            dph_noise=dph_noise,
            spread=spread,
        )

    else:
        # DEBUG: default case

        log("Smoothing in time")

        if unwrap_method == "3D_FULL":  # FIXME: use an elif instead?
            # DEBUG: default case

            dph_smooth_ifg = np.full(dph_space.shape, np.nan, dtype=np.float64)

            for i in range(n_image):
                log(f"Smoothing in time: {i + 1}/{n_image}")

                ix = G[:, i] != 0
                if np.count_nonzero(ix) >= n_image - 2:
                    gsub = G[ix, i]
                    dph_sub = dph_space[:, ix]

                    sign_ix = -np.sign(gsub.flatten())
                    dph_sub[:, sign_ix == -1] = np.conj(dph_sub[:, sign_ix == -1])

                    slave_ix = np.sum(ifgday_ix[ix, :], axis=1) - i
                    day_sub = day[slave_ix]

                    sort_ix = np.argsort(day_sub)
                    day_sub = day_sub[sort_ix]
                    dph_sub = dph_sub[:, sort_ix]

                    dph_sub_angle = np.angle(dph_sub + 1e-9)
                    n_sub = day_sub.shape[0]

                    dph_smooth = np.zeros((ui.n_edge, n_sub), dtype=np.complex64)

                    for i1 in range(n_sub):
                        time_diff = (day_sub[i1] - day_sub).flatten()

                        # Exponential weighting
                        weight_factor = np.exp(-(time_diff**2) / (2 * time_win**2))
                        weight_factor = weight_factor / np.sum(weight_factor)

                        # Calculate the weighted mean phase
                        dph_mean = np.sum(
                            dph_sub * weight_factor[np.newaxis, :], axis=1
                        )
                        dph_mean_adj = (
                            np.mod(
                                dph_sub_angle
                                - np.angle(dph_mean)[:, np.newaxis]
                                + np.pi,
                                2 * np.pi,
                            )
                            - np.pi
                        )

                        GG = np.column_stack((np.ones(n_sub), time_diff))

                        if GG.shape[0] > 1:
                            m = lscov(GG, dph_mean_adj.T, weight_factor)
                        else:
                            m = np.zeros((GG.shape[1], ui.n_edge))

                        dph_smooth[:, i1] = dph_mean * np.exp(1j * m[0, :])

                        # GG matches to 6 decimal places at this point
                        # dph_mean matches to 6 decimal places at this point
                        # dph_mean_adj has problems as difference can be -pi vs pi

                        # check(
                        #    f"test_{i+1}_{i1+1}",
                        #    dph_mean,
                        #    tol=1e-6,
                        #    # modulo=2 * np.pi,
                        # )

                    check(f"dph_smooth_{i+1}", dph_smooth, tol=1e-6)

                    dph_smooth_sub = np.cumsum(
                        np.hstack(
                            (
                                np.angle(dph_smooth[:, :1]),
                                np.angle(
                                    dph_smooth[:, 1:] * np.conj(dph_smooth[:, :-1])
                                ),
                            )
                        ),
                        axis=1,
                    )

                    close_master_ix = np.where(slave_ix - i > 0)[0]

                    if close_master_ix.size == 0:
                        close_master_ix = np.array([n_sub - 1])
                    else:
                        close_master_ix = close_master_ix[0]
                        if close_master_ix > 0:
                            close_master_ix = np.array(
                                [close_master_ix - 1, close_master_ix]
                            )

                    close_master_ix = np.atleast_1d(close_master_ix)
                    dph_close_master = np.nanmean(
                        dph_smooth_sub[:, close_master_ix], axis=1
                    )

                    dph_smooth_sub = (
                        dph_smooth_sub
                        - (dph_close_master - np.angle(np.exp(1j * dph_close_master)))[
                            :, np.newaxis
                        ]
                    )
                    dph_smooth_sub = dph_smooth_sub * sign_ix

                    asix = np.where(~np.isnan(dph_smooth_ifg[0, ix]))[0]
                    ix = np.where(ix)[0]
                    aix = ix[asix]

                    std_noise1 = np.std(
                        np.angle(
                            dph_space[:, aix] * np.exp(-1j * dph_smooth_ifg[:, aix])
                        )
                    )
                    std_noise2 = np.std(
                        np.angle(
                            dph_space[:, aix] * np.exp(-1j * dph_smooth_sub[:, asix])
                        )
                    )

                    keep_ix = np.ones(n_sub, dtype=bool)
                    keep_ix[asix[std_noise1 < std_noise2]] = False

                    dph_smooth_ifg[:, ix[keep_ix]] = dph_smooth_sub[:, keep_ix]

            # DEBUG: default case

            check("dph_space", dph_space, tol=1e-6)
            check("dph_smooth_ifg", dph_smooth_ifg, tol=1e-5)

            dph_noise = np.angle(chop(dph_space * np.exp(-1j * dph_smooth_ifg)))

            check("dph_noise", dph_noise, tol=1e-5)

            # FIXME: hardcoded value
            mask = np.std(dph_noise, ddof=1, axis=1) > 1.2
            dph_noise[mask, :] = np.nan

        else:
            raise NotImplementedError("Not checked for correctness")

            # DEBUG: Are these all the cases?
            assert unwrap_method in ["3D_SMALL_DEF", "3D_QUICK", "3D_SMALL", "3D"]

            x = (day - day[0]) * (n - 1) / (day[-1] - day[0])

            if predef_flag == "y":
                raise NotImplementedError("Not checked for correctness")

                n_dph = dph_space.shape[0]
                dph_space_angle = np.angle(dph_space)
                dph_space_angle[predef_ix] = dph_space_uw
                dph_space_series = np.zeros((n, n_dph))
                for i in range(n_dph):
                    W = predef_ix[i, :] + 0.01
                    dph_space_series[1:, i] = lstsq(G[:, 1:], dph_space_angle[i, :], W)[
                        0
                    ]

            if predef_flag == "n":
                # DEBUG: default case
                sol = lstsq(G[:, 1:], np.angle(dph_space).T)[0]
                dph_space_series = np.vstack((np.zeros((1, ui.n_edge)), sol))

            dph_smooth_series = np.zeros((G.shape[1], ui.n_edge), dtype=np.float32)

            for i1 in range(n):
                time_diff_sq = (day[i1] - day) ** 2
                weight_factor = np.exp(-time_diff_sq / (2 * time_win**2))
                weight_factor = weight_factor / np.sum(weight_factor)
                dph_smooth_series[i1, :] = np.sum(
                    dph_space_series * np.tile(weight_factor, (1, ui.n_edge)), axis=0
                )

            dph_smooth_ifg = (G @ dph_smooth_series).T
            dph_noise = np.angle(dph_space * np.exp(-1j * dph_smooth_ifg))

            if unwrap_method in ["3D_SMALL_DEF", "3D_QUICK"]:
                raise NotImplementedError("Not checked for correctness")

                not_small_ix = np.where(np.std(dph_noise, axis=1) > 1.3)[0]
                log(f"{len(not_small_ix)} edges with high std dev in time")
                dph_noise[not_small_ix, :] = np.nan

            else:
                # DEBUG: Are these all the cases?
                assert unwrap_method in ["3D_SMALL", "3D"]

                uw = stamps_load("uw_grid")

                ph_noise = np.angle(uw["ph"] * np.conj(uw["ph_lowpass"]))

                dph_noise_sf = (
                    ph_noise[ui["edgs"][:, 2], :] - ph_noise[ui["edgs"][:, 1], :]
                )

                m_minmax = np.tile(np.array([[-np.pi, np.pi]]), (5, 1)) * np.tile(
                    np.array([[0.5], [0.25], [1], [0.25], [1]]), (1, 2)
                )

                anneal_opts = np.array([[1], [15], [0], [0], [0], [0], [0]])
                covm = np.cov(dph_noise_sf)

                try:
                    W = cholesky(inv(covm))
                    P = 0
                except np.linalg.LinAlgError:
                    W = np.diag(1 / np.sqrt(np.diag(covm)))
                    P = 1

                not_small_ix = np.where(np.std(dph_noise, axis=1) > 1)[0]

                log(f"Performing complex smoothing on {len(not_small_ix)} edges")

                n_proc = 0
                for i in not_small_ix:
                    dph = np.angle(dph_space[i, :])

                    dph_smooth_series[:, i] = uw_sb_smooth_unwrap(
                        m_minmax, anneal_opts, G, W, dph, x
                    )

                    n_proc += 1

                    if n_proc % 1000 == 0:
                        stamps_save(
                            "uw_unwrap_time",
                            G=G,
                            dph_space=dph_space,
                            dph_smooth_series=dph_smooth_series,
                        )

                        log(f"{n_proc} edges of {len(not_small_ix)} reprocessed")

                dph_smooth_ifg = (G @ dph_smooth_series).T
                dph_noise = np.angle(dph_space * np.exp(-1j * dph_smooth_ifg))

        # DEBUG: default case

        dph_space_uw = dph_smooth_ifg + dph_noise

        if la_flag == "y":
            dph_space_uw = dph_space_uw + K @ bperp[np.newaxis, :]

        if temp_flag == "y":
            raise NotImplementedError("Not checked for correctness")
            dph_space_uw = dph_space_uw + Kt * temp[:, np.newaxis]

        if scf_flag == "y":
            raise NotImplementedError("Not checked for correctness")

            log("Calculating local phase gradients")

            ifreq_ij = np.full((n_ps, n_ifg), np.nan, dtype=np.float32)
            jfreq_ij = np.full((n_ps, n_ifg), np.nan, dtype=np.float32)

            ifgw = np.zeros((nrow, ncol))
            uw = stamps_load("uw_grid")

            for i in range(n_ifg):
                ifgw[nzix] = uw["ph"][:, i]
                ifreq, jfreq, grad_ij, Hmag = gradient_filter(ifgw, prefilt_win)
                ix = (~np.isnan(ifreq)) & (Hmag / (np.abs(ifreq) + 1) > 3)

                if np.sum(ix) > 2:
                    ifreq_ij[:, i] = griddata(
                        grad_ij[ix, 1],
                        grad_ij[ix, 0],
                        ifreq[ix],
                        ij[:, 1],
                        ij[:, 0],
                        method="linear",
                    )

                ix = (~np.isnan(jfreq)) & (Hmag / (np.abs(jfreq) + 1) > 3)

                if np.sum(ix) > 2:
                    jfreq_ij[:, i] = griddata(
                        grad_ij[ix, 1],
                        grad_ij[ix, 0],
                        jfreq[ix],
                        ij[:, 1],
                        ij[:, 0],
                        method="linear",
                    )

            spread2 = np.zeros(spread.shape, dtype=np.float32)
            dph_smooth_uw2 = np.full((ui.n_edge, n_ifg), np.nan, dtype=np.float32)

            log("Smoothing using local phase gradients")

            for i in range(ui.n_edge):
                nodes_ix = ui["edgs"][i, 1:3]
                ifreq_edge = np.mean(ifreq_ij[nodes_ix, :], axis=0)
                jfreq_edge = np.mean(jfreq_ij[nodes_ix, :], axis=0)
                diff_i = np.diff(ij[nodes_ix, 0])
                diff_j = np.diff(ij[nodes_ix, 1])
                dph_smooth_uw2[i, :] = diff_i * ifreq_edge + diff_j * jfreq_edge

                spread2[i, :] = np.diff(ifreq_ij[nodes_ix, :], axis=0) + np.diff(
                    jfreq_ij[nodes_ix, :], axis=0
                )

            log("Choosing between time and phase gradient smoothing")

            std_noise = np.std(dph_noise, axis=1)
            dph_noise2 = np.angle(np.exp(1j * (dph_space_uw - dph_smooth_uw2)))
            std_noise2 = np.std(dph_noise2, axis=1)
            dph_noise2[std_noise2 > 1.3, :] = np.nan

            shaky_ix = np.isnan(std_noise) | (std_noise > std_noise2)

            log(
                f"{ui.n_edge - np.sum(shaky_ix)} arcs smoothed in time, {np.sum(shaky_ix)} in space"
            )

            dph_noise[shaky_ix, :] = dph_noise2[shaky_ix, :]
            dph_space_uw[shaky_ix, :] = (
                dph_smooth_uw2[shaky_ix, :] + dph_noise2[shaky_ix, :]
            )
            spread[shaky_ix, :] = spread2[shaky_ix, :]

        else:
            shaky_ix = np.array([], dtype=np.int32)

        check("dph_space_uw", dph_space_uw, tol=1e-6)

        stamps_save(
            "uw_space_time",
            dph_space_uw=dph_space_uw,
            dph_noise=dph_noise,
            G=G,
            spread=spread,
            # ifreq_ij=ifreq_ij,
            # jfreq_ij=jfreq_ij,
            shaky_ix=shaky_ix,
            predef_ix=predef_ix if len(predef_ix) > 0 else None,
        )


def uw_sb_smooth_unwrap(bounds, options, G, W, dph, x1):  # type: ignore
    raise NotImplementedError


def uw_grid_wrapped(
    ph_in: Array,
    xy_in: Array,
    pix_size: int = 200,
    prefilt_win: int = 32,
    goldfilt_flag: str = "y",
    lowfilt_flag: str = "y",
    gold_alpha: float = 0.8,
    ph_in_predef: Optional[Array] = None,
) -> None:
    """
    Resample unwrapped phase to a grid and filter.

    :param ph_in: N x M matrix of wrapped phase values.
    :param xy_in: N x 2 matrix of coordinates in meters.
    :param pix_size: Size of grid in m to resample data to.
    :param prefilt_win: Size of prefilter window in resampled grid cells.
    :param goldfilt_flag: Goldstein filtering flag ('y' or 'n').
    :param lowfilt_flag: Low pass filtering flag ('y' or 'n').
    :param gold_alpha: Alpha value for Goldstein filter.
    :param ph_in_predef: Predefined phase input (optional).
    """

    if ph_in is None or xy_in is None:
        raise ValueError("not enough arguments")

    n_ps, n_ifg = ph_in.shape

    log(f"Number of interferograms: {n_ifg}")
    log(f"Number of points per ifg: {n_ps}")

    if not np.isreal(ph_in).all() and np.sum(ph_in == 0) > 0:
        raise ValueError("Some phase values are zero")

    xy_in[:, 0] = np.arange(1, n_ps + 1)

    if pix_size == 0:
        grid_x_min = 1
        grid_y_min = 1
        n_i = np.max(xy_in[:, 2])  # seems weird? x and y inverted?
        n_j = np.max(xy_in[:, 1])
        grid_ij = xy_in[:, [2, 1]]
    else:
        grid_x_min = np.min(xy_in[:, 1])
        grid_y_min = np.min(xy_in[:, 2])

        grid_ij = np.zeros((n_ps, 2), dtype=int)
        grid_ij[:, 0] = np.ceil((xy_in[:, 2] - grid_y_min + 1e-3) / pix_size)
        grid_ij[grid_ij[:, 0] == np.max(grid_ij[:, 0]), 0] = np.max(grid_ij[:, 0]) - 1
        grid_ij[:, 1] = np.ceil((xy_in[:, 1] - grid_x_min + 1e-3) / pix_size)
        grid_ij[grid_ij[:, 1] == np.max(grid_ij[:, 1]), 1] = np.max(grid_ij[:, 1]) - 1

        n_i = np.max(grid_ij[:, 0])
        n_j = np.max(grid_ij[:, 1])

    grid_ij -= 1  # 0-based indexing

    log(f"Grid size: {n_i} x {n_j}")
    log(f"Grid pixel size: {pix_size} m")
    log(f"Grid origin: {grid_x_min} x {grid_y_min}")
    log(f"Prefilter window size: {prefilt_win}")

    log("Resampling phase to grid:")

    ph_grid = np.zeros((n_i, n_j), dtype=np.complex64)

    if ph_in_predef is not None:
        ph_grid_uw = np.zeros((n_i, n_j), dtype=np.complex64)
        N_grid_uw = np.zeros((n_i, n_j), dtype=np.float32)

    if np.min(ph_grid.shape) < prefilt_win:
        raise ValueError(
            f"Minimum dimension of the resampled grid ({np.min(ph_grid.shape)} pixels) is less than prefilter window size ({prefilt_win})"
        )

    for i1 in range(n_ifg):
        if np.isreal(ph_in).all():
            ph_this = np.exp(1j * ph_in[:, i1])
        else:
            # DEBUG: default case
            ph_this = ph_in[:, i1]

        if ph_in_predef is not None:
            # DEBUG: ignored by default
            ph_this_uw = ph_in_predef[:, i1]
            ph_grid_uw[:] = 0
            N_grid_uw[:] = 0

        ph_grid[:] = 0

        if pix_size == 0:
            ph_grid[(xy_in[:, 1] - 1) * n_i + xy_in[:, 2]] = ph_this
            if ph_in_predef is not None:
                ph_grid_uw[(xy_in[:, 1] - 1) * n_i + xy_in[:, 2]] = ph_this_uw

        else:
            # DEBUG: default case
            for i in range(n_ps):
                ph_grid[grid_ij[i, 0], grid_ij[i, 1]] += ph_this[i]

            if ph_in_predef is not None:
                # DEBUG: ignored by default
                for i in range(n_ps):
                    if not np.isnan(ph_this_uw[i]):
                        ph_grid_uw[grid_ij[i, 0], grid_ij[i, 1]] += ph_this_uw[i]
                        N_grid_uw[grid_ij[i, 0], grid_ij[i, 1]] += 1

                ph_grid_uw = ph_grid_uw / N_grid_uw

        if i1 == 0:
            # DEBUG: default case

            nzix = ph_grid != 0
            n_ps_grid = np.sum(nzix)

            ph = np.zeros((n_ps_grid, n_ifg), dtype=np.complex64)

            if lowfilt_flag.lower() == "y":
                ph_lowpass = ph
            else:
                # DEBUG: default case
                ph_lowpass = None

            if ph_in_predef is not None:
                ph_uw_predef = np.zeros((n_ps_grid, n_ifg), dtype=np.complex64)
            else:
                # DEBUG: default case
                ph_uw_predef = None

        if goldfilt_flag.lower() == "y" or lowfilt_flag.lower() == "y":
            # DEBUG: default case
            ph_this_gold, ph_this_low = wrap_filter(
                ph_grid, prefilt_win, gold_alpha, low_flag=lowfilt_flag
            )

            if lowfilt_flag.lower() == "y" and ph_lowpass is not None:
                ph_lowpass[:, i1] = ph_this_low[nzix]

        if goldfilt_flag.lower() == "y":
            # DEBUG: default case
            ph[:, i1] = ph_this_gold.T[nzix.T]  # Matlab ravels in column-major order
        else:
            ph[:, i1] = ph_grid.T[nzix.T]

        if ph_in_predef is not None and ph_uw_predef is not None:
            ph_uw_predef[:, i1] = ph_grid_uw[nzix]
            ix = ~np.isnan(ph_uw_predef[:, i1])
            ph_diff = np.angle(ph[ix, i1] * np.conj(np.exp(1j * ph_uw_predef[ix, i1])))
            ph_diff[np.abs(ph_diff) > 1] = np.nan
            ph_uw_predef[ix, i1] = ph_uw_predef[ix, i1] + ph_diff

        # check(f"ph_grid_{i1+1}", ph_grid, atol=1e-2, rtol=1e-2)
        log(
            f"{i1+1:{len(str(n_ifg))}d}/{n_ifg}: "
            f"nansum(abs(ph_grid)) = {np.nansum(np.abs(ph_grid)):.2f} "
            f"nansum(abs(ph)) = {np.nansum(np.abs(ph)):.2f}"
        )

    n_ps = n_ps_grid

    log(f"Number of resampled points: {n_ps}")

    check("ph_grid", ph_grid, tol=1e-2)

    nz_j, nz_i = np.where(ph_grid.T != 0)
    if pix_size == 0:
        xy = xy_in
    else:
        xy = np.column_stack(
            (
                np.arange(1, n_ps + 1),
                (nz_j + 1 - 0.5) * pix_size,
                (nz_i + 1 - 0.5) * pix_size,
            )
        )

    ij = np.column_stack((nz_i, nz_j))

    stamps_save(
        "uw_grid",
        ph=ph,
        ph_in=ph_in,
        ph_lowpass=ph_lowpass,
        ph_uw_predef=ph_uw_predef,
        ph_in_predef=ph_in_predef,
        xy=xy,
        ij=ij,
        nzix=nzix,
        grid_x_min=grid_x_min,
        grid_y_min=grid_y_min,
        n_i=n_i,
        n_j=n_j,
        n_ifg=n_ifg,
        n_ps=n_ps,
        grid_ij=grid_ij + 1,
        pix_size=pix_size,
    )


def wrap_filter(
    ph_in: Array,
    n_win: int,
    alpha: float,
    n_pad: Optional[int] = None,
    low_flag: str = "n",
) -> Tuple[Array, Array]:
    """
    Apply Goldstein adaptive and optional lowpass filtering to phase data.

    Parameters:
    - ph: 2D numpy array of phase values.
    - n_win: Size of the window for the filter.
    - alpha: Alpha parameter for the Goldstein filter.
    - n_pad: Padding size, default is 25% of n_win.
    - low_flag: Flag for performing lowpass filtering ('y' for yes, 'n' for no).

    Returns:
    - Filtered phase array. If low_flag is 'y', also returns low-pass filtered phase array.
    """

    ph = np.array(ph_in, dtype=np.complex64)

    # Set default padding if not provided
    if n_pad is None:
        n_pad = round(n_win * 0.25)

    # Initialize variables and compute increments for window processing
    n_i, n_j = ph.shape
    n_inc = n_win // 2
    n_win_i = int(np.ceil(n_i / n_inc) - 1)
    n_win_j = int(np.ceil(n_j / n_inc) - 1)

    # Initialize output arrays
    ph_out = np.zeros_like(ph, dtype=ph.dtype)

    if low_flag == "y":
        ph_out_low = np.zeros_like(ph)
    else:
        ph_out_low = np.array([])

    # Create the wind function for filtering
    x = np.arange(1, n_win / 2 + 1)
    X, Y = np.meshgrid(x, x)
    wind_func = np.block(
        [[X + Y, np.fliplr(X + Y)], [np.flipud(X + Y), np.flipud(np.fliplr(X + Y))]]
    )

    # Replace NaN values with 0 in the input phase array
    ph[np.isnan(ph)] = 0

    # Create Gaussian windows for filtering
    B = np.outer(gausswin(7), gausswin(7))
    L = ifftshift(np.outer(gausswin(n_win + n_pad, 16), gausswin(n_win + n_pad, 16)))

    ph_bit = np.zeros((n_win + n_pad, n_win + n_pad), dtype=ph.dtype)

    HH = np.zeros((n_win_i, n_win_j), dtype=np.complex64)

    for ix1 in range(n_win_i):
        wf = wind_func.copy()
        i1 = ix1 * n_inc + 1
        i2 = i1 + n_win - 1
        if i2 > n_i:  # Adjust the window if it exceeds the image bounds
            i_shift = i2 - n_i
            i2 = n_i
            i1 = n_i - n_win + 1
            wf = np.vstack((np.zeros((i_shift, n_win)), wf[: (n_win - i_shift), :]))

        for ix2 in range(n_win_j):
            wf2 = wf.copy()
            j1 = ix2 * n_inc + 1
            j2 = j1 + n_win - 1
            if j2 > n_j:  # Adjust the window for the horizontal dimension
                j_shift = j2 - n_j
                j2 = n_j
                j1 = n_j - n_win + 1
                wf2 = np.hstack(
                    (np.zeros((n_win, j_shift)), wf2[:, : (n_win - j_shift)])
                )

            # Initialize the phase bit for the current window
            ph_bit[:n_win, :n_win] = ph[(i1 - 1) : i2, (j1 - 1) : j2]

            # Apply FFT and filter the phase data
            ph_fft = np.fft.fft2(ph_bit)
            H = np.abs(ph_fft)
            H = ifftshift(
                convolve2d(fftshift(H), B, mode="same")
            )  # Smooth the frequency response

            medianH = np.median(H)
            if medianH != 0:
                H = (H / medianH) ** alpha
            else:
                H = H**alpha

            HH[ix1, ix2] = np.mean(np.abs(ph_bit))

            # Apply inverse FFT and window function
            ph_filt = np.fft.ifft2(ph_fft * H)

            ph_filt = ph_filt[:n_win, :n_win] * wf2

            # Optionally apply lowpass filtering
            if low_flag == "y":
                ph_filt_low = (
                    np.fft.ifft2(ph_fft * L)[:n_win, :n_win] * wf2
                )  # Lowpass filter
                ph_out_low[i1 - 1 : i2, j1 - 1 : j2] += ph_filt_low

            # Update the output array with filtered data
            ph_out[i1 - 1 : i2, j1 - 1 : j2] = (
                ph_out[i1 - 1 : i2, j1 - 1 : j2] + ph_filt
            )

    # Reset the magnitude of the output phase to match the input phase
    ph_out = abs(ph) * np.exp(1j * np.angle(ph_out))

    if low_flag == "y":
        ph_out_low = abs(ph) * np.exp(1j * np.angle(ph_out_low))  # Reset magnitude

    return ph_out, ph_out_low


def dsearchn(data: Array, query: Array) -> Tuple[Array, Array]:
    """
    Find nearest neighbors for query points in data set.
    """
    tree = KDTree(data, leafsize=10)
    dist, idx = tree.query(query, p=2, k=1)
    return np.array(idx), np.array(dist)


def uw_interp() -> None:
    """Interpolate grid using nearest neighbour."""

    log("Interpolating grid")

    # uw_grid_wrapped generated the file `uw_grid`
    uw = stamps_load("uw_grid")

    assert isinstance(uw, dotdict)
    n_ps = uw.n_ps
    nzix = uw.nzix

    nrow, ncol = nzix.shape
    II, JJ = np.meshgrid(np.arange(1, nrow + 1), np.arange(1, ncol + 1))
    PQ = np.column_stack((II.ravel(), JJ.ravel()))

    # Make i,j the indices of the non-zero indices which are the
    # pixel locations of the PS points in the grid (1-based indexing)
    jj, ii = np.where(nzix.T)
    ij = np.column_stack((np.arange(1, n_ps + 1), ii + 1, jj + 1))

    use_triangle = True
    if use_triangle:
        log("Using TRIANGLE to generate a Delaunay triangulation")

        nodepath = Path("unwrap.1.node")
        with open(nodepath, "w") as fid:
            fid.write(f"{n_ps} 2 0 0\n")
            np.savetxt(fid, ij, fmt="%d %d %d")

        run_triangle_on(nodepath)

        with open("unwrap.2.edge", "r") as fid:
            header = np.fromstring(fid.readline(), sep=" ", dtype=int)
            N = header[0]
            edgs = np.loadtxt(fid, dtype=int)

        n_edge = edgs.shape[0]
        if n_edge != N:
            raise ValueError("missing lines in unwrap.2.edge")

        with open("unwrap.2.ele", "r") as fid:
            header = np.fromstring(fid.readline(), sep=" ", dtype=int)
            N = header[0]
            ele = np.loadtxt(fid, dtype=int)

        n_ele = ele.shape[0]
        if n_ele != N:
            raise ValueError("missing lines in unwrap.2.ele")

        P = ij[:, 1:]
        # Z is the index (0-indexing) of the nearest point in P for each point in PQ
        Z, D = dsearchn(P, PQ)

        Z = Z.reshape(nrow, ncol)
        D = D.reshape(nrow, ncol)

    else:
        log("Using scipy.Delaunay for interpolation")
        raise NotImplementedError

        from scipy.spatial import Delaunay

        tri = Delaunay(ij[:, 1:])
        ele = tri.simplices
        edgs = tri.convex_hull

        n_edge = edgs.shape[0]
        edgs = np.column_stack(([np.arange(1, n_edge + 1), edgs]))
        n_ele = ele.shape[0]
        ele = np.column_stack(([np.arange(1, n_ele + 1), ele]))
        Z = tri.find_simplex(np.column_stack((II.ravel(), JJ.ravel())))

    if DEBUG:
        # Hard to get the same results as the original code as there could be
        # multiple solutions for the nearest neighbors
        Z = loadmat("Z")["Z"]

    # Identify the grid edges
    Zvec = np.ravel(Z.T)  # Column edges
    grid_edges = np.vstack((Zvec[:-nrow], Zvec[nrow:])).T
    Zvec = np.ravel(Z)  # Add the row edges
    grid_edges = np.vstack(
        (grid_edges, np.column_stack((Zvec[:-ncol], Zvec[ncol:])))
    )  # OK
    del Zvec

    # Sort each edge to have lowest pixel node first
    I_sort = np.argsort(grid_edges, axis=1)
    sort_edges = np.take_along_axis(grid_edges, I_sort, axis=1)
    edge_sign = I_sort[:, 1] - I_sort[:, 0]  # OK

    alledges, II, JJ = np.unique(
        sort_edges, axis=0, return_index=True, return_inverse=True
    )
    sameix = alledges[:, 0] == alledges[:, 1]
    alledges[sameix, :] = 0  # Set edges connecting identical nodes to (0,0)

    # (I+1) matches, (J+1) matches

    check("alledges", alledges)

    edgs, I2, J2 = np.unique(alledges, axis=0, return_index=True, return_inverse=True)
    # We drop (0,0) edge from the list, so we have (n_edge - 1) edges
    n_edge = I2.shape[0] - 1
    edgs = np.column_stack(([np.arange(1, n_edge + 1), edgs[1:, :]]))  # OK

    check("edgs", edgs)

    gridedgeix = J2[JJ] * edge_sign

    colix = gridedgeix[: nrow * (ncol - 1)].reshape(ncol - 1, nrow).T
    rowix = gridedgeix[nrow * (ncol - 1) :].reshape(nrow - 1, ncol)

    log(f"Number of unique edges in grid: {n_edge}")

    # check("gridedgeix", gridedgeix)
    # check("rowix", rowix)
    # check("colix", colix)

    stamps_save("uw_interp", edgs=edgs, n_edge=n_edge, rowix=rowix, colix=colix, Z=Z)

    log("Interpolation done")


def stage7_calc_scla(
    use_small_baselines: int = 0, coest_mean_vel: int = 0, opts: dotdict = dotdict()
) -> None:
    """Estimate spatially-correlated look angle error."""

    log("# Stage 7: Estimating spatially-correlated look angle error")

    small_baseline_flag = getparm("small_baseline_flag")
    drop_ifg_index = getparm("drop_ifg_index")
    scla_method = getparm("scla_method")
    scla_deramp = getparm("scla_deramp")
    subtr_tropo = getparm("subtr_tropo")
    tropo_method = getparm("tropo_method")

    if use_small_baselines != 0:
        raise NotImplementedError("Small baseline support not implemented")

        if small_baseline_flag != "y":
            raise ValueError("Use small baselines requested but there are none")

    if use_small_baselines == 0:
        scla_drop_index = getparm("scla_drop_index")
    else:
        scla_drop_index = getparm("sb_scla_drop_index")
        log("Using small baseline interferograms")

    psver = get_psver()

    psname = f"./ps{psver}"
    bpname = f"./bp{psver}"
    meanvname = f"./mv{psver}"
    ifgstdname = f"./ifgstd{psver}"
    phuwsbresname = f"./phuw_sb_res{psver}"

    if use_small_baselines == 0:
        phuwname = f"./phuw{psver}"
        sclaname = f"./scla{psver}"
        apsname_old = f"./aps{psver}"
        apsname = f"./tca{psver}"
    else:
        phuwname = f"./phuw_sb{psver}"
        sclaname = f"./scla_sb{psver}"
        apsname_old = f"./aps_sb{psver}"
        apsname = f"./tca_sb{psver}"

    if use_small_baselines == 0:
        Path(meanvname).unlink(missing_ok=True)

    ps = stamps_load(psname)
    assert isinstance(ps, dotdict)

    if Path(bpname).exists():
        bp = stamps_load(bpname)
    else:
        bperp = ps.bperp

        if small_baseline_flag != "y":
            bperp = np.delete(bperp, ps.master_ix - 1)

        bp = bperp[:, np.newaxis]

    assert isinstance(bp, np.ndarray)

    uw = stamps_load(phuwname)
    assert isinstance(uw, dotdict)

    if small_baseline_flag == "y" and use_small_baselines == 0:
        unwrap_ifg_index = np.arange(1, ps.n_image + 1)
    else:
        # DEBUG: default case
        unwrap_ifg_index = np.setdiff1d(np.arange(1, ps.n_ifg + 1), drop_ifg_index)

    if subtr_tropo == "y":
        raise NotImplementedError("Not checked for correctness")
        aps = stamps_load(apsname)
        aps_corr, fig_name_tca, tropo_method = ps_plot_tca(aps, tropo_method)
        uw["ph_uw"] += aps_corr

    if scla_deramp == "y":
        # DEBUG: default case
        log("Deramping ifgs")
        [ph_all, ph_ramp] = ps_deramp(ps, uw.ph_uw)
        uw["ph_uw"] -= ph_ramp
    else:
        ph_ramp = None

    unwrap_ifg_index = np.setdiff1d(unwrap_ifg_index, scla_drop_index)

    if Path(apsname_old).exists():
        if subtr_tropo == "y":
            log("You are removing atmosphere twice, do not do this.")
        aps = stamps_load(apsname_old)
        assert isinstance(aps, dotdict)
        uw["ph_uw"] -= aps.ph_aps_slave

    ref_ps = ps_setref()

    uw["ph_uw"] = uw.ph_iw - np.nanmean(uw.ph_uw[ref_ps, :], axis=0)[:, np.newaxis]

    assert isinstance(bp, dotdict)

    if use_small_baselines == 0:
        if small_baseline_flag == "y":
            bperp_mat = np.zeros((ps.n_ps, ps.n_image))
            G = np.zeros((ps.n_ifg, ps.n_image))
            for i in range(ps.n_ifg):
                G[i, ps.ifgday_ix[i, 0]] = -1
                G[i, ps.ifgday_ix[i, 1]] = 1
            if "unwrap_ifg_index_sm" in uw:
                unwrap_ifg_index = np.setdiff1d(uw.unwrap_ifg_index_sm, scla_drop_index)
            unwrap_ifg_index = np.setdiff1d(unwrap_ifg_index, ps.master_ix)

            G = G[:, unwrap_ifg_index]
            bperp_some = np.linalg.solve(G.T, bp.bperp_mat.T).T
            bperp_mat[:, unwrap_ifg_index] = bperp_some

        else:
            bperp_mat = bp.bperp_mat
            assert isinstance(bperp_mat, np.ndarray)
            bperp_mat = np.insert(bp.bperp_mat, ps.master_ix, 0, axis=1)

        day = np.diff(ps.day[unwrap_ifg_index])
        ph = np.diff(uw.ph_uw[:, unwrap_ifg_index], axis=1)
        bperp = np.diff(bperp_mat[:, unwrap_ifg_index], axis=1)

    else:
        # DEBUG: default case
        bperp_mat = bp.bperp_mat
        bperp = bperp_mat[:, unwrap_ifg_index]
        day = ps.ifgday[unwrap_ifg_index, 1] - ps.ifgday[unwrap_ifg_index, 0]
        ph = uw.ph_uw[:, unwrap_ifg_index]

    del bp

    bprint = np.mean(bperp)
    log(f"{ph.shape[1]} ifgs used in estimation:")

    for i in range(ph.shape[1]):
        if use_small_baselines != 0:
            log(
                f"{ps.ifgday[unwrap_ifg_index[i], 0]} to {ps.ifgday[unwrap_ifg_index[i], 1]} {day[i]} days {np.round(bprint[i])} m"
            )
        else:
            log(
                f"{ps.day[unwrap_ifg_index[i]]} to {ps.day[unwrap_ifg_index[i+1]]} {day[i]} days {np.round(bprint[i])} m"
            )

    K_ps_uw = np.zeros((ps.n_ps, 1))

    if coest_mean_vel == 0 or len(unwrap_ifg_index) < 4:
        G = np.column_stack((np.ones((ph.shape[1], 1)), np.mean(bperp)))
    else:
        G = np.column_stack((np.ones((ph.shape[1], 1)), np.mean(bperp), day))

    ifg_vcm = np.eye(ps.n_ifg)

    if small_baseline_flag == "y":
        if use_small_baselines == 0:
            phuwres = stamps_load(phuwsbresname)
            if "sm_cov" in phuwres:
                ifg_vcm = phuwres.sm_cov
        else:
            phuwres = stamps_load(phuwsbresname)
            if "sb_cov" in phuwres:
                ifg_vcm = phuwres.sb_cov
    else:
        if Path(ifgstdname).exists():
            ifgstd = stamps_load(ifgstdname)
            ifg_vcm = np.diag((ifgstd.ifg_std * np.pi / 180) ** 2)
            del ifgstd

    if use_small_baselines == 0:
        ifg_vcm_use = np.eye(ph.shape[1])
    else:
        ifg_vcm_use = ifg_vcm[unwrap_ifg_index - 1, unwrap_ifg_index - 1]

    m = lscov(G, ph.T, ifg_vcm_use)
    K_ps_uw = m[1, :]

    # if coest_mean_vel != 0:
    #    v_ps_uw = m[2, :]

    if scla_method == "L1":
        for i in range(ps.n_ps):
            d = ph[i, :]
            m2 = least_squares(lambda x: d - G @ x, m[:, i]).x
            K_ps_uw[i] = m2[1]

            if i % 10000 == 0:
                log(f"{i} of {ps.n_ps} pixels processed")

    ph_scla = np.tile(K_ps_uw[:, np.newaxis], (1, bperp_mat.shape[1])) * bperp_mat

    if use_small_baselines == 0:
        unwrap_ifg_index = np.setdiff1d(unwrap_ifg_index, ps.master_ix)

        if coest_mean_vel == 0:
            C_ps_uw = np.mean(
                uw.ph_uw[:, unwrap_ifg_index] - ph_scla[:, unwrap_ifg_index],
                axis=1,
            )

        else:
            G = np.column_stack(
                (
                    np.ones((len(unwrap_ifg_index), 1)),
                    ps.day[unwrap_ifg_index] - ps.day[ps.master_ix],
                )
            )
            m = lscov(
                G,
                (uw.ph_uw[:, unwrap_ifg_index] - ph_scla[:, unwrap_ifg_index]).T,
                ifg_vcm[unwrap_ifg_index - 1, unwrap_ifg_index - 1],
            )
            C_ps_uw = m[0, :]

    else:
        C_ps_uw = np.zeros((ps.n_ps, 1))

    oldscla = Path(".").glob(f"{sclaname}.mat")
    if len(oldscla) > 0 and oldscla[0].exists():
        olddatenum = datetime.fromtimestamp(os.path.getmtime(oldscla[0])).strftime(
            "%Y%m%d_%H%M%S"
        )
        import shutil

        shutil.move(oldscla[0], f"tmp_{sclaname[:-4]}_{olddatenum}.mat")

    stamps_save(sclaname, ph_scla, K_ps_uw, C_ps_uw, ph_ramp, ifg_vcm)


def ps_deramp(
    ps: dotdict, ph_all: Array, degree: Optional[int] = None
) -> Tuple[Array, Array]:
    """
    Deramps the data. Deramping is done by fitting a polynomial to the data and
    subtracting it from the original data.
    """

    if degree is None:
        try:
            degree = int(stamps_load("deramp_degree"))
            log("Deramping degree loaded from file `deramp_degree`")
        except (FileNotFoundError, IOError):
            degree = 1

    log(f"{degree = }")

    # SM from SB inversion deramping
    if ps.n_ifg != ph_all.shape[1]:
        ps["n_ifg"] = ph_all.shape[1]

    # detrenting of the data
    if degree == 1:
        # z = ax + by + c
        A = np.column_stack((ps.xy[:, 1:] / 1000, np.ones((ps.n_ps, 1))))
        log("**** z = ax + by + c")

    elif degree == 1.5:
        # z = ax + by + cxy + d
        A = np.column_stack(
            (
                ps.xy[:, 1:] / 1000,
                (ps.xy[:, 1] / 1000) * (ps.xy[:, 2] / 1000),
                np.ones((ps.n_ps, 1)),
            )
        )
        log("**** z = ax + by + cxy + d")

    elif degree == 2:
        # z = ax^2 + by^2 + cxy + d
        A = np.column_stack(
            (
                ((ps.xy[:, 1:] / 1000) ** 2),
                (ps.xy[:, 1] / 1000) * (ps.xy[:, 2] / 1000),
                np.ones((ps.n_ps, 1)),
            )
        )
        log("**** z = ax^2 + by^2 + cxy + d")

    elif degree == 3:
        # z = ax^3 + by^3 + cx^2y + dy^2x + ex^2 + fy^2 + gxy + h
        A = np.column_stack(
            (
                ((ps.xy[:, 1:] / 1000) ** 3),
                ((ps.xy[:, 1] / 1000) ** 2 * (ps.xy[:, 2] / 1000)),
                ((ps.xy[:, 2] / 1000) ** 2 * (ps.xy[:, 1] / 1000)),
                ((ps.xy[:, 1:] / 1000) ** 2),
                (ps.xy[:, 1] / 1000) * (ps.xy[:, 2] / 1000),
                np.ones((ps.n_ps, 1)),
            )
        )
        log("**** z = ax^3 + by^3 + cx^2y + dy^2x + ex^2 + fy^2 + gxy + h")

    else:
        raise ValueError("Invalid degree value. Expected 1, 1.5, 2, or 3.")

    ph_ramp = np.full(ph_all.shape, np.nan)
    for k in range(ps.n_ifg):
        ix = np.isnan(ph_all[:, k])
        if ps.n_ps - np.sum(ix) > 5:
            coeff = np.linalg.lstsq(A[~ix, :], ph_all[~ix, k], rcond=None)[0]
            ph_ramp[:, k] = np.dot(A, coeff)
            ph_all[:, k] -= ph_ramp[:, k]
        else:
            log(f"Ifg {k + 1} is not deramped")

    return ph_all, ph_ramp


def ps_setref(ps: Optional[dotdict] = None) -> Array:
    """
    Find reference PS.
    """

    log("Setting reference PS")

    psver = get_psver()

    ps_temp = stamps_load(f"ps{psver}")
    assert isinstance(ps_temp, dotdict)

    if ps is None:
        ps = ps_temp
    else:
        ps["ll0"] = ps_temp["ll0"]
        ps["n_ps"] = ps["lonlat"].shape[0]

    del ps_temp

    param = getparm("ref_x")
    if param != "":
        parmname = param.split(" ")[1]
    else:
        parmname = ""

    breakpoint()

    if parmname == "ref_x":
        ref_x = getparm("ref_x")
        ref_y = getparm("ref_y")
        ref_ps = np.where(
            (ps["xy"][:, 1] > ref_x[0])
            & (ps["xy"][:, 1] < ref_x[1])
            & (ps["xy"][:, 2] > ref_y[0])
            & (ps["xy"][:, 2] < ref_y[1])
        )[0]

    else:
        ref_lon = np.fromstring(getparm("ref_lon"), sep=" ")
        ref_lat = np.fromstring(getparm("ref_lat"), sep=" ")
        ref_centre_lonlat = np.fromstring(getparm("ref_centre_lonlat"), sep=" ")
        ref_radius = float(getparm("ref_radius"))

        print(f"{ref_lon = }")
        print(f"{ref_lat = }")
        print(f"{ref_centre_lonlat = }")
        print(f"{ref_radius = }")

        if ref_radius == -np.inf:
            ref_ps = np.array([0])

        else:
            ref_ps = np.where(
                (ps["lonlat"][:, 0] > ref_lon[0])
                & (ps["lonlat"][:, 0] < ref_lon[1])
                & (ps["lonlat"][:, 1] > ref_lat[0])
                & (ps["lonlat"][:, 1] < ref_lat[1])
            )[0]

            if ref_radius < np.inf:
                ref_xy = llh2local(ref_centre_lonlat.T, ps["ll0"]) * 1000
                xy = llh2local(ps["lonlat"][ref_ps, :].T, ps["ll0"]) * 1000
                dist_sq = (xy[0, :] - ref_xy[0]) ** 2 + (xy[1, :] - ref_xy[1]) ** 2
                ref_ps = ref_ps[dist_sq <= ref_radius**2]

    if len(ref_ps) == 0:
        if ps is not None:
            log(
                "None of your external data points have a reference, all are set as reference."
            )
            ref_ps = np.arange(ps.n_ps)

    if ps is None:
        if ref_ps == 0:
            log("No reference set")
        else:
            log(f"{len(ref_ps)} ref PS selected")

    log(f"Reference PS: {ref_ps}")

    return ref_ps


def lscov(A: Array, B: Array, w: Array) -> Array:
    """
    Solves the weighted least squares problem given by A*x = B with weights w.
    The weights are applied to both A and B by the square root of the weights.

    Parameters:
    A : ndarray
        A 2-D array with shape (m, n), where m is the number of observations
        and n is the number of variables.
    B : ndarray
        A 1-D or 2-D array with shape (m,) or (m, k), where m is the number
        of observations and k is the number of response variables.
    w : ndarray
        A 1-D array of weights with shape (m,), where m is the number of
        observations.
    """

    # Ensure w is a 1-D array and has the same length as the number of rows in A
    if w.ndim != 1 or len(w) != A.shape[0]:
        raise ValueError(
            "Weights w must be a 1-D array with the same length as the number of rows in A"
        )

    # Weight A and B by the square root of weights
    W = np.sqrt(np.diag(w))
    Aw = np.dot(W, A)
    Bw = np.dot(W, B)

    # Solve the least squares problem
    x, _, _, _ = np.linalg.lstsq(Aw, Bw, rcond=None)

    return np.array(x)


def ts_export_csv() -> None:
    """
    Export time series data to CSV files.
    """

    import csv

    psver = get_psver()

    psname = f"ps{psver}"
    phuwname = f"phuw{psver}"
    rcname = f"rc{psver}"
    hgtname = f"hgt{psver}"
    pmname = f"pm{psver}"
    sclaname = f"scla{psver}"

    hgt = loadmat(hgtname)

    if "ph_mm" not in locals():
        with open("ps_plot_ts_matname.txt", "r") as file:
            tsmat = file.read().strip()
        data = loadmat(tsmat)
        for key, value in data.items():
            if not key.startswith("__"):
                locals()[key] = value

    ijs = np.loadtxt("input_azrg", dtype=int)

    lijns = []
    with open("input_azrg_names", "r") as file:
        reader = csv.reader(file)
        for row in reader:
            lijns.append(row)
    ijns = np.array(lijns)

    ps = stamps_load(psname)
    assert isinstance(ps, dotdict)

    n_ifg = ps.n_ifg
    day = ps.day[0]
    master_day = ps.master_day
    dds = [
        datetime.strftime(master_day, "%Y%m%d") + "_" + datetime.strftime(el, "%Y%m%d")
        for el in day
    ]
    master_ix = sum(day < master_day) + 1

    ixs = np.isin(ijs, ps["ij"][:, 1:], assume_unique=True).all(axis=1)
    ijs = ijs[ixs]
    ijns = ijns[ixs]

    ixs = np.where(np.isin(ps["ij"][:, 1:], ijs, assume_unique=True).all(axis=1))[0]
    lls = ps["lonlat"][ixs]
    hgt = hgt["hgt"][0][ixs]

    phuw = stamps_load(phuwname)
    uw = phuw["ph_uw"][ixs].T

    pm = stamps_load(pmname)
    ph_all = pm["ph_patch"] / np.abs(pm["ph_patch"])
    if n_ifg != ph_all.shape[1]:
        ph_all = np.hstack(
            (
                ph_all[:, : ps["master_ix"][0][0] - 1],
                np.zeros((ps["n_ps"][0][0], 1)),
                ph_all[:, ps["master_ix"][0][0] :],
            )
        )
    p = ph_all[ixs].T

    phuw = stamps_load(phuwname)
    scla = stamps_load(sclaname)
    m = scla["C_ps_uw"][ixs]
    u_d = phuw["ph_uw"] - scla["ph_scla"]
    u_d = u_d[ixs].T

    phuw = stamps_load(phuwname)
    scla = stamps_load(sclaname)
    ph_all = phuw["ph_uw"]
    ph_all = phuw["ph_uw"] - scla["C_ps_uw"][np.newaxis, :]
    ph_all[:, master_ix - 1] = 0
    u_m = ph_all[ixs].T

    phuw = stamps_load(phuwname)
    ph_all = phuw["ph_uw"]
    ph_all = ps_deramp(ps, ph_all)
    u_o = ph_all[ixs].T

    phuw = stamps_load(phuwname)
    scla = stamps_load(sclaname)
    ph_all = phuw["ph_uw"]
    ph_all = phuw["ph_uw"] - scla["C_ps_uw"][np.newaxis, :] - scla["ph_scla"]
    ph_all = ps_deramp(ps, ph_all)
    ph_all[:, master_ix - 1] = 0
    u_dmo = ph_all[ixs].T

    nlls = lls.shape[0]
    log(f"{nlls} lon/lats")

    for i in range(nlls):
        lon0 = lls[i, 0]
        lat0 = lls[i, 1]
        name = ijns[i, 2]
        log(f"{lon0} {lat0} {name}")

        data = dotdict(
            {
                "date": dds,
                "u": uw[:, i],
                "p": p[:, i],
                "u-d": u_d[:, i],
                "u-m": u_m[:, i],
                "u-o": u_o[:, i],
                "u-dmo": u_dmo[:, i],
            }
        )

        with open(f"{i+1}.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data.keys())
            writer.writerows(zip(*data.values()))

    data = dotdict(
        {
            "Index": np.arange(1, nlls + 1),
            "Master_AOE": m,
            "Lon": lls[:, 0],
            "Lat": lls[:, 1],
            "Height": hgt,
            "Azimuth_Line": ijns[:, 0].astype(int),
            "Range_Sample": ijns[:, 1].astype(int),
            "Name": ijns[:, 2],
        }
    )

    with open("ts.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data.keys())
        writer.writerows(zip(*data.values()))


def print_accuracy(*args: str, tol: float = 1e-6) -> None:
    """
    Print the accuracy of the results.
    """

    log("# Difference of results compared to MATLAB implementation")

    table: dict[str, list] = {
        "name": [],
        "diff": [],
        "ndiff": [],
        "p.shape": [],
        "m.shape": [],
        "p.type:": [],
        "m.type:": [],
    }

    not_found = []

    for arg in args:
        try:
            p = stamps_load(arg, squeeze=False)
            m = loadmat(arg)

            if len(p) == 1 and len(m) == 1:
                nk = str(next(iter(m)))
                pk = next(iter(p))
                assert isinstance(p[pk], np.ndarray)
                p = dotdict({nk: p[pk]})

            if isinstance(p, dict) and isinstance(m, dict):
                for key in p:
                    if key in m:
                        table["name"].append(f"{arg}.{key}")
                        if hasattr(p[key], "shape") and hasattr(m[key], "shape"):
                            table["p.shape"].append(p[key].shape)
                            table["m.shape"].append(m[key].shape)
                            table["p.type:"].append(p[key].dtype.str)
                            table["m.type:"].append(m[key].dtype.str)
                        else:
                            table["p.shape"].append(1)
                            table["m.shape"].append(1)
                            table["p.type:"].append(type(p[key]).__name__)
                            table["m.type:"].append(type(m[key]).__name__)
                        try:
                            diff = np.abs(p[key] - m[key])
                            table["diff"].append(np.nanmax(diff))
                        except (TypeError, ValueError):
                            table["diff"].append("∞")
                        if isinstance(diff, np.ndarray):
                            table["ndiff"].append(np.sum(diff > tol))
                        else:
                            table["ndiff"].append("-")
        except FileNotFoundError:
            not_found.append(arg)

    tabulate(table)

    if len(not_found) > 0:
        log("")
        log("ERROR, Files not found:")
        for nf in not_found:
            log(f"  {nf}")


def test_getparm() -> None:
    """Test the getparm function."""
    log("Testing getparm function")
    getparm()


def check_results() -> None:
    for p in patchdirs():
        with chdir(p):
            print_accuracy(
                # Stage 1
                "ps1",
                "ph1",
                "bp1",
                "la1",
                "da1",
                "hgt1",
                # Stage 2
                "pm1",
                # Stage 3
                "select1",
                # Stage 4
                "weed1",
                "pm2",
                "ps2",
                "hgt2",
                "la2",
                # Stage 5
                "rc1",
                # Stage 6
                "phuw1",
            )


def test_stage1() -> None:
    log("Testing Stage 1")
    for p in patchdirs():
        with chdir(p):
            stage1_load_data()
            assert results_equal("ps1")
            assert results_equal("ph1")
            assert results_equal("bp1")
            assert results_equal("la1")
            assert results_equal("da1")
            assert results_equal("hgt1")

    log("Stage 1 test passed")


def test_stage2() -> None:
    log("Testing Stage 2")
    for p in patchdirs():
        with chdir(p):
            stage2_estimate_noise()
            assert results_equal("pm1", tol=1e-2)

    log("Stage 2 test passed")


def test_stage3() -> None:
    log("Testing Stage 3")
    for p in patchdirs():
        with chdir(p):
            stage3_select_ps()
            assert results_equal("select1", tol=1e-2)

    log("Stage 3 test passed")


def test_stage4() -> None:
    log("Testing Stage 4")
    for p in patchdirs():
        with chdir(p):
            stage4_weed_ps()
            assert results_equal("weed1", tol=1e-2)
            assert results_equal("pm2", tol=1e-2)
            assert results_equal("ps2", tol=1e-2)
            assert results_equal("hgt2", tol=1e-2)
            assert results_equal("la2", tol=1e-2)

    log("Stage 4 test passed")


def test_stage5() -> None:
    log("Testing Stage 5")
    for p in patchdirs():
        with chdir(p):
            stage5_correct_phases()
            assert results_equal("rc1", tol=1e-2)

    log("Stage 5 test passed")


def test_stage6() -> None:
    log("Testing Stage 6")
    for p in patchdirs():
        with chdir(p):
            stage6_unwrap_phases()
            assert results_equal("phuw1", tol=1e-1)

    log("Stage 6 test passed")


def test_ps_calc_ifg_std() -> None:
    log("Testing ps_calc_ifg_std function")
    for p in patchdirs():
        with chdir(p):
            ps_calc_ifg_std()
            assert results_equal("ifgstd1", tol=1e-2)


def test_uw_interp() -> None:
    log("Testing uw_interp function")
    for p in patchdirs():
        with chdir(p):
            uw_interp()
            assert results_equal("uw_interp", tol=1e-2)


def test_interp() -> None:
    log("Testing interp function 1")
    x = np.arange(1, 10, dtype=np.float64)
    y = interp(x, 2)
    assert np.allclose(
        y,
        np.fromstring(
            "1.0000, 1.4996, 2.0000, 2.4993, 3.0000,"
            "3.4990, 4.0000, 4.4987, 5.0000, 5.4984,"
            "6.0000, 6.4982, 7.0000, 7.4979, 8.0000,"
            "8.4976, 9.0000, 9.4973",
            sep=",",
        ),
        atol=1e-4,
        rtol=1e-4,
    )
    log("Testing interp function 2")
    x = np.ones(100)
    y = interp(x, 10)
    assert np.allclose(y, np.ones(1000), atol=1e-2)
    log("Testing interp function 3")
    t = np.linspace(0, 1, endpoint=True, num=1001)
    x = np.sin(2 * np.pi * 30 * t) + np.sin(2 * np.pi * 60 * t)
    y = interp(x, 4)
    yy = loadmat("interp")["y"]
    assert np.allclose(y, yy, atol=1e-2)
    log("Testing interp function 4")
    cwd = Path.cwd()
    os.chdir(patchdirs()[0])
    x = loadmat("Prand_1")["Prand"]
    y = interp(np.insert(x, 0, 1), 10)[:-9]
    z = loadmat("Prand_after_1")["Prand"]
    assert np.allclose(y, z, atol=1e-2)
    os.chdir(cwd)
    log("Testing interp function 5")
    cwd = Path.cwd()
    os.chdir(patchdirs()[0])
    x = loadmat("Prand_2")["Prand"]
    y = interp(np.insert(x, 0, 1), 10)[:-9]
    z = loadmat("Prand_after_2")["Prand"]
    assert np.allclose(y, z, atol=1e-2)
    os.chdir(cwd)


def test_dates() -> None:
    pscname = Path("pscphase.in")
    with pscname.open() as f:
        ifgs = [Path(line.strip()) for line in f.readlines()][1:]

    origstrs = [f"{ifg.name[9:13]}-{ifg.name[13:15]}-{ifg.name[15:17]}" for ifg in ifgs]
    datenums = datenum(np.array(origstrs, dtype="datetime64"))
    datestrs = datestr(datenums)

    x = np.array(origstrs, dtype="datetime64")
    y = np.array(datestrs, dtype="datetime64")
    assert (x - y).sum() == 0


def test_params() -> None:
    import tomllib

    fn = Path("test.toml")

    with open(fn, "w") as f:
        print_all_parameters(f)

    toml = fn.read_text()

    try:
        tomllib.loads(toml)
    except tomllib.TOMLDecodeError as e:
        raise AssertionError(f"Error decoding toml file: {e}")
    finally:
        fn.unlink()


def run_tests() -> None:
    test_params()
    test_dates()
    test_interp()
    test_stage1()
    test_stage2()
    test_stage3()
    test_stage4()
    test_stage5()
    test_stage6()
    test_ps_calc_ifg_std()
    test_uw_interp()
    log("\nAll tests passed!\n")


def run_all_stages(opts: dotdict = dotdict()) -> None:
    """Run all stages."""
    for i in range(8):
        run_stage(i, opts=opts)
    log("\nAll stages completed!\n")


def run_stage(i: int, opts: dotdict = dotdict()) -> None:
    """Run a specific stage."""
    if i == 0:
        cwd = Path.cwd()
        log(f"Running stage {i} in {cwd}")
        stage0_preprocess(opts=opts)
    else:
        for p in patchdirs():
            log(f"Running stage {i} in {p}")
            with chdir(p):
                if i == 1:
                    stage1_load_data(opts=opts)
                elif i == 2:
                    stage2_estimate_noise(opts=opts)
                elif i == 3:
                    stage3_select_ps(opts=opts)
                elif i == 4:
                    stage4_weed_ps(opts=opts)
                elif i == 5:
                    stage5_correct_phases(opts=opts)
                elif i == 6:
                    stage6_unwrap_phases(opts=opts)
                elif i == 7:
                    stage7_calc_scla(opts=opts)

    log(f"\nStage {i} complete!\n")


def parse_human_size(size: str) -> int:
    """Parse a human-readable size string."""
    size = size.upper()
    if size.endswith("KB"):
        return int(float(size[:-2]) * 1024)
    if size.endswith("MB"):
        return int(float(size[:-2]) * 1024**2)
    if size.endswith("GB"):
        return int(float(size[:-2]) * 1024**3)
    if size.endswith("TB"):
        return int(float(size[:-2]) * 1024**4)
    if size.endswith("B"):
        return int(size[:-1])
    raise RuntimeError(f"Invalid size string `{size}`")


def human_size(size: int) -> str:
    """Convert a size in bytes to a human-readable string."""
    x = float(size)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if x < 1024:
            return f"{x:.2f}{unit}"
        x /= 1024
    return f"{x:.2f}PB"


def max_memory_used() -> int:
    """Return the maximum memory used by the process."""
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def limit_memory(maxmem: int | str) -> None:
    """Limit the maximum memory usage."""
    import resource
    import atexit

    # If not running on Linux, do nothing

    if isinstance(maxmem, str):
        maxmem = parse_human_size(maxmem)

    if sys.platform != "linux" or maxmem < 0:
        return

    curmem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    curmax = resource.getrlimit(resource.RLIMIT_AS)[1]

    log(
        f"Limiting memory usage from {human_size(curmax)} to {human_size(maxmem)}. Current usage: {human_size(curmem)}"
    )

    resource.setrlimit(resource.RLIMIT_AS, (maxmem, maxmem))

    # At exit, print out the maximum memory used

    atexit.register(
        lambda: log(f"Maximum memory used: {human_size(max_memory_used())}")
    )


def setup_logging(logging_config: Optional[Path] = None) -> None:
    """Setup to use the `logging` module."""

    import logging
    import logging.config

    global show_progress
    global log

    logging.basicConfig(
        stream=sys.stdout,
        format="%(asctime)s | %(message)s [%(levelname)s]",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=(logging.DEBUG if VERBOSE else logging.INFO),
    )

    if logging_config:
        logging.config.fileConfig(logging_config)

    # We disable the progress bar when using the `logging` module

    def show_progress(*args: List[Any], **kwargs: Dict[Any, Any]) -> None:
        pass

    # We use the `logging` module for logging

    def log(msg: str) -> None:
        if msg.startswith("#"):
            logging.info(msg[1:].lstrip())
        else:
            logging.debug(msg.strip())

    # We setup a global exception handler that logs the exception

    def excepthook(exc_type, exc_value, exc_traceback):  # type: ignore
        # Walk up the stack to the original exception
        while exc_traceback.tb_next:
            exc_traceback = exc_traceback.tb_next

        # Nicely log the exception
        logging.error(
            f"{exc_value} ({exc_type.__name__} at line {exc_traceback.tb_lineno})"
        )

    sys.excepthook = excepthook

    if logging_config:
        log(f"Using logging configuration file: {logging_config}")
    else:
        log(
            "Using default logging configuration as no configfile provided using `--logconfig`"
        )


def load_and_normalise_config(
    args: Any, config: Optional[Path] = None, other: Optional[List] = None
) -> dotdict:
    """Load the configuration file from a toml file and combine with the command line arguments."""
    import tomllib as toml

    options = dotdict()

    if config:
        try:
            log(f"Using configuration file: `{config}`")

            with open(config, "rb") as f:
                opts = toml.load(f)
                if DEBUG and len(opts) > 0:
                    log(
                        f"Options from config file: {", ".join([k for k in opts.keys()])}"
                    )
                    options.update(opts)

        except toml.TOMLDecodeError as e:
            raise RuntimeError(f"Problem loading configuration file `{config}`: {e}")

    else:
        # Parse the default options from DEFAULT_OPTIONS
        opts = toml.loads(DEFAULT_OPTIONS)
        if DEBUG:
            log(f"Options from default options: {", ".join([k for k in opts.keys()])}")
        options.update(opts)

    # We update the configuration with the command line arguments

    opts = vars(args).copy()

    # We remove the global options that are not relevant for the model science

    for opt in [
        "config",
        "logging",
        "logconfig",
        "test",
        "check",
        "run",
        "snaphu",
        "triangle",
        "nofancy",
        "quiet",
        "debug",
        "processor",
        "params",
    ]:
        if opt in opts:
            del opts[opt]

    if DEBUG and len(opts) > 0:
        log(f"Options from command line: {", ".join([k for k in opts.keys()])}")

    options.update(opts)

    # We update the configuration with the other options

    if other:
        # For pairs of options of the form ['--foo', 'bar'], we add the
        # key-value pair to the configuration dictionary

        for i in range(0, len(other), 2):
            key = other[i][2:]
            value = other[i + 1]

            # Try to parse using various types

            for parse in [int, float, str]:
                try:
                    value = parse(value)
                    break
                except ValueError:
                    pass

            options[key] = value

    # If an option is a Path object, resolve it

    for k, v in options.items():
        if isinstance(v, Path):
            resolved = v.resolve()
            if DEBUG and v != resolved:
                log(f"Resolved path `{v}` to `{resolved}`")
            options[k] = resolved

    # Print the final options if debug is enabled

    if DEBUG:
        for k, v in options.items():
            if isinstance(v, str):
                vv = f"'{v}'"
            else:
                vv = v
            log(f"Option: {k} = {vv} ({type(v).__name__})")

    return options


def print_all_parameters(file: TextIO = sys.stdout) -> None:
    """Print all the model parameters."""

    def get_and_print(p: str) -> None:
        v = getparm(p)

        if v.startswith("[") and v.endswith("]"):
            a = np.fromstring(v[1:-1], sep=",")
            print(f"{p} = {a}", file=file)
            return

        for parse in [int, float, str]:
            try:
                v = parse(v)
                break
            except ValueError:
                pass

        if isinstance(v, str):
            v = f"'{v}'"

        print(f"{p} = {v}", file=file)

    get_and_print("clap_alpha")
    get_and_print("clap_beta")
    get_and_print("clap_low_pass_wavelength")
    get_and_print("clap_win")
    get_and_print("density_rand")
    get_and_print("drop_ifg_index")
    get_and_print("filter_grid_size")
    get_and_print("filter_weighting")
    get_and_print("gamma_change_convergence")
    get_and_print("gamma_max_iterations")
    get_and_print("gamma_stdev_reject")
    get_and_print("lambda")
    get_and_print("max_topo_err")
    get_and_print("percent_rand")
    get_and_print("ref_centre_lonlat")
    get_and_print("ref_lat")
    get_and_print("ref_lon")
    get_and_print("ref_radius")
    get_and_print("ref_x")
    get_and_print("ref_y")
    get_and_print("sb_scla_drop_index")
    get_and_print("scla_deramp")
    get_and_print("scla_drop_index")
    get_and_print("scla_method")
    get_and_print("select_method")
    get_and_print("slc_osf")
    get_and_print("small_baseline_flag")
    get_and_print("subtr_tropo")
    get_and_print("tropo_method")
    get_and_print("unwrap_gold_alpha")
    get_and_print("unwrap_gold_n_win")
    get_and_print("unwrap_grid_size")
    get_and_print("unwrap_hold_good_values")
    get_and_print("unwrap_la_error_flag")
    get_and_print("unwrap_method")
    get_and_print("unwrap_patch_phase")
    get_and_print("unwrap_prefilter_flag")
    get_and_print("unwrap_spatial_cost_func_flag")
    get_and_print("unwrap_time_win")
    get_and_print("weed_max_noise")
    get_and_print("weed_neighbours")
    get_and_print("weed_standard_dev")
    get_and_print("weed_time_win")
    get_and_print("weed_zero_elevation")


def cli() -> None:
    """Command line interface."""

    global FANCY_PROGRESS
    global TRIANGLE
    global SNAPHU
    global OPTIONS
    global VERBOSE
    global DEBUG

    from argparse import (
        ArgumentTypeError,
        ArgumentParser,
        ArgumentDefaultsHelpFormatter,
    )

    def parse_stages(s: str) -> list[int]:
        """Parse the stage range argument."""
        min_stage, max_stage = 0, 7
        ss = s.split("-")
        if ss[0] == "":
            raise ArgumentTypeError(
                "Invalid stage range, it should be of the form 'n' or 'n-m'"
            )

        start = int(ss[0])
        end = int(ss[1]) if len(ss) > 1 else start

        if start < min_stage or end > max_stage:
            raise ArgumentTypeError(
                f"Invalid stage, it should be between {min_stage} and {max_stage}"
            )
        return list(range(start, end + 1))

    def parse_exec(p: str) -> Path:
        """Parse the executable argument and check it can run."""
        try:
            subprocess.run([p], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            raise ArgumentTypeError(f"Dependency '{p}' cannot be run: {e}")
        return Path(p)

    parser = ArgumentParser(
        description=" ".join(__doc__.split("\n")[:14]),
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    # Global options

    parser.add_argument("--nofancy", action="store_true", help="Disable fancy outputs")
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Disable verbose outputs"
    )
    parser.add_argument(
        "--logging", action="store_true", help="Use the `logging` module"
    )
    parser.add_argument(
        "--logconfig", type=Path, help="Use `logging` configuration file"
    )
    parser.add_argument(
        "-c", "--config", type=Path, help="Configuration file in .toml format"
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable debug outputs"
    )
    parser.add_argument("--test", action="store_true", help="Run the tests")
    parser.add_argument(
        "--check", action="store_true", help="Check against MATLAB outputs"
    )
    parser.add_argument(
        "--triangle", type=parse_exec, default=TRIANGLE, help="Triangle executable"
    )
    parser.add_argument(
        "--snaphu", type=parse_exec, default=SNAPHU, help="Snaphu executable"
    )
    parser.add_argument(
        "--processor", type=str, default="gamma", help="Processor to use"
    )
    parser.add_argument(
        "--maxmem", type=str, default="-1B", help="Maximum memory usage"
    )
    parser.add_argument("--params", action="store_true", help="Print all parameters")
    parser.add_argument("run", nargs="*", type=parse_stages, metavar="1 2 3-5")

    # Model options

    parser.add_argument(
        "--datadir", type=Path, default=Path(".."), help="Data directory"
    )
    parser.add_argument(
        "--master_date", type=str, default="", help="Master date", metavar="YYYYMMDD"
    )
    parser.add_argument("--da_thresh", type=float, default=0.4, help="DA threshold")
    parser.add_argument(
        "--rg_patches", type=int, default=1, help="Number of range patches"
    )
    parser.add_argument(
        "--az_patches", type=int, default=1, help="Number of azimuth patches"
    )
    parser.add_argument("--rg_overlap", type=int, default=50, help="Range overlap")
    parser.add_argument("--az_overlap", type=int, default=50, help="Azimuth overlap")
    parser.add_argument("--maskfile", type=Path, help="Mask file")

    try:
        # Parse the command line but also allow for other options to be passed
        # that are not defined in the parser but will be passed to the model

        args, other_opts = parser.parse_known_args()

        if args.params:
            print_all_parameters()
            sys.exit(0)

        if args.maxmem:
            limit_memory(args.maxmem)

        if args.triangle:
            TRIANGLE = args.triangle

        if args.snaphu:
            SNAPHU = args.snaphu

        if args.nofancy:
            FANCY_PROGRESS = False

        if not sys.stdout.isatty():
            FANCY_PROGRESS = False

        if args.quiet:
            VERBOSE = False

        if args.debug:
            DEBUG = True

        if args.test:
            run_tests()

        if args.check:
            check_results()
            sys.exit(0)

        if args.logging:
            setup_logging(args.logconfig)

        opts = load_and_normalise_config(args, args.config, other_opts)

        # Set the global options

        OPTIONS = opts

    except ArgumentTypeError as e:
        log(f"\nError: {e}")
        sys.exit(1)

    # To allow large-scale processing, we catch memory errors and try to restart the process
    # with more patches. We allow up to 3 restart attempts. This can be used in conjunction
    # with the --maxmem option to limit the memory usage so that the supercomputer does not
    # kill the process.

    restart_attempts = 0
    restart = True

    while restart:
        try:
            if restart_attempts == 0:
                if len(args.run) == 0:
                    log("Running all stages: 0-7")
                    run_all_stages(opts=opts)
                else:
                    stages = set(sorted([s for sec in args.run for s in sec]))
                    log(f"Running stages: {', '.join(map(str,stages))}")
                    for stage in stages:
                        run_stage(stage, opts=opts)

            else:
                log("Restarting after memory error and running all stages")
                run_all_stages(opts=opts)

            restart = False

        except (MemoryError, np.core._exceptions._ArrayMemoryError):  # type: ignore
            # Try to recover from a memory error

            log("Memory Error!")

            # We double the number of patches in both directions, effectively
            # quartering the patch size

            old_rg_patches = opts["rg_patches"]
            old_az_patches = opts["az_patches"]
            opts["rg_patches"] *= 2
            opts["az_patches"] *= 2

            log(
                "Attempting to run with more patches. "
                f"rg_patches: {old_rg_patches} -> {opts['rg_patches']}, "
                f"az_patches: {old_az_patches} -> {opts['az_patches']}"
            )

            restart_attempts += 1

            if restart_attempts < 3:
                restart = True
            else:
                log("Too many restart attempts, exiting now.")
                sys.exit(2)

        except NotImplementedError:
            import traceback

            line = traceback.extract_tb(sys.exc_info()[2])[-1][1]
            log(f"Attempted to run untested functionality at line {line}, exiting now.")
            sys.exit(3)

        except (RuntimeError, FileNotFoundError) as e:
            log(f"\nError: {e}")
            sys.exit(1)

        except KeyboardInterrupt:
            log("\nInterrupted! User pressed Ctrl-C")
            sys.exit(9)


if __name__ == "__main__":
    cli()
