#!/usr/bin/env python3
"""
This file contains the Python implementation of the STAMPS algorithm.
This Python implementation was written by Dale Roberts, and is based
on the original MATLAB code by Andrew Hooper.

The methodology consists of the following stages:

    - Stage 1: Load data from preprocessed GAMMA outputs
    - Stage 2: Estimate the initial coherence
    - Stage 3: Select stable pixels from candidate pixels
    - Stage 4: Weeding out unstable pixels chosen in stage 3
    - Stage 5: Correct the phase for look angle errors
    - Stage 6: Unwrapping the phase
    - Stage 7: Spatial filtering of the unwrapped phase
    ...

"""

import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import subprocess
import shutil
import sys
import os

from scipy.optimize import least_squares
from scipy.fft import fftshift, ifftshift  # do we need these? replace by np.fft?
from scipy.signal import fftconvolve, convolve2d, lfilter, firls
from scipy.signal.windows import gaussian
from scipy.linalg import lstsq  # Can be replaced with np.linalg.lstsq?
from scipy.spatial import KDTree

from datetime import datetime, timezone, timedelta
from joblib import dump, load
from pathlib import Path

from typing import Any, Tuple, Optional, List, no_type_check
from numpy.typing import NDArray as Array


np.set_printoptions(precision=4, suppress=True, linewidth=110, threshold=10000)

DEBUG = True
TRIANGLE = "/Users/daleroberts/Work/.envbin/bin/triangle"


class dotdict(dict):
    """A dictionary that allows access to its keys as attributes."""

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


def log(msg: str) -> None:
    """Prints a message to stderr."""
    if msg.startswith("#"):
        print("\n" + msg, file=sys.stderr)
    else:
        print(msg, file=sys.stderr)


def run_triangle_on(fn: Path) -> None:
    """Run the Triangle program on the given file."""
    base = fn.stem
    if DEBUG:
        out = sys.stdout
        subprocess.call([TRIANGLE, "-V", "-e", str(fn)], stdout=out)
    else:
        with open(f"triangle_{base}.log", "w") as out:
            subprocess.call([TRIANGLE, "-e", str(fn)], stdout=out)


def results_equal(
    name: str,
    is_idx: bool = False,
    atol: float = 1e-8,
    rtol: float = 1e-8,
    equal_nan: bool = True,
) -> bool:
    """Check if the results of the current run match the expected results that
    were obtained with the MATLAB version of the code."""

    python = stamps_load(name)
    matlab = loadmat(Path(f"{name}.mat"))

    if name.endswith("1") and "_" not in name:
        name = name[:-1]

    ndiffs = 0

    def compare_array(python: Array, matlab: Array) -> None:
        print(f"{python.dtype=} vs {matlab.dtype=}")
        print(f"{python.shape=} vs {matlab.shape=}")
        print(f"Python:\n{python}")
        print(f"MATLAB:\n{matlab}")
        diff = np.abs(python - matlab)
        mask = np.logical_and(np.isnan(python), np.isnan(matlab))
        diff[mask] = 0
        ndiff = np.nansum(diff > 1e-5)
        print(f"Count of differences: {ndiff}")
        ix = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"Location of max difference: {ix}")
        print(f"Python value: {python[ix]}")
        print(f"MATLAB value: {matlab[ix]}")
        print(f"Difference: {diff[ix]}")
        print("")
        if python.shape == matlab.shape:
            # print out values from equivalent columns next to each other
            nprinted = 0
            for i in range(python.shape[0]):
                if np.allclose(python[i], matlab[i], rtol=rtol, atol=atol, equal_nan=equal_nan):
                    continue
                nprinted += 1
                print(f"{i:6d}:\t{python[i]}\t{matlab[i]}")
                if nprinted > 10:
                    break
        print("")
        if DEBUG:
            import pdb
            pdb.set_trace()

    if isinstance(python, np.ndarray):
        if isinstance(matlab, dotdict):
            try:
                if len(matlab.keys()) == 1:
                    matlab = matlab[list(matlab.keys())[0]]
                else:
                    matlab = matlab[name]
            except KeyError:
                print(f"Unknown key `{name}`, debugging...")
                if DEBUG:
                    import pdb
                    pdb.set_trace()

        assert isinstance(matlab, np.ndarray)
        if is_idx:
            matlab = matlab - 1
        if python.shape != matlab.shape:
            print(f"\nError: `{name}` does not match (shape mismatch) "
                  f"{python.shape=} {matlab.shape=}")
            compare_array(python, matlab)
            return False
        if not np.allclose(python, matlab, rtol=rtol, atol=atol, equal_nan=equal_nan):
            print(f"\nError: `{name}` does not match")
            compare_array(python, matlab)
            return False
        else:
            log(f"`{name}` matches. {python.dtype=} {matlab.dtype=}. {atol=}, {rtol=}")
            return True

    for key in python:
        # Exclude various keys that are not relevant for comparison
        if (
            "ix" in key
            or "ij" in key
            or "loop" in key
            or "sort" in key
            or "sort_ix" in key
            or "sortix" in key
            or "bins" in key
        ):
            log(f"`{name}.{key}` check skipped")
            continue

        if key in matlab:
            log(f"Checking `{name}.{key}`")
            if isinstance(python[key], np.ndarray):
                if not np.allclose(
                    python[key],
                    matlab[key],
                    rtol=rtol,
                    atol=atol,
                    equal_nan=equal_nan,
                ):
                    print(f"\nError: `{key}` does not match")
                    compare_array(python[key], matlab[key])
                    ndiffs += 1
                else:
                    log(f"`{key}` matches with {atol = } and {rtol = }")
            elif isinstance(python[key], str):
                if python[key] != matlab[key]:
                    print(f"\nError: `{key}` does not match")
                    print(f"Python: {python[key]}")
                    print(f"MATLAB: {matlab[key]}")
                    print("")
                    ndiffs += 1
                else:
                    log(f"`{key}` matches")
            else:
                if abs(python[key] - matlab[key]) > atol:
                    print(f"\nError: `{key}` does not match")
                    print(f"Python: {python[key]}")
                    print(f"MATLAB: {matlab[key]}")
                    print("")
                    ndiffs += 1
                else:
                    log(f"`{key}` matches with {atol = } and {rtol = }")

    if ndiffs > 0:
        print(f"Total number of differences: {ndiffs}")
        if DEBUG:
            import pdb

            pdb.set_trace()

    return ndiffs == 0


def check(
    name: str,
    x: dict | Array,
    is_idx: bool = False,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    exit: bool = False,
) -> None:
    """A debug function to check if the results match the results that have
    been saved in MATLAB."""

    if isinstance(x, dict):
        stamps_save(name, **x)
    else:
        stamps_save(name, x)
    with np.printoptions(precision=4, suppress=True, linewidth=120, threshold=10):
        assert results_equal(name, is_idx=is_idx, atol=atol, rtol=rtol)
    if exit:
        sys.exit(0)


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


def getpsver() -> int:
    """Retrieve the PS version from the 'psver' file."""
    with open("psver", "r") as f:
        return int(f.read().strip())


def setpsver(version: int) -> None:
    """Set the PS version in the 'psver' file."""
    with open("psver", "w") as f:
        f.write(str(version))


def stamps_save(fn: str, *args: Optional[Array | dict], **kwargs: Optional[Any]) -> None:
    """Save a data file with the given name."""

    assert not fn.endswith(".mat")

    if fn.endswith(".pkl"):
        f = Path(fn)
    else:
        f = Path(f"{fn}.pkl")

    if len(args) > 0 and isinstance(args[0], np.ndarray):
        dump(args[0], f)
    else:
        dump(dotdict(kwargs), f)


def stamps_load(fn: str) -> dotdict | Array:
    """Load a data file with the given name."""

    assert not fn.endswith(".mat")

    if fn.endswith(".pkl"):
        f = Path(fn)
    else:
        f = Path(f"{fn}.pkl")

    data = load(f)

    if isinstance(data, dotdict):
        return data

    if isinstance(data, dict):
        return dotdict(data)

    if isinstance(data, np.ndarray):
        return data

    raise ValueError(f"File {fn} contains unknown data type: {type(data)}")


def stamps_exists(fn: str) -> bool:
    """Check if a data file with the given name exists."""

    assert not fn.endswith(".mat")

    if fn.endswith(".pkl"):
        f = Path(fn)
    else:
        f = Path(f"{fn}.pkl")

    return f.exists()


def loadmat(fname: Path) -> dotdict:
    """Loads a .mat file."""

    mat = sio.loadmat(str(fname), squeeze_me=True)
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


def interp(data: Array[np.floating], r: int, n: int = 4, cutoff: float = 0.5) -> Array[np.floating]:
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


def getparm(parmname: Optional[str] = None, verbose: bool = True) -> str:
    """Retrieves a parameter value from parameter files."""

    # TODO: Add support for other file formats (txt?) instead of just .mat

    def pprint(k: str, v: Any) -> None:
        if isinstance(v, str):
            log(f"{k} = '{v}'")
        else:
            log(f"{k} = {v}")

    # Load global parameters

    parmfile = Path("parms.mat").absolute()
    if parmfile.exists():
        parms = loadmat(parmfile)
    elif (parmfile.parent.parent / parmfile.name).exists():
        parmfile = parmfile.parent.parent / parmfile.name
        parms = loadmat(parmfile)
    else:
        raise FileNotFoundError(f"`{parmfile}` not found")

    # Load local parameters, if available

    localparmfile = Path("localparms.mat")
    if localparmfile.exists():
        localparms = loadmat(localparmfile)
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

    parmfile = Path("parms.mat").absolute()
    if parmfile.exists():
        parms = loadmat(parmfile)
    elif (parmfile.parent.parent / parmfile.name).exists():
        parmfile = parmfile.parent.parent / parmfile.name
        parms = loadmat(parmfile)
    else:
        raise FileNotFoundError(f"`{parmfile}` not found")

    parms[parmname] = value
    sio.savemat(str(parmfile), parms)


def readparm(fname: Path, parm: str) -> str:
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


def readparms(fname: Path, parm: str, numval: int) -> List[str]:
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
        # read directory names from `patch.list`
        with patchlist.open("r") as f:
            dirs = [Path(line.strip()) for line in f.readlines()]
    else:
        # if patch.list does not exist, find all directories with PATCH_ in the name
        dirs = [d for d in Path(".").iterdir() if d.is_dir() and "PATCH_" in d.name]

    if len(dirs) == 0:
        # patch directories not found, use current directory
        dirs.append(Path("."))

    return dirs


def clap_filt(
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
    wind_func = np.pad(np.add.outer(x, x), ((0, n_win // 2), (0, n_win // 2)), mode="symmetric")

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

            ph_out[i1:i2, j1:j2] = ph_out[i1:i2, j1:j2] + ph_filt[: i2 - i1, : j2 - j1] * wf2

    return ph_out


def clap_filt_patch(
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


def goldstein_filt(
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
        wf = wind_func[: i2 - i1, : i2 - i1]  # Adjust the window function for window size
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
            H = ifftshift(convolve2d(fftshift(np.abs(ph_fft)), gauss_kernel, mode="same"))
            meanH = np.median(H)
            H = ((H / meanH) ** alpha if meanH != 0 else H**alpha) * (n_win + n_pad) ** 2

            # Inverse FFT and update the output array
            # NOTE: changed scipy.fft.ifft2 to numpy.fft.ifft2
            ph_filt = np.fft.ifft2(ph_fft * H).real[: i2 - i1, : j2 - j1] * (wf_i[:, None] * wf_j)
            ph_out[i1:i2, j1:j2] += ph_filt

    return ph_out


# def gradient_filt(
#   ph: np.ndarray, n_win: int
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#   """
#   Determine 2-D gradient through FFT over windows of size n_win.
#   """
#
#   # Initialize variables
#   n_i, n_j = ph.shape
#   n_inc = n_win // 4
#   n_win_i = -(-n_i // n_inc) - 3  # Ceiling division
#   n_win_j = -(-n_j // n_inc) - 3
#
#   # Replace NaNs with zeros
#   ph = np.nan_to_num(ph)
#
#   # Initialize output arrays
#   Hmag = np.full((n_win_i, n_win_j), np.nan)
#   ifreq = Hmag.copy()
#   jfreq = Hmag.copy()
#   ij = np.full((n_win_i * n_win_j, 2), np.nan)
#
#   def calc_bounds(
#       ix: int, n_inc: int, n_win: int, max_dim: int
#   ) -> Tuple[int, int, int, int]:
#       i1 = ix * n_inc
#       i2 = min(i1 + n_win, max_dim)
#       return max(i2 - n_win, 0), i2
#
#   def calc_freq(I1: Array, I2: Array, n_win: int) -> Tuple[Array, Array]:
#       I1 = (I1 + n_win // 2) % n_win
#       I2 = (I2 + n_win // 2) % n_win
#       return (
#           (I1 - n_win // 2) * 2 * np.pi / n_win,
#           (I2 - n_win // 2) * -2 * np.pi / n_win,
#       )
#
#   idx = 0
#   for ix1 in range(n_win_i):
#       for ix2 in range(n_win_j):
#           # Calculate window bounds
#           i1, i2, j1, j2 = calc_bounds(ix1, ix2, n_inc, n_win, n_i, n_j)
#
#           # Extract phase window and apply FFT
#           ph_bit = ph[i1:i2, j1:j2]
#           if np.count_nonzero(ph_bit) < 6:  # Check for enough non-zero points
#               continue
#
#           ph_fft = np.fft.fft2(ph_bit)
#           H = np.abs(ph_fft)
#
#           # Find the index of the maximum magnitude
#           I = np.argmax(H)
#           Hmag_this = H.flat[I] / H.mean()
#
#           # Calculate frequencies
#           I1, I2 = np.unravel_index(I, (n_win, n_win))
#           ifreq_val, jfreq_val = calc_freq(I1, I2, n_win)
#
#           # Update output arrays
#           Hmag[ix1, ix2] = Hmag_this
#           ifreq[ix1, ix2] = ifreq_val
#           jfreq[ix1, ix2] = jfreq_val
#           ij[idx] = [(i1 + i2) / 2, (j1 + j2) / 2]
#           idx += 1
#
#   return ifreq.T, jfreq.T, ij, Hmag.T


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
    n_trials = len(trial_mult)

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

    # Linearize and solve for residual phase
    resphase = cpxphase * np.exp(-1j * (K0 * bperp))
    offset_phase = np.sum(resphase)
    resphase = np.angle(resphase * np.conj(offset_phase))

    # Weighted least squares fit for residual phase
    weighting = np.abs(cpxphase)
    mopt = np.linalg.lstsq(weighting[:, None] * bperp[:, None], weighting * resphase, rcond=None)[0]
    K0 += mopt[0]

    # Calculate phase residuals
    phase_residual = cpxphase * np.exp(-1j * (K0 * bperp))
    mean_phase_residual = np.sum(phase_residual)
    C0 = np.angle(mean_phase_residual)  # Updated static offset
    coh0 = np.abs(mean_phase_residual) / np.sum(np.abs(phase_residual))  # Updated coherence

    return K0, C0, coh0, phase_residual


def stage1_load_data(endian: str = "b") -> None:
    """Load all the data we need from GAMMA outputs, process it, and
    save it into our own data format."""

    log("# Stage 1: Load initial data from GAMMA outputs")

    # File names assume we are in a PATCH_ directory
    assert Path(".").name.startswith("PATCH_")

    phname = Path("./pscands.1.ph")  # phase data
    ijname = Path("./pscands.1.ij")  # pixel location data
    llname = Path("./pscands.1.ll")  # latitude, longitude data
    xyname = Path("./pscands.1.xy")  # local coordinates
    hgtname = Path("./pscands.1.hgt")  # height data
    daname = Path("./pscands.1.da")  # dispersion data
    rscname = Path("../rsc.txt")  # config with master rslc.par file location
    pscname = Path("../pscphase.in")  # config with width and diff phase file locataions

    # Read master day from rsc file
    with rscname.open() as f:
        rslcpar = Path(f.readline().strip())

    log(f"{rslcpar = }")

    # Read interferogram dates
    with pscname.open() as f:
        ifgs = [Path(line.strip()) for line in f.readlines()][1:]

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
    heading = float(readparm(rslcpar, "heading"))
    setparm("heading", heading)

    freq = float(readparm(rslcpar, "radar_frequency"))
    lam = 299792458 / freq
    setparm("lambda", lam)

    sensor = readparm(rslcpar, "sensor")
    platform = sensor  # S1 case
    setparm("platform", platform)

    rps = float(readparm(rslcpar, "range_pixel_spacing"))
    rgn = float(readparm(rslcpar, "near_range_slc"))
    se = float(readparm(rslcpar, "sar_to_earth_center"))
    re = float(readparm(rslcpar, "earth_radius_below_sensor"))
    rgc = float(readparm(rslcpar, "center_range_slc"))
    naz = int(readparm(rslcpar, "azimuth_lines"))
    prf = float(readparm(rslcpar, "prf"))

    mean_az = naz / 2.0 - 0.5

    # Processing of the id, azimuth, range data
    with ijname.open("rb") as f:
        ij = np.loadtxt(f, dtype=np.uint16)

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
        B_TCN = np.array([float(x) for x in readparms(basename, "initial_baseline(TCN)", 3)])
        BR_TCN = np.array([float(x) for x in readparms(basename, "initial_baseline_rate", 3)])
        bc = B_TCN[1] + BR_TCN[1] * (ij[:, 1] - mean_az) / prf
        bn = B_TCN[2] + BR_TCN[2] * (ij[:, 1] - mean_az) / prf
        # Convert baselines from (T)CN to perpendicular-parallel coordinates
        bperp_mat[:, i] = bc * np.cos(look) - bn * np.sin(look)

    # Calculate mean perpendicular baselines
    bperp = np.mean(bperp_mat, axis=0)

    log("Mean perpendicular baseline for each interferogram:")
    for i in range(n_ifg):
        log(f"{i+1}:\t{ifgs[i]}\tmean(bperp) = {bperp[i]:+.3f}")

    # Calculate incidence angles
    inci = np.arccos((se**2 - re**2 - rg**2) / (2 * re * rg))

    # Calculate mean incidence angle
    mean_incidence = np.mean(inci)

    # Mean range is given by the center range distance
    mean_range = rgc

    # Processing of the phase data
    with phname.open("rb") as f:
        log(f"Loading phase data from `{phname.absolute()}`")
        ph = np.fromfile(f, dtype=">c8").reshape((n_ifg, n_ps)).T

    # Calculate mean phases
    mu = np.mean(ph, axis=0)
    for i in range(n_ifg):
        log(f"{i+1}:\t{ifgs[i]}\tmean(phase) = {mu[i]:+.3f}")
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
        log(f"Rotation improved alignment, applying rotation {theta * 180 / np.pi:.2f}°")
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

    setpsver(1)
    psver = getpsver()

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

    stamps_save(f"ph{psver}", ph)
    stamps_save(f"bp{psver}", bperp_mat)
    stamps_save(f"la{psver}", la)
    stamps_save(f"da{psver}", D_A)
    stamps_save(f"hgt{psver}", hgt)

    log("Stage 1 complete.")


def stage2_estimate_noise(max_iters: int = 1000) -> None:
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
            rand_ifg[:, i] = rand_image[:, ifgday_ix[i, 1]] - rand_image[:, ifgday_ix[i, 0]]
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

    log(f"Fitting topographic phase models to {n_rand} random interferogram")

    # Iterate through random phase points in reverse order
    coh_rand = np.zeros(n_rand)
    for i in reversed(range(n_rand)):
        # Fit a topographic phase model to each random phase point
        K_r, C_r, coh_r, res_r = topofit(exp_rand_ifg[i, :], bperp, n_trial_wraps)
        # Store the first coherence value for each random point
        coh_rand[i] = coh_r

    # check("coh_rand", coh_rand)

    coh_bins = np.arange(0.0, 1.01, 0.01)  # old matlab hist uses bin centers

    log(f"Generating histogram of {n_rand} coherences using {len(coh_bins)} bins")

    Nr, _ = np.histogram(coh_rand, bins=coh_bins)
    Nr = Nr.astype(np.float64)  # Fix type - StaMPS error

    # check("Nr", Nr)

    # Find the last non-zero bin index using np.max and np.nonzero
    Nr_max_nz_ix = np.max(np.nonzero(Nr)[0])

    K_ps, C_ps, coh_ps, coh_ps_save, N_opt = (np.zeros(n_ps) for _ in range(5))
    ph_res = np.zeros((n_ps, n_ifg), dtype=np.float32)
    ph_patch = np.zeros_like(ph, dtype=ph.dtype)
    N_patch = np.zeros(n_ps)

    # Calculate grid indices for the third column of 'xy'
    grid_ij = np.zeros((xy.shape[0], 2), dtype=int)
    grid_ij[:, 0] = np.ceil((xy[:, 2] - np.min(xy[:, 2]) + 1e-6) / grid_size).astype(int)
    # Adjust indices to ensure they are within bounds for the first column
    grid_ij[grid_ij[:, 0] == np.max(grid_ij[:, 0]), 0] = np.max(grid_ij[:, 0]) - 1
    # Calculate grid indices for the second column of 'xy'
    grid_ij[:, 1] = np.ceil((xy[:, 1] - np.min(xy[:, 1]) + 1e-6) / grid_size).astype(int)
    # Adjust indices to ensure they are within bounds for the second column
    grid_ij[grid_ij[:, 1] == np.max(grid_ij[:, 1]), 1] = np.max(grid_ij[:, 1]) - 1

    grid_ij -= 1  # 0-based indexing

    n_i = np.max(grid_ij[:, 0]) + 1
    n_j = np.max(grid_ij[:, 1]) + 1

    # check("grid_ij", grid_ij+1)

    weighting = 1.0 / da
    weighting_save = weighting.copy()
    gamma_change_save = 0

    log(f"Processing {n_ps} PS candidates")

    for iter in range(1, max_iters + 1):
        log(f"* Iteration {iter}, processing patch phases")

        # check(f"K_ps_{iter}", K_ps, atol=1e-3, rtol=1e-3)
        # check(f"weighting_{iter}", weighting, atol=1e-3, rtol=1e-3)

        # Initialize phase grids for raw phases, filtered phases, and weighted phases
        ph_grid = np.zeros((n_i, n_j, n_ifg), dtype=np.complex64)
        ph_filt = np.copy(ph_grid)  # Copy ph_grid structure for filtered phases

        # Calculate weighted phases, adjusting for baseline and applying weights
        ph_weight = ph * np.exp(-1j * bperp_mat * K_ps[:, None]) * weighting[:, None]

        # check(f"ph_weight_{iter}", ph_weight)

        # Accumulate weighted phases into grid cells
        for i in range(n_ps):
            ph_grid[grid_ij[i, 0], grid_ij[i, 1], :] = (
                ph_grid[grid_ij[i, 0], grid_ij[i, 1], :] + ph_weight[i, :]
            )

        # check(f"ph_grid_{iter}", ph_grid)

        # Apply filtering to each interferogram in the grid
        for i in range(n_ifg):
            # Apply a CLAP filter (an edge-preserving smoothing filter) to the phase grid
            ph_filt[:, :, i] = clap_filt(
                ph_grid[:, :, i],
                clap_alpha,
                clap_beta,
                int(n_win * 0.75),
                int(n_win * 0.25),
                low_pass,
            )

        # check(f"ph_filt_{iter}", ph_filt, atol=1e-3, rtol=1e-3)

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
                Kopt, Copt, cohopt, ph_residual = topofit(psdph, bperp_mat[i, :], n_trial_wraps)

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

            # Log progress for every 1000 points processed
            if i % 1000 == 0 and i > 0:
                log(f"{i} PS processed")

        log(f"Done. {n_ps} PS processed.")

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
            Prand = lfilter(gauss_filter, 1, np.concatenate((np.ones(7), Prand))) / np.sum(
                gauss_filter
            )
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
        ph_patch=ph_patch,
        K_ps=K_ps,
        C_ps=C_ps,
        coh_ps=coh_ps,
        N_opt=N_opt,
        ph_res=ph_res,
        ph_grid=ph_grid,
        n_trial_wraps=n_trial_wraps,
        grid_ij=grid_ij,
        grid_size=grid_size,
        low_pass=low_pass,
        i_loop=iter,
        coh_bins=coh_bins,
        Nr=Nr,
        # step_number=step_number,
        # ph_weight=ph_weight,
        # Nr_max_nz_ix=Nr_max_nz_ix,
        # coh_ps_save=coh_ps_save,
        # gamma_change_save=gamma_change_save,
    )

    log("Stage 2 complete.")


def stage3_select_ps(reest_flag: int = 0, plots: bool = False) -> None:
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

    psver = getpsver()
    if psver > 1:
        setpsver(1)
        psver = 1

    # Retrieve parameters
    slc_osf = float(getparm("slc_osf"))
    clap_alpha = float(getparm("clap_alpha"))
    clap_beta = float(getparm("clap_beta"))
    n_win = int(getparm("clap_win"))
    select_method = getparm("select_method")

    if select_method.lower() == "percent":
        max_percent_rand = float(getparm("percent_rand"))
    else:
        max_density_rand = float(getparm("density_rand"))

    gamma_stdev_reject = float(getparm("gamma_stdev_reject"))
    small_baseline_flag = getparm("small_baseline_flag")  # string
    drop_ifg_index = np.array(getparm("drop_ifg_index"))  # FIXME: This could be a list!

    # Setting low coherence threshold based on small_baseline_flag

    if small_baseline_flag == "y":
        low_coh_thresh = 15
    else:
        low_coh_thresh = 31

    log(f"{slc_osf = } (SLC oversampling factor)")
    log(f"{clap_alpha = } (CLAP alpha)")
    log(f"{clap_beta = } (CLAP beta)")
    log(f"{n_win = } (CLAP window size)")
    log(f"{select_method = } (selection method)")
    log(f"{max_percent_rand = } (maximum percent random)")
    log(f"{max_density_rand = } (maximum density random)")
    log(f"{gamma_stdev_reject = } (gamma standard deviation reject)")
    log(f"{small_baseline_flag = } (small baseline flag)")
    log(f"{drop_ifg_index = } (interferogram indices to drop)")
    log(f"{low_coh_thresh = } (low coherence threshold)")

    # Load data
    ps = stamps_load(f"ps{psver}")
    if stamps_exists(f"ph{psver}"):
        ph = stamps_load(f"ph{psver}")
    else:
        ph = ps["ph"]

    bperp = ps["bperp"]
    n_ifg = int(ps["n_ifg"])

    ifg_index = np.setdiff1d(np.arange(1, ps["n_ifg"] + 1), drop_ifg_index)

    # Adjust ifg_index based on small_baseline_flag
    if not small_baseline_flag == "y":
        master_ix = np.sum(ps["master_day"] > ps["day"]) + 1
        no_master_ix = np.array([i for i in range(1, ps["n_ifg"] + 1) if i not in [master_ix]])
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
        bin_size = 10000 if D_A.size >= 50000 else 2000
        D_A_max = np.concatenate(
            (np.array([0]), D_A_sort[bin_size:-bin_size:bin_size], D_A_sort[-1])
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
            Nr = Nr_dist * Na[1 : low_coh_thresh + 1].sum() / Nr_dist[1 : low_coh_thresh + 1].sum()

            # check(f"Na_{i+1}", Na)

            Na[Na == 0] = 1  # avoid divide by zero

            if select_method.lower() == "percent":
                percent_rand = np.flip(np.cumsum(np.flip(Nr)) / np.cumsum(np.flip(Na)) * 100)
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

    # check("coh_thresh", coh_thresh)
    # check("coh_thresh_coeffs", coh_thresh_coeffs)

    log(
        f"Initial gamma threshold: {min(coh_thresh):.3f} at D_A={min(D_A):.2f}"
        f"to {max(coh_thresh):.3f} at D_A={max(D_A):.2f}"
    )

    if plots:
        plt.figure(3)
        plt.plot(D_A_mean, min_coh, "*")
        if len(coh_thresh_coeffs) > 0:
            plt.plot(D_A_mean, np.polyval(coh_thresh_coeffs, D_A_mean), "r")
        plt.ylabel(r"$\gamma_{thresh}$")
        plt.xlabel("D_A")
        plt.show()

    if plots:
        plt.figure(4)
        plt.hist(
            pm["coh_ps"],
            bins=50,
            color="magenta",
            alpha=0.4,
            edgecolor="magenta",
        )
        for ct in coh_thresh:
            plt.axvline(x=ct, color="k", linestyle="--", label="coh_thresh")
        plt.title("coh_ps")
        plt.show()

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
                    log(f'{datestr(ps["day"][i])} is dropped from noise re-estimation')

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
                    ph_bit = pm["ph_grid"][i_min : (i_max + 1), j_min : (j_max + 1), :].copy()
                    ph_bit[ps_bit_i, ps_bit_j, :] = 0

                    # Oversample update for PS removal + general usage update
                    ix_i = np.arange(
                        max(ps_bit_i - (slc_osf - 1), 0),
                        min(ps_bit_i + slc_osf, ph_bit.shape[0]),
                    )
                    ix_j = np.arange(
                        max(ps_bit_j - (slc_osf - 1), 0),
                        min(ps_bit_j + slc_osf, ph_bit.shape[1]),
                    )
                    # Set the oversampled region to 0
                    ph_bit[np.ix_(ix_i, ix_j)] = 0

                    for ifg in range(n_ifg):
                        ph_filt[:, :, ifg] = clap_filt_patch(
                            ph_bit[:, :, ifg],
                            clap_alpha,
                            clap_beta,
                            pm["low_pass"],
                        )

                    ph_patch2[i] = ph_filt[ps_bit_i, ps_bit_j, :]

                if i % 10000 == 0:
                    log(f"{i} patches re-estimated")

            # check("ij_idxs", ij_idxs + 1)
            # check("ph_patch2", ph_patch2, atol=1e-3, rtol=1e-3)

            del pm["ph_grid"]
            bp = stamps_load(f"bp{psver}")
            bperp_mat = bp[ix, :]

            for i in range(n_ps):
                psdph = ph[i] * np.conj(ph_patch2[i])
                if not np.any(psdph == 0):  # Ensure there's a non-null value in every interferogram
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

                if i % 10000 == 0:
                    log(f"{i} coherences re-estimated")

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

        # 433
        for i in range(len(D_A_max) - 1):
            coh_chunk = pm["coh_ps"][(D_A > D_A_max[i]) & (D_A <= D_A_max[i + 1])]
            D_A_mean[i] = D_A[(D_A > D_A_max[i]) & (D_A <= D_A_max[i + 1])].mean()
            coh_chunk = coh_chunk[
                coh_chunk != 0
            ]  # Discard PSC for which coherence was not calculated

            Na, _ = np.histogram(coh_chunk, bins=pm["coh_bins"])
            Nr = Nr_dist * Na[: low_coh_thresh + 1].sum() / Nr_dist[: low_coh_thresh + 1].sum()

            Na[Na == 0] = 1  # Avoid divide by zero

            if select_method.lower() == "percent":
                percent_rand = np.flip(np.cumsum(np.flip(Nr)) / np.cumsum(np.flip(Na)) * 100)
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

        # 450
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

        # 473
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

    # 485
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
        print("***No PS points left. Updating the stamps log for this***")
        stamps_step_no_ps[2] = 1

    stamps_save("no_ps_info", stamps_step_no_ps=stamps_step_no_ps)

    if plots:
        plt.figure(6)
        plt.plot(D_A_mean, min_coh, "*")
        if len(coh_thresh_coeffs) > 0:
            plt.plot(D_A_mean, np.polyval(coh_thresh_coeffs, D_A_mean), "r")
        plt.ylabel(r"$\gamma_{thresh}$")
        plt.xlabel("D_A")
        plt.show()

    stamps_save(
        f"select{psver}",
        ix=ix,
        keep_ix=keep_ix,
        ph_patch2=ph_patch2,
        ph_res2=ph_res2,
        K_ps2=K_ps2,
        C_ps2=C_ps2,
        coh_ps2=coh_ps2,
        coh_thresh=coh_thresh,
        coh_thresh_coeffs=coh_thresh_coeffs,
        clap_alpha=clap_alpha,
        clap_beta=clap_beta,
        n_win=n_win,
        max_percent_rand=round(max_percent_rand, 0),
        gamma_stdev_reject=gamma_stdev_reject,
        small_baseline_flag=small_baseline_flag,
        ifg_index=ifg_index + 1,  # 1-based indexing to match MATLAB
    )

    log("Stage 3 complete.")


def stage4_weed_ps(
    all_da_flag: bool = False,
    no_weed_adjacent: bool = False,
    no_weed_noisy: bool = False,
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

    psver = getpsver()
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
        neigh_ix = np.zeros((np.max(ij_shift[:, 0]) + 1, np.max(ij_shift[:, 1]) + 1), dtype=int)
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

        check("neigh_ix", neigh_ix + 1, exit=True)

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
    dups = np.setdiff1d(np.arange(np.sum(ix_weed)), unique_indices)  # pixels with duplicate lon/lat

    for i in range(len(dups)):
        dups_ix_weed = np.where(
            (xy_weed[:, 1] == xy_weed[dups[i], 1]) & (xy_weed[:, 2] == xy_weed[dups[i], 2])
        )[0]
        dups_ix = ix_weed_num[dups_ix_weed]
        max_coh_ix = np.argmax(coh_ps2[dups_ix])
        ix_weed[dups_ix[np.arange(len(dups_ix)) != max_coh_ix]] = False  # drop dups with lowest coh

    if len(dups) > 0:
        xy_weed = xy2[ix_weed, :]
        log(f"{len(dups)} PS with duplicate lon/lat dropped")
    else:
        log("No PS with duplicate lon/lat")

    n_ps = np.sum(ix_weed)
    ix_weed2 = np.ones(n_ps, dtype=bool)

    check("xy_weed", xy_weed)

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
                    [TRIANGLE, "-e", "psweed.1.node"], stdout=open("triangle_weed.log", "w")
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

        check("edgs", edgs + 1)  # 1-based indexing

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
                ds = datetime.strptime(str(ps["ifgday"][i, 1]), "%Y%m%d").strftime("%Y-%m-%d")
                print(f"{ds}-{ds} dropped from noise estimation")
            else:
                ds = datetime.strptime(str(day[i]), "%Y%m%d").strftime("%Y-%m-%d")
                print(f"{ds} dropped from noise estimation")

        if not small_baseline_flag.lower() == "y":
            print(f"Estimating noise for {n_use} arcs...")

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

        check("edge_std", edge_std, atol=1e-3, rtol=1e-3)
        check("edge_max", edge_max, atol=1e-3, rtol=1e-3)

        # We now remove points with excessive noise. We calculate the standard
        # deviation and maximum noise level for each pixel based on noise
        # estimates for edges connecting the pixels. Then, we apply thresholds
        # to identify and keep only the pixels with noise levels below these
        # thresholds (`weed_standard_dev` and `weed_max_noise`), effectively
        # weeding out noisy pixels from the dataset.

        print("Estimating max noise for all pixels...")
        ps_std = np.full(n_ps, np.inf, dtype=np.float32)
        ps_max = np.full(n_ps, np.inf, dtype=np.float32)
        for i in range(n_edge):
            ps_std[edgs[i, :]] = np.minimum(ps_std[edgs[i, :]], [edge_std[i], edge_std[i]])
            ps_max[edgs[i, :]] = np.minimum(ps_max[edgs[i, :]], [edge_max[i], edge_max[i]])

        ix_weed2 = (ps_std < weed_standard_dev) & (ps_max < weed_max_noise)
        ix_weed[ix_weed] = ix_weed2
        n_ps = np.sum(ix_weed)

        check("ps_std", ps_std, atol=1e-3, rtol=1e-3)
        check("ps_max", ps_max, atol=1e-3, rtol=1e-3)

        print(f"{n_ps} PS kept after dropping noisy pixels")

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
        ph_patch=ph_patch,
        ph_res=ph_res,
        coh_ps=coh_ps,
        K_ps=K_ps,
        C_ps=C_ps,
    )

    # Prepare phase data for saving
    ph2 = ph2[ix_weed, :]
    stamps_save(f"ph{psver+1}", ph2)

    # Update PS information with weed results
    xy2 = xy2[ix_weed, :]
    ij2 = ij2[ix_weed, :]
    lonlat2 = lonlat2[ix_weed, :]

    ps.update({"xy": xy2, "ij": ij2, "lonlat": lonlat2, "n_ps": ph2.shape[0]})
    psname = f"ps{psver + 1}"
    stamps_save(psname, ps)

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
        stamps_save(f"la{psver + 1}", la)

    # Process and save incidence angle data if available
    if stamps_exists(f"inc{psver}"):
        inc = stamps_load(f"inc{psver}")
        inc = inc[ix2]
        if all_da_flag:
            inc_other = inc[ix_other]
            inc = np.concatenate([inc, inc_other])
        inc = inc[ix_weed]
        stamps_save(f"inc{psver + 1}", inc)

    # Process and save baseline data if available
    if stamps_exists(f"bp{psver}"):
        bperp_mat = stamps_load(f"bp{psver}")
        bperp_mat = bperp_mat[ix2, :]
        if all_da_flag:
            bperp_other = bperp[ix_other, :]
            bperp_mat = np.concatenate([bperp_mat, bperp_other])
        bperp_mat = bperp_mat[ix_weed, :]
        stamps_save(f"bp{psver + 1}", bperp_mat)

    log("Stage 4 complete.")


def stage5_correct_phases() -> None:
    """
    Correct the wrapped phases of the selected PS for spatially-uncorrelated
    look angle (DEM) error. This is done by subtracting the range error and
    master noise from the phase data. The corrected phase data is saved for
    further processing.
    """

    log("# Stage 5: Correcting phase for look angle (DEM) error")

    small_baseline_flag = getparm("small_baseline_flag")

    psver = getpsver()
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
                -K_ps[:, np.newaxis] * bperp_mat  # range error
                - C_ps[:, np.newaxis] * np.ones(n_ifg)  # master noise
            )
        )

        ph_reref = np.hstack(
            (
                ph_patch[:, :master_ix],
                np.ones((n_ps, 1)),
                ph_patch[:, master_ix:],
            )
        )

        # Save the corrected phase and ph_reref
        stamps_save(f"rc{psver}", ph_rc=ph_rc, ph_reref=ph_reref)

    log("Stage 5 complete.")


def ps_calc_ifg_std() -> None:
    """Calculate the standard deviation of the interferograms."""

    small_baseline_flag = getparm("small_baseline_flag")
    psver = getpsver()

    ps = stamps_load(f"ps{psver}")
    pm = stamps_load(f"pm{psver}")
    bp = stamps_load(f"bp{psver}")
    ph = stamps_load(f"ph{psver}")

    assert isinstance(ps, dict)
    assert isinstance(pm, dict)
    assert isinstance(bp, np.ndarray)
    assert isinstance(ph, np.ndarray)

    n_ifg = ps["n_ifg"]
    n_ps = ps["n_ps"]
    master_ix = np.sum(ps["master_day"] > ps["day"])

    log("Estimating noise standard deviation (in degrees)")

    if small_baseline_flag == "y":
        ph_diff = np.angle(ph * np.conj(pm.ph_patch) * np.exp(-1j * (pm.K_ps[:, np.newaxis] * bp)))
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
            log(f"{i+1:3d}\t{datestr(ifgday[i, 0])}_{datestr(ifgday[i, 1])}\t{ifg_std[i]:>3.2f}")
    else:
        day = ps.day
        log(f"INDEX    IFG_DATE       MEAN    STD_DEV")
        for i in range(ps["n_ifg"]):
            log(f"{i+1:5d}  {datestr(day[i]):10s}    {ifg_mean[i]:>6.2f}°    {ifg_std[i]:>6.2f}°")

    stamps_save(f"ifgstd{psver}", ifg_std=ifg_std)


def stage6_unwrap_phases() -> None:
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

    psver = getpsver()
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

            if scla_deramp == "y" and "ph_ramp" in scla and scla.ph_ramp.shape[0] == ps.n_ps:
                ramp_subtracted_sw = 1  # FIXME: Change to bool
                # Subtract orbital ramps
                ph_w *= np.exp(-1j * scla.ph_ramp)
            else:
                log("   wrong number of PS in scla - subtraction skipped...")
                os.remove(sclaname + ".mat")  # FIXME: Check if this is correct

    if small_baseline_flag == "y" and os.path.exists(sclaname + ".mat"):  # Small baselines
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

            if scla_deramp == "y" and "ph_ramp" in scla and scla.ph_ramp.shape[0] == ps.n_ps:
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
        ifgday_ix = np.column_stack(
            (np.ones((ps.n_ifg, 1)) * ps.master_ix, np.arange(1, ps.n_ifg + 1))
        )
        master_ix = np.sum(ps.master_day > ps.day)
        unwrap_ifg_index = np.setdiff1d(
            unwrap_ifg_index, master_ix
        )  # leave master ifg (which is only noise) out
        day = ps.day - ps.master_day

    if unwrap_hold_good_values == "y":
        options["ph_uw_predef"] = options["ph_uw_predef"][:, unwrap_ifg_index]

    if sys.platform.startswith("win"):
        log("Windows detected: using old unwrapping code without statistical cost processing")
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
    msd = np.zeros((ps.n_ifg, 1), dtype=np.float32)

    ph_uw[:, unwrap_ifg_index] = ph_uw_some

    if "msd_some" in locals():  # FIXME
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

    stamps_save(phuwname, ph_uw, msd)


def sb_identify_good_pixels() -> None:
    raise NotImplementedError


def ps_plot_tca(aps, aps_name) -> Tuple[np.ndarray, str, str]: # type: ignore
    raise NotImplementedError


def uw_nosnaphu(ph_w:Array, xy:Array, day:Array, options:Optional[dict]=None) -> Array:
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

    single_master_flag = len(np.unique(ifgday_ix[:, 0])) == 1

    # Set default option values if not provided
    options.setdefault("master_day", 0)
    options.setdefault("grid_size", 5)
    options.setdefault("prefilt_win", 16)
    options.setdefault("time_win", 365)
    options.setdefault("unwrap_method", "3D_FULL" if not single_master_flag else "3D")
    options.setdefault("goldfilt_flag", "n")
    options.setdefault("lowfilt_flag", "n")
    options.setdefault("gold_alpha", 0.8)
    options.setdefault("n_trial_wraps", 6)
    options.setdefault("la_flag", "y")
    options.setdefault("scf_flag", "y")
    options.setdefault("temp", [])
    options.setdefault("n_temp_wraps", 2)
    options.setdefault("max_bperp_for_temp_est", 100)
    options.setdefault("variance", [])
    options.setdefault("ph_uw_predef", None)

    # Ensure correct shape for input arrays
    if xy.shape[1] == 2:
        xy = np.hstack((np.arange(1, xy.shape[0] + 1).reshape(-1, 1), xy))

    day = np.array(day).flatten()

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


    # TODO: Implement the rest of the function
    raise NotImplementedError

    return ph_uw, msd


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
            ph_this = ph_in[:, i1]

        if ph_in_predef is not None:
            ph_this_uw = ph_in_predef[:, i1]
            ph_grid_uw[:] = 0
            N_grid_uw[:] = 0

        ph_grid[:] = 0

        if pix_size == 0:
            ph_grid[(xy_in[:, 1] - 1) * n_i + xy_in[:, 2]] = ph_this
            if ph_in_predef is not None:
                ph_grid_uw[(xy_in[:, 1] - 1) * n_i + xy_in[:, 2]] = ph_this_uw
        else:
            for i in range(n_ps):
                ph_grid[grid_ij[i, 0], grid_ij[i, 1]] += ph_this[i]
            if ph_in_predef is not None:
                for i in range(n_ps):
                    if not np.isnan(ph_this_uw[i]):
                        ph_grid_uw[grid_ij[i, 0], grid_ij[i, 1]] += ph_this_uw[i]
                        N_grid_uw[grid_ij[i, 0], grid_ij[i, 1]] += 1
                ph_grid_uw = ph_grid_uw / N_grid_uw

        if i1 == 0:
            nzix = ph_grid != 0
            n_ps_grid = np.sum(nzix)
            ph = np.zeros((n_ps_grid, n_ifg), dtype=np.complex64)

            if lowfilt_flag.lower() == "y":
                ph_lowpass = ph
            else:
                ph_lowpass = None

            if ph_in_predef is not None:
                ph_uw_predef = np.zeros((n_ps_grid, n_ifg), dtype=np.complex64)
            else:
                ph_uw_predef = None

        if goldfilt_flag.lower() == "y" or lowfilt_flag.lower() == "y":
            ph_this_gold, ph_this_low = wrap_filt(
                ph_grid, prefilt_win, gold_alpha, low_flag=lowfilt_flag
            )

            if lowfilt_flag.lower() == "y" and ph_lowpass is not None:
                ph_lowpass[:, i1] = ph_this_low[nzix]

        if goldfilt_flag.lower() == "y":
            ph[:, i1] = ph_this_gold[nzix]
        else:
            ph[:, i1] = ph_grid[nzix]

        if ph_in_predef is not None and ph_uw_predef is not None:
            ph_uw_predef[:, i1] = ph_grid_uw[nzix]
            ix = ~np.isnan(ph_uw_predef[:, i1])
            ph_diff = np.angle(ph[ix, i1] * np.conj(np.exp(1j * ph_uw_predef[ix, i1])))
            ph_diff[np.abs(ph_diff) > 1] = np.nan
            ph_uw_predef[ix, i1] = ph_uw_predef[ix, i1] + ph_diff

        # check(f"ph_grid_{i1+1}", ph_grid, atol=1e-2, rtol=1e-2)
        log(f"{i1+1:{len(str(n_ifg))}d}/{n_ifg}: "
            f"nansum(abs(ph_grid)) = {np.nansum(np.abs(ph_grid)):.2f}")

    n_ps = n_ps_grid

    log(f"Number of resampled points: {n_ps}")

    # check("ph_grid", ph_grid, atol=1e-2, rtol=1e-2)

    nz_i, nz_j = np.where(ph_grid != 0)
    if pix_size == 0:
        xy = xy_in
    else:
        xy = np.column_stack(
            (
                np.arange(1, n_ps + 1),
                (nz_j - 0.5) * pix_size,
                (nz_i - 0.5) * pix_size,
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
        grid_ij=grid_ij,
        pix_size=pix_size,
    )


def wrap_filt(
    ph_in: Array, n_win: int, alpha: float, n_pad: Optional[int] = None, low_flag: str = "n"
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
    n_win_i = int(np.ceil(n_i // n_inc) - 1)
    n_win_j = int(np.ceil(n_j // n_inc) - 1)

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

    for ix1 in range(n_win_i):
        wf = wind_func.copy()
        i1, i2 = ix1 * n_inc + 1, min(ix1 * n_inc + n_win, n_i)
        if i2 > n_i:  # Adjust the window if it exceeds the image bounds
            wf = np.pad(wf[: i2 - n_i, :], ((0, i2 - n_i), (0, 0)), "constant")

        for ix2 in range(n_win_j):
            wf2 = wf.copy()
            j1, j2 = ix2 * n_inc + 1, min(ix2 * n_inc + n_win, n_j)
            if j2 > n_j:  # Adjust the window for the horizontal dimension
                wf2 = np.pad(wf2[:, : j2 - n_j], ((0, 0), (0, j2 - n_j)), "constant")

            # Initialize the phase bit for the current window
            ph_bit = np.zeros((n_win + n_pad, n_win + n_pad), dtype=ph.dtype)
            ph_bit[:n_win, :n_win] = ph[(i1 - 1) : i2, (j1 - 1) : j2]

            # Apply FFT and filter the phase data
            ph_fft = np.fft.fft2(ph_bit)
            H = np.abs(ph_fft)
            H = ifftshift(convolve2d(fftshift(H), B, mode="same"))  # Smooth the frequency response

            meanH = np.median(H)
            if meanH != 0:
                H = (H / meanH) ** alpha
            else:
                H = H**alpha

            ph_filt = (
                np.fft.ifft2(ph_fft * H)[:n_win, :n_win] * wf2
            )  # Apply inverse FFT and window function

            # Optionally apply lowpass filtering
            if low_flag == "y":
                ph_filt_low = np.fft.ifft2(ph_fft * L)[:n_win, :n_win] * wf2  # Lowpass filter
                ph_out_low[i1 - 1 : i2, j1 - 1 : j2] += ph_filt_low

            # Update the output array with filtered data
            ph_out[i1 - 1 : i2, j1 - 1 : j2] += ph_filt

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
    I, J = np.meshgrid(np.arange(1, nrow+1), np.arange(1, ncol+1))
    PQ = np.column_stack((I.ravel(), J.ravel()))

    use_triangle = False
    if Path(TRIANGLE).exists():
        use_triangle = True

    # Make i,j the indices of the non-zero indices which are the 
    # pixel locations of the PS points in the grid (1-based indexing)
    jj, ii = np.where(nzix.T)
    ij = np.column_stack((np.arange(1, n_ps+1), ii+1, jj+1))

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
        Z = tri.find_simplex(np.column_stack((I.ravel(), J.ravel())))

    if DEBUG:
        # Hard to get the same results as the original code as there could be
        # multiple solutions for the nearest neighbors
        Z = loadmat(Path("Z.mat"))["Z"]

    # Column edges
    Zvec = Z.T.ravel()
    grid_edges = np.vstack((Zvec[:-nrow], Zvec[nrow:])).T

    # Row edges
    Zvec = Z.ravel()
    grid_edges = np.vstack((grid_edges, np.column_stack((Zvec[:-ncol], Zvec[ncol:]))))

    sort_edges = np.sort(grid_edges, axis=1)
    I_sort = np.argsort(grid_edges, axis=1)
    edge_sign = I_sort[:, 1] - I_sort[:, 0]

    _, I, J = np.unique(sort_edges, axis=0, return_index=True, return_inverse=True)
    sameix = sort_edges[:, 0] == sort_edges[:, 1]
    sort_edges[sameix, :] = 0

    _, I2, J2 = np.unique(sort_edges, axis=0, return_index=True, return_inverse=True)
    n_edge = I2.shape[0] - 1

    edgs = np.column_stack(([np.arange(1, n_edge + 1), sort_edges[I2[1:], :]]))

    gridedgeix = (J2[J] - 1) * edge_sign

    colix = gridedgeix[: nrow * (ncol - 1)].reshape(nrow, ncol - 1)
    rowix = gridedgeix[nrow * (ncol - 1) :].reshape(ncol, nrow - 1).T

    log(f"Number of unique edges in grid: {n_edge}")

    check("edgs", edgs, atol=1e-2, rtol=1e-2)

    stamps_save("uw_interp", edgs=edgs, n_edge=n_edge, rowix=rowix, colix=colix, Z=Z)

    log("Interpolation done")


def ps_calc_scla(use_small_baselines: int = 0, coest_mean_vel: int = 0) -> None:
    """Estimate spatially-correlated look angle error."""

    log("# Stage 7: Estimating spatially-correlated look angle error")

    small_baseline_flag = getparm("small_baseline_flag")
    drop_ifg_index = getparm("drop_ifg_index")
    scla_method = getparm("scla_method")
    scla_deramp = getparm("scla_deramp")
    subtr_tropo = getparm("subtr_tropo")
    tropo_method = getparm("tropo_method")

    if use_small_baselines != 0:
        if small_baseline_flag != "y":
            raise ValueError("Use small baselines requested but there are none")

    if use_small_baselines == 0:
        scla_drop_index = getparm("scla_drop_index")
    else:
        scla_drop_index = getparm("sb_scla_drop_index")
        print("   Using small baseline interferograms")

    psver = stamps_load("psver")
    psname = f"./ps{psver}.mat"
    rcname = f"./rc{psver}.mat"
    pmname = f"./pm{psver}.mat"
    bpname = f"./bp{psver}.mat"
    meanvname = f"./mv{psver}.mat"
    ifgstdname = f"./ifgstd{psver}.mat"
    phuwsbresname = f"./phuw_sb_res{psver}.mat"

    if use_small_baselines == 0:
        phuwname = f"./phuw{psver}.mat"
        sclaname = f"./scla{psver}.mat"
        apsname_old = f"./aps{psver}.mat"
        apsname = f"./tca{psver}.mat"
    else:
        phuwname = f"./phuw_sb{psver}.mat"
        sclaname = f"./scla_sb{psver}.mat"
        apsname_old = f"./aps_sb{psver}.mat"
        apsname = f"./tca_sb{psver}.mat"

    if use_small_baselines == 0:
        os.system(f"rm -f {meanvname}")

    ps = stamps_load(psname)
    assert isinstance(ps, dotdict)

    if os.path.exists(bpname):
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
        n_ifg = ps.n_image
    else:
        unwrap_ifg_index = np.setdiff1d(np.arange(1, ps.n_ifg + 1), drop_ifg_index)
        n_ifg = ps.n_ifg

    if subtr_tropo == "y":
        # Remove the tropo correction - TRAIN support
        # Recompute the APS inversion on the fly as the user might have dropped
        # SB ifgs before and needs a new update of the SM APS too.
        aps = stamps_load(apsname)
        aps_corr, fig_name_tca, tropo_method = ps_plot_tca(aps, tropo_method)
        uw["ph_uw"] += aps_corr

    if scla_deramp == "y":
        print("\n   deramping ifgs...")
        [ph_all, ph_ramp] = ps_deramp(ps, uw.ph_uw)
        uw["ph_uw"] -= ph_ramp
    else:
        ph_ramp = None

    unwrap_ifg_index = np.setdiff1d(unwrap_ifg_index, scla_drop_index)

    if os.path.exists(apsname_old):
        if subtr_tropo == "y":
            print(
                f"You are removing atmosphere twice. Do not do this, either do:\n use {apsname_old} with"
                f" subtr_tropo='n'\n remove {apsname_old} use subtr_tropo='y'"
            )
        aps = stamps_load(apsname_old)
        assert isinstance(aps, dotdict)
        uw["ph_uw"] -= aps.ph_aps_slave

    ref_ps = ps_setref()

    uw["ph_uw"] -= np.tile(np.nanmean(uw.ph_uw[ref_ps, :], axis=0), (ps.n_ps, 1))

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
            bperp_mat = np.insert(bp.bperp_mat, ps.master_ix, 0, axis=1)

        day = np.diff(ps.day[unwrap_ifg_index])
        ph = np.diff(uw.ph_uw[:, unwrap_ifg_index], axis=1)
        bperp = np.diff(bperp_mat[:, unwrap_ifg_index], axis=1)
    else:
        bperp_mat = bp.bperp_mat
        bperp = bperp_mat[:, unwrap_ifg_index]
        day = ps.ifgday[unwrap_ifg_index, 1] - ps.ifgday[unwrap_ifg_index, 0]
        ph = uw.ph_uw[:, unwrap_ifg_index]

    del bp

    bprint = np.mean(bperp)
    print(f"%d ifgs used in estimation:" % (ph.shape[1]))

    for i in range(ph.shape[1]):
        if use_small_baselines != 0:
            print(
                f"   {ps.ifgday[unwrap_ifg_index[i], 0]} to {ps.ifgday[unwrap_ifg_index[i], 1]} %5d days %5d m"
                % (day[i], round(bprint[i]))
            )
        else:
            print(
                f"   {ps.day[unwrap_ifg_index[i]]} to {ps.day[unwrap_ifg_index[i+1]]} %5d days %5d m"
                % (day[i], round(bprint[i]))
            )

    K_ps_uw = np.zeros((ps.n_ps, 1))

    if coest_mean_vel == 0 or len(unwrap_ifg_index) < 4:
        G = np.column_stack((np.ones((ph.shape[1], 1)), np.mean(bperp)))
    else:
        G = np.column_stack((np.ones((ph.shape[1], 1)), np.mean(bperp), day))

    ifg_vcm = np.eye(ps.n_ifg)

    if small_baseline_flag == "y":
        if use_small_baselines == 0:
            phuwres = stamps_load(phuwsbresname, "sm_cov")
            if "sm_cov" in phuwres:
                ifg_vcm = phuwres.sm_cov
        else:
            phuwres = stamps_load(phuwsbresname, "sb_cov")
            if "sb_cov" in phuwres:
                ifg_vcm = phuwres.sb_cov
    else:
        if os.path.exists(ifgstdname):
            ifgstd = stamps_load(ifgstdname)
            ifg_vcm = np.diag((ifgstd.ifg_std * np.pi / 180) ** 2)
            del ifgstd

    if use_small_baselines == 0:
        ifg_vcm_use = np.eye(ph.shape[1])
    else:
        ifg_vcm_use = ifg_vcm[unwrap_ifg_index - 1, unwrap_ifg_index - 1]

    m = lscov(G, ph.T, ifg_vcm_use)
    K_ps_uw = m[1, :]
    if coest_mean_vel != 0:
        v_ps_uw = m[2, :]

    if scla_method == "L1":
        for i in range(ps.n_ps):
            d = ph[i, :]
            m2 = least_squares(lambda x: d - G @ x, m[:, i]).x
            K_ps_uw[i] = m2[1]
            if i % 10000 == 0:
                print(f"%d of %d pixels processed" % (i, ps.n_ps))

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

    oldscla = glob.glob(f"{sclaname}.mat")
    if oldscla:
        olddatenum = datetime.fromtimestamp(os.path.getmtime(oldscla[0])).strftime("%Y%m%d_%H%M%S")
        shutil.move(oldscla[0], f"tmp_{sclaname[:-4]}_{olddatenum}.mat")

    stamps_save(sclaname, ph_scla, K_ps_uw, C_ps_uw, ph_ramp, ifg_vcm)


def ps_deramp(ps: dotdict, ph_all: Array, degree: Optional[int] = None) -> Tuple[Array, Array]:
    """
    Deramps the inputted data and gives that as output. Needs ps struct information!
    """

    print("Deramping computed on the fly.")

    assert isinstance(ps, dotdict)

    if degree is None:
        try:
            deree = stamps_load("deramp_degree")
            print("Found deramp_degree.mat file will use that value to deramp")
        except (FileNotFoundError, IOError):
            degree = 1

    # SM from SB inversion deramping
    if ps.n_ifg != ph_all.shape[1]:
        ps["n_ifg"] = ph_all.shape[1]

    # detrenting of the data
    if degree == 1:
        # z = ax + by + c
        A = np.column_stack((ps.xy[:, 1:] / 1000, np.ones((ps.n_ps, 1))))
        print("**** z = ax + by + c")
    elif degree == 1.5:
        # z = ax + by + cxy + d
        A = np.column_stack(
            (
                ps.xy[:, 1:] / 1000,
                (ps.xy[:, 1] / 1000) * (ps.xy[:, 2] / 1000),
                np.ones((ps.n_ps, 1)),
            )
        )
        print("**** z = ax + by + cxy + d")
    elif degree == 2:
        # z = ax^2 + by^2 + cxy + d
        A = np.column_stack(
            (
                ((ps.xy[:, 1:] / 1000) ** 2),
                (ps.xy[:, 1] / 1000) * (ps.xy[:, 2] / 1000),
                np.ones((ps.n_ps, 1)),
            )
        )
        print("**** z = ax^2 + by^2 + cxy + d")
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
        print("**** z = ax^3 + by^3 + cx^2y + dy^2x + ex^2 + fy^2 + gxy + h")
    else:
        raise ValueError("Invalid degree value. Expected 1, 1.5, 2, or 3.")

    ph_ramp = np.full(ph_all.shape, np.nan)
    for k in range(ps.n_ifg):
        ix = np.isnan(ph_all[:, k])
        if ps.n_ps - np.sum(ix) > 5:
            coeff = lscov(A[~ix, :], ph_all[~ix, k])
            ph_ramp[:, k] = np.dot(A, coeff)
            ph_all[:, k] -= ph_ramp[:, k]
        else:
            print(f"Ifg {k + 1} is not deramped")

    return ph_all, ph_ramp


def lscov(A: np.ndarray, B: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Solves the weighted least squares problem A * x = B.

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

    # log(f"lscov {Aw.shape=}, {Bw.shape=}")

    # Solve the least squares problem
    x, _, _, _ = np.linalg.lstsq(Aw, Bw, rcond=None)

    return np.array(x)


def test_getparm() -> None:
    """Test the getparm function."""
    log("Testing getparm function")
    getparm()


def test_stage1() -> None:
    log("Testing Stage 1")
    cwd = Path.cwd()
    new = patchdirs()[0]
    os.chdir(new)
    stage1_load_data()
    assert results_equal("ps1")
    assert results_equal("ph1")
    assert results_equal("bp1")
    assert results_equal("la1")
    assert results_equal("da1")
    assert results_equal("hgt1")
    os.chdir(cwd)
    log("Stage 1 test passed")


def test_stage2() -> None:
    log("Testing Stage 2")
    cwd = Path.cwd()
    new = patchdirs()[0]
    os.chdir(new)
    stage2_estimate_noise()
    assert results_equal("pm1", atol=1e-3, rtol=1e-3)
    os.chdir(cwd)
    log("Stage 2 test passed")


def test_stage3() -> None:
    log("Testing Stage 3")
    cwd = Path.cwd()
    new = patchdirs()[0]
    os.chdir(new)
    stage3_select_ps()
    assert results_equal("select1", atol=1e-2, rtol=1e-2)
    os.chdir(cwd)
    log("Stage 3 test passed")


def test_stage4() -> None:
    log("Testing Stage 4")
    cwd = Path.cwd()
    new = patchdirs()[0]
    os.chdir(new)
    stage4_weed_ps()
    assert results_equal("weed1", atol=1e-2, rtol=1e-2)
    assert results_equal("pm2", atol=1e-2, rtol=1e-2)
    assert results_equal("ps2", atol=1e-2, rtol=1e-2)
    assert results_equal("hgt2", atol=1e-2, rtol=1e-2)
    assert results_equal("la2", atol=1e-2, rtol=1e-2)
    os.chdir(cwd)
    log("Stage 4 test passed")


def test_stage5() -> None:
    log("Testing Stage 5")
    cwd = Path.cwd()
    new = patchdirs()[0]
    os.chdir(new)
    stage5_correct_phases()
    assert results_equal("rc1", atol=1e-2, rtol=1e-2)
    os.chdir(cwd)
    log("Stage 5 test passed")


def test_stage6() -> None:
    log("Testing Stage 6")
    cwd = Path.cwd()
    new = patchdirs()[0]
    os.chdir(new)
    stage6_unwrap_phases()
    assert results_equal("phuw1", atol=1e-2, rtol=1e-2)
    os.chdir(cwd)
    log("Stage 6 test passed")


def test_ps_calc_ifg_std() -> None:
    log("Testing ps_calc_ifg_std function")
    cwd = Path.cwd()
    new = patchdirs()[0]
    os.chdir(new)
    ps_calc_ifg_std()
    assert results_equal("ifgstd1", atol=1e-2, rtol=1e-2)
    os.chdir(cwd)


def test_interp() -> None:
    log("Testing interp function 1")
    x = np.arange(1, 10, dtype=np.float64)
    y = interp(x, 2)
    assert np.allclose(
        y,
        np.array(
            [
                1.0000,
                1.4996,
                2.0000,
                2.4993,
                3.0000,
                3.4990,
                4.0000,
                4.4987,
                5.0000,
                5.4984,
                6.0000,
                6.4982,
                7.0000,
                7.4979,
                8.0000,
                8.4976,
                9.0000,
                9.4973,
            ]
        ),
        rtol=1e-4,
        atol=1e-4,
    )
    log("Testing interp function 2")
    x = np.ones(100)
    y = interp(x, 10)
    assert np.allclose(y, np.ones(1000), atol=1e-2, rtol=1e-2)
    log("Testing interp function 3")
    t = np.linspace(0, 1, endpoint=True, num=1001)
    x = np.sin(2 * np.pi * 30 * t) + np.sin(2 * np.pi * 60 * t)
    y = interp(x, 4)
    yy = loadmat(Path("interp.mat"))["y"]
    assert np.allclose(y, yy, atol=1e-2, rtol=1e-2)
    log("Testing interp function 4")
    cwd = Path.cwd()
    os.chdir(patchdirs()[0])
    x = loadmat(Path("Prand_1.mat"))["Prand"]
    y = interp(np.insert(x, 0, 1), 10)[:-9]
    z = loadmat(Path("Prand_after_1.mat"))["Prand"]
    assert np.allclose(y, z, atol=1e-2, rtol=1e-2)
    os.chdir(cwd)
    log("Testing interp function 5")
    cwd = Path.cwd()
    os.chdir(patchdirs()[0])
    x = loadmat(Path("Prand_2.mat"))["Prand"]
    y = interp(np.insert(x, 0, 1), 10)[:-9]
    z = loadmat(Path("Prand_after_2.mat"))["Prand"]
    assert np.allclose(y, z, atol=1e-2, rtol=1e-2)
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


def run_matlab() -> None:
    result = subprocess.check_output(
        ["/Applications/MATLAB_R2023b.app/bin/matlab", "-nodisplay"],
        input="addpath ../../StaMPS/matlab\n" "stamps(1,1)\n" "exit\n",
        text=True,
    )
    print(result)


def test_uw_interp() -> None:
    log("Testing uw_interp function")
    cwd = Path.cwd()
    new = patchdirs()[0]
    os.chdir(new)
    uw_interp()
    assert results_equal("uw_interp", atol=1e-2, rtol=1e-2)
    os.chdir(cwd)


def run_tests() -> None:
    # test_dates()
    # test_interp()
    # test_stage1()
    # test_estimate_coherence()
    # test_ps_select()
    # test_ps_weed()
    # test_ps_correct_phase()
    # test_ps_calc_ifg_std()
    test_uw_interp()
    log("\nAll tests passed!\n")


if __name__ == "__main__":
    run_tests()
