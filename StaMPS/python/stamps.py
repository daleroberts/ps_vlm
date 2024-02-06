#!/usr/bin/env python3
"""

"""

import scipy.io as sio
import numpy as np
import sys
import os

from pathlib import Path
from datetime import datetime
from typing import Any, Tuple, Optional, List
from joblib import dump, load


def log(msg: str) -> None:
    """Prints a message to stderr."""
    print(msg, file=sys.stderr)


def stamps_save(fn, *args, **kwargs):
    print(kwargs)


def llh2local(llh, origin):
    """
    Converts from longitude and latitude to local coordinates given an origin.
    llh (lon, lat, height) and origin should be in decimal degrees.
    Note that heights are ignored and that xy is in km.
    """

    # Set ellipsoid constants (WGS84)
    a = 6378137.0
    e = 0.08209443794970

    # Convert to radians
    llh = np.double(llh) * np.pi / 180
    origin = np.double(origin) * np.pi / 180

    # Initialize xy array
    xy = np.zeros((2, llh.shape[1]))

    # Do the projection
    z = llh[1, :] != 0
    dlambda = llh[0, z] - origin[0]

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

    xy[0, z] = N * 1 / np.tan(llh[1, z]) * np.sin(E)
    xy[1, z] = M - M0 + N * 1 / np.tan(llh[1, z]) * (1 - np.cos(E))

    # Handle special case of latitude = 0
    dlambda = llh[0, ~z] - origin[0]
    xy[0, ~z] = a * dlambda
    xy[1, ~z] = -M0

    # Convert to km
    xy = xy / 1000

    return xy


def loadmat(fname: Path) -> Any:
    """Loads a .mat file."""
    mat = sio.loadmat(str(fname), squeeze_me=True)
    kvs = {k: v for k, v in mat.items() if not k.startswith("__")}
    for k in kvs.keys():
        try:
            v = kvs[k]
            v = v.flat[0] if isinstance(v, np.ndarray) else v
        except IndexError:
            pass
        kvs[k] = v
    return kvs


def getparm(parmname: Optional[str] = None, verbose: bool = False) -> Tuple[Optional[Any], Optional[str]]:
    """Retrieves a parameter value from parameter files."""

    # TODO: Add support for other file formats (txt?)

    def pprint(k, v):
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
        localparms = {"Created": "today"}  # Placeholder for creation date

    value = None
    if parmname is None:
        for k, v in parms.items():
            pprint(k, v)
        if len(localparms) > 1:
            log(localparms)
    else:
        # Find the parameter in global or local parameters
        parmnames = [k for k in parms if not k.startswith("__")]
        matches = [pn for pn in parmnames if pn.startswith(parmname)]
        if len(matches) > 1:
            raise ValueError(f"Parameter {parmname}* is not unique")
        elif not matches:
            return None, None
        else:
            parmname = matches[0]
            value = localparms.get(parmname, parms.get(parmname))

        if verbose:
            pprint(parmname, value)

    return value, parmname


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


def readparm(fname: Path, parm: str, numval: int = 1) -> Any | List[Any]:
    """Reads a parameter value(s) from a file."""
    with fname.open("r") as f:
        lines = f.readlines()

    def convert(val: str) -> Any:
        for conv in (int, float):
            try:
                return conv(val)
            except ValueError:
                pass
        return val

    if numval == 1:
        for line in lines:
            if line.startswith(parm):
                return convert(line.split()[1])
    else:
        values = []
        for line in lines:
            if line.startswith(parm):
                values = [convert(x) for x in line.split()[1:]]
                return values[:numval]


def patchdirs() -> List[Path]:
    """Get the patch directories."""
    dirs = []

    if Path("patch.list").exists():
        # read directory names from patch.list
        with Path("patch.list").open("r") as f:
            dirs = [Path(line.strip()) for line in f.readlines()]
    else:
        # if patch.list does not exist, find all directories with PATCH_ in the name
        dirs = [d for d in Path(".").iterdir() if d.is_dir() and "PATCH_" in d.name]

    if len(dirs) == 0:
        # patch directories not found, use current directory
        dirs.append(Path("."))

    return dirs


def load_initial_gamma(endian: str = "b") -> None:
    """
    Load all the data we need from GAMMA outputs.
    """
    phname = Path("./pscands.1.ph")  # phase data
    ijname = Path("./pscands.1.ij")  # pixel location data
    llname = Path("./pscands.1.ll")  # latitude, longitude data
    xyname = Path("./pscands.1.xy")  #
    hgtname = Path("./pscands.1.hgt")  # height data
    daname = Path("./pscands.1.da")
    rscname = Path("../rsc.txt")
    pscname = Path("../pscphase.in")

    # Read master day from rsc file
    with rscname.open() as f:
        rslcpar = Path(f.readline().strip())
        master_day = datetime.strptime(rslcpar.name[:8], "%Y%m%d").date()

    log(f"{rslcpar = }")
    log(f"{master_day = }")

    # Read interferogram dates
    with pscname.open() as f:
        ifgs = [Path(line.strip()) for line in f.readlines()][1:]

    ifgday = np.array([[ifg.name[:8], ifg.name[9:17]] for ifg in ifgs], dtype="datetime64")
    n_ifg = len(ifgs)
    days, ifgday_ix = np.unique(ifgday, return_inverse=True)
    ifgday_ix = ifgday_ix.reshape(n_ifg, 2)   

    # Save interferogram dates
    np.savetxt("../small_baselines.list", ifgday, fmt="%s")

    # Set and save heading parameter
    heading = readparm(rslcpar, "heading")
    setparm("heading", heading)

    freq = readparm(rslcpar, "radar_frequency")
    lam = 299792458 / freq
    setparm("lambda", lam)

    sensor = readparm(rslcpar, "sensor")
    platform = sensor  # S1 case
    setparm("platform", platform)

    rps = readparm(rslcpar, "range_pixel_spacing")
    rgn = readparm(rslcpar, "near_range_slc")
    se = readparm(rslcpar, "sar_to_earth_center")
    re = readparm(rslcpar, "earth_radius_below_sensor")
    rgc = readparm(rslcpar, "center_range_slc")
    naz = readparm(rslcpar, "azimuth_lines")
    prf = readparm(rslcpar, "prf")

    mean_az = naz / 2 - 0.5

    # Processing of the phase data
    with phname.open("rb") as f:
        ph = np.fromfile(f, dtype=np.float32)
        ph = ph.view(np.complex64)  # Interpret the float32 pairs as complex numbers

    # Processing of the id, azimuth, range data
    with ijname.open("rb") as f:
        ij = np.loadtxt(f, dtype=np.int32)

    n_ps = len(ij)

    # Processing of the longitude and latitude data
    with llname.open("rb") as f:
        lonlat = np.fromfile(f, dtype=">f4").reshape((-1, 2))

    # Processing of the Height data
    if hgtname.exists():
        with hgtname.open("rb") as f:
            hgt = np.fromfile(f, dtype=np.float32)
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
        B_TCN = readparm(basename, "initial_baseline(TCN)", 3)
        BR_TCN = readparm(basename, "initial_baseline_rate", 3)
        bc = B_TCN[1] + BR_TCN[1] * (ij[:, 1] - mean_az) / prf
        bn = B_TCN[2] + BR_TCN[2] * (ij[:, 1] - mean_az) / prf
        # Convert baselines from (T)CN to perpendicular-parallel coordinates
        bperp_mat[:, i] = bc * np.cos(look) - bn * np.sin(look)
        # log(f"{i = }\n\t{basename = }\n\t{B_TCN = }\n\t{BR_TCN = }\n\t{bc = }\n\t{bn = }")

    # Calculate mean perpendicular baseline
    bperp = np.mean(bperp_mat, axis=1)

    log(f"{bperp = }")

    # Find center longitude and latitude
    ll0 = (np.max(lonlat, axis=0) + np.min(lonlat, axis=0)) / 2

    log(f"{ll0 = }")

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

    log(f"{bl = }\n{tl = }\n{br = }\n{tr = }")

    # Calculate rotation angle
    theta = (180 - heading) * np.pi / 180
    if theta > np.pi:
        theta -= 2 * np.pi

    log(f"{theta = }")

    # Rotation matrix
    rotm = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

    # Rotate coordinates
    xynew = rotm @ xy.T

    # Check if rotation improves alignment and apply if it does
    if (np.max(xynew[0]) - np.min(xynew[0]) < np.max(xy[0]) - np.min(xy[0])) and (
        np.max(xynew[1]) - np.min(xynew[1]) < np.max(xy[1]) - np.min(xy[1])
    ):
        xy = xynew.T

    # Convert xy to single precision after rotation
    xy = np.array(xy, dtype=np.float32)

    # Sort xy in ascending y, then x order and apply sort index to other arrays
    sort_ix = np.lexsort((xy[:, 0], xy[:, 1]))
    xy = xy[sort_ix]
    ph = ph[sort_ix]
    ij = ij[sort_ix]
    lonlat = lonlat[sort_ix]
    bperp_mat = bperp_mat[sort_ix]

    # Update ij with new point IDs and round xy to nearest mm
    ij[:, 0] = np.arange(1, n_ps + 1)
    xy[:, 1:] = np.round(xy[:, 1:] * 1000) / 1000

    psver = 1

    stamps_save(
        f"ps{psver}.pkl",
        ij=ij,
        lonlat=lonlat,
        xy=xy,
        bperp=bperp,
        day=ifgday,
        master_day=master_day,
        master_ix=master_ix,
        ifgday=ifgday,
        ifgday_ix=ifgday_ix,
        n_ifg=n_ifg,
        n_image=n_image,
        n_ps=n_ps,
        sort_ix=sort_ix,
        ll0=ll0,
        mean_incidence=mean_incidence,
        mean_range=mean_range,
    )


def test_getparm() -> None:
    """Test the getparm function."""
    getparm()


def test_load_initial_gamma() -> None:
    """Test the load_initial_gamma function."""
    cwd = Path.cwd()
    new = patchdirs()[0]
    os.chdir(new)
    load_initial_gamma()
    os.chdir(cwd)


if __name__ == "__main__":
    test_load_initial_gamma()
