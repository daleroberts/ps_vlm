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


def log(msg: str) -> None:
    """Prints a message to stderr."""
    print(msg, file=sys.stderr)


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
    phname = Path("./pscands.1.ph") # phase data
    ijname = Path("./pscands.1.ij") # pixel location data
    llname = Path("./pscands.1.ll") # latitude, longitude data
    xyname = Path("./pscands.1.xy") #
    hgtname = Path("./pscands.1.hgt") # height data
    daname = Path("./pscands.1.da")
    rscname = Path("../rsc.txt")
    pscname = Path("../pscphase.in")

    # Read master day from rsc file
    with rscname.open() as f:
        rslcpar = Path(f.readline().strip())
        master_day = datetime.strptime(rslcpar.name[:8], "%Y%m%d")

    log(f"{rslcpar = }")
    log(f"{master_day = }")

    # Read interferogram dates
    with pscname.open() as f:
        ifgs = [Path(line.strip()) for line in f.readlines()][1:]
        log(f"{ifgs = }")
        # ifgday = np.array([[datetime.strptime(ifg.name[:8], '%Y%m%d').date(), datetime.strptime(ifg.name[9:17], '%Y%m%d').date()] for ifg in ifgs])
        ifgday = np.array([[ifg.name[:8], ifg.name[9:17]] for ifg in ifgs], dtype="datetime64")

    # Save interferogram dates
    np.savetxt("../small_baselines.list", ifgday, fmt="%s")
    # TODO: Ensure correct datetime formatting for np.savetxt

    # Set and save parameters
    heading = readparm(rslcpar, "heading")
    setparm("heading", heading)

    getparm()

    # Load and process phase, location, and height data
    # Placeholder for actual data loading and processing
    # TODO: Add data loading and processing using numpy and other necessary libraries

    log("Data loading complete.")


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
