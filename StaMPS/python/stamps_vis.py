#!/usr/bin/env python3
"""
Visualise the results of the STaMPS methodology. Many of these
visualisations are not found in the original STaMPS MATLAB code.

This code is seperate from the main program to avoid dependencies
on matplotlib in the main program. This also allows us to
quickly generate and tweak figures without waiting for the main
program to run.

The approach is that data can be saved in the main program using
`stamps_save` and then loaded here using `stamps_load`.

This code has been written by Dale Roberts.
"""

import matplotlib.pyplot as plt
import numpy as np

from stamps import dotdict, patchdirs, datestr, stamps_load
from contextlib import chdir
from typing import Tuple


def lonlat_to_xy(lon: float, lat: float, zoom: int = 10) -> Tuple[int, int]:
    """Convert longitude, latitude to tile coordinates."""
    n = 2**zoom
    x_tile = int(n * ((lon + 180) / 360))
    y_tile = int(
        n * (1 - (np.log(np.tan(np.radians(lat)) + 1 / np.cos(np.radians(lat))) / np.pi)) / 2
    )
    return x_tile, y_tile


def fig_initial_ps(patch: str) -> None:
    ps = stamps_load("ps1")
    assert isinstance(ps, dotdict)

    ph = stamps_load("ph1")
    assert isinstance(ph, np.ndarray)

    print(ph.shape)

    lon = ps.lonlat[:, 0]
    lat = ps.lonlat[:, 1]

    zoom = 10

    lon_min, lon_max = np.min(lon), np.max(lon)
    lat_min, lat_max = np.min(lat), np.max(lat)

    x_min, y_max = lonlat_to_xy(lon_min, lat_max, zoom)
    x_max, y_min = lonlat_to_xy(lon_max, lat_min, zoom)

    fig, ax = plt.subplots(figsize=(10, 10))

    # color the PS candidates based on 'ph' values using a spectral colormap
    ax.scatter(lon, lat, c=ph[:, -1], cmap="Spectral", marker=".", s=0.5)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Add a colorbar
    cbar = plt.colorbar(ax.collections[0], ax=ax, orientation="vertical")
    cbar.set_label("Phase")

    # import urllib.request
    # import matplotlib.image as mpimg
    # from io import BytesIO
    # url = f"https://tile.openstreetmap.org/{zoom}/{x_min}/{y_min}.png"
    ##url = f"https://mt1.google.com/vt/lyrs=s&x={x_min}&y={y_min}&z={zoom}"
    # print(url)

    # req = urllib.request.Request(url, headers={'User-Agent': 'Safari/537.36'})
    # with urllib.request.urlopen(req) as response:
    #    data = response.read()
    #    img = np.asarray(mpimg.imread(BytesIO(data), format='png'))
    #    ax.imshow(img, extent=[lon_min, lon_max, lat_min, lat_max], aspect='auto')

    plt.title(f"Initial PS candidates ({patch})")
    plt.show()


def fig_initial_ps_mesh(patch: str) -> None:
    ps = stamps_load("ps1")
    assert isinstance(ps, dotdict)

    ph = stamps_load("ph1")
    assert isinstance(ph, np.ndarray)

    print(ph.shape)

    lon = ps.lonlat[:, 0]
    lat = ps.lonlat[:, 1]

    zoom = 10

    lon_min, lon_max = np.min(lon), np.max(lon)
    lat_min, lat_max = np.min(lat), np.max(lat)

    x_min, y_max = lonlat_to_xy(lon_min, lat_max, zoom)
    x_max, y_min = lonlat_to_xy(lon_max, lat_min, zoom)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(lon, lat, ".", color="magenta", markersize=0.5)
    ax.tripcolor(lon, lat, np.angle(ph[:, -1]), cmap="twilight")

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Add a colorbar
    cbar = plt.colorbar(ax.collections[0], ax=ax, orientation="vertical")
    cbar.set_label("Phase angle (radians)")

    # Set labels for colorbar specify $-\pi$ to $\pi$
    cbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cbar.set_ticklabels([r"$-\pi$", r"$-\pi/2$", r"0", r"$\pi/2$", r"$\pi$"])

    plt.title(f"Initial PS candidates ({patch})")
    plt.show()


def fig_initial_dispersion_mesh(patch: str) -> None:
    ps = stamps_load("ps1")
    assert isinstance(ps, dotdict)

    da = stamps_load("da1")
    assert isinstance(da, np.ndarray)

    lon = ps.lonlat[:, 0]
    lat = ps.lonlat[:, 1]

    fig, ax = plt.subplots(figsize=(10, 5))

    # ax.plot(lon, lat, '.', color="black", markersize=0.1)
    ax.tripcolor(lon, lat, da, cmap="bone")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Add a colorbar
    cbar = plt.colorbar(ax.collections[0], ax=ax, orientation="vertical")
    cbar.set_label(r"Dispersion / Fano factor ($\sigma^2/\mu$)")

    plt.title(f"Dispersion ($\\sigma^2/\\mu$) of PS candidates ({patch})")
    plt.show()


def fig_stage2(patch: str) -> None:
    ps = stamps_load("ps1")
    assert isinstance(ps, dotdict)

    ph = stamps_load("ph1")
    assert isinstance(ph, np.ndarray)

    pm = stamps_load("pm1")
    assert isinstance(pm, dotdict)

    fig, ax = plt.subplots(figsize=(10, 10))

    # delete the master PS candidate
    ph = np.delete(ph, ps.master_ix, axis=1)

    psdph = np.angle(ph * np.conj(pm.ph_patch))
    mean_psdph = np.mean(psdph, axis=1)

    N = 160
    bottom = 0
    max_height = 1

    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    width = (2 * np.pi) / N

    radii = np.histogram(mean_psdph, bins=theta)[0]
    radii = radii / np.max(radii) * max_height

    ax = plt.subplot(111, polar=True)
    bars = ax.bar(theta[:-1], radii, width=width, bottom=bottom)
    for r, bar in zip(radii, bars):
        bar.set_facecolor(plt.get_cmap("Spectral")(r / 10.0))
        bar.set_alpha(0.8)

    ax.spines["polar"].set_visible(False)
    ax.set_yticklabels([])

    plt.show()


def fig_ps_height_vs_phase(patch: str) -> None:
    ps = stamps_load("ps1")
    assert isinstance(ps, dotdict)

    ph = stamps_load("ph1")
    assert isinstance(ph, np.ndarray)

    hgt = stamps_load("hgt1")
    assert isinstance(hgt, np.ndarray)

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(hgt, ph[:, -1], marker=".", s=0.5)
    ax.set_xlabel("Height")
    ax.set_ylabel("Phase")

    plt.title(f"Height vs. Phase for PS candidates ({patch})")
    plt.show()


def fig_ps_phase_vs_prev_phase(patch: str) -> None:
    ps = stamps_load("ps1")
    assert isinstance(ps, dotdict)

    ph = stamps_load("ph1")
    assert isinstance(ph, np.ndarray)

    fig, ax = plt.subplots(figsize=(10, 3))

    dates = [f"{datestr(d1)} - {datestr(d2)}" for d1, d2 in zip(ps.day[:-1], ps.day[1:])]
    diff = np.diff(ph, axis=1)

    parts = ax.violinplot(
        np.abs(diff), showmeans=False, showmedians=False, showextrema=False, widths=1.0
    )
    for pc in parts["bodies"]:
        pc.set_facecolor("magenta")
        pc.set_edgecolor("magenta")
        pc.set_alpha(0.5)

    ax.set_xticks(range(1, len(dates) + 1))
    ax.set_xticklabels(dates, rotation=90)
    ax.set_xlabel("Date")
    ax.set_ylabel("Phase difference")
    plt.xticks(fontsize=6)
    plt.subplots_adjust(bottom=0.3)

    plt.title(f"Absolute Phase at $t-1$ vs. Phase at $t$ for PS candidates ({patch})")
    plt.show()


def fig_ps_phase_vs_master(patch: str) -> None:
    ps = stamps_load("ps1")
    assert isinstance(ps, dotdict)

    ph = stamps_load("ph1")
    assert isinstance(ph, np.ndarray)

    m, n = 6, 6

    fig, ax = plt.subplots(figsize=(10, 10 + 1), ncols=m, nrows=n)

    dates = [f"{datestr(d)}" for d in ps.day]
    master_ix = ps.master_ix

    # diff of ph with master
    ph = ph - ph[:, master_ix][:, None]

    center_real = np.mean(np.real(ph))
    center_imag = np.mean(np.imag(ph))
    sdt_real = np.std(np.real(ph))
    sdt_imag = np.std(np.imag(ph))

    print(f"Center: {center_real} + {center_imag}i")

    x_min, x_max = center_real - sdt_real, center_real + sdt_real
    y_min, y_max = center_imag - sdt_imag, center_imag + sdt_imag

    # plot hist2d of phase vs master phase
    for i in range(m):
        for j in range(n):
            k = i * m + j
            ax[i, j].hist2d(
                np.real(ph[:, i]),
                np.imag(ph[:, k]),
                bins=30,
                cmap="bone_r",
                range=[[x_min, x_max], [y_min, y_max]],
            )
            ax[i, j].set_title(f"{dates[k]} vs. master", fontsize=6)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].axvline(x=center_real, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
            ax[i, j].axhline(y=center_imag, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
            ax[i, j].set_xlim(center_real - sdt_real, center_real + sdt_real)
            ax[i, j].set_ylim(center_imag - sdt_imag, center_imag + sdt_imag)

    ax[m - 1, 0].set_xlabel("Real", fontsize=6)
    ax[m - 1, 0].set_ylabel("Imag", fontsize=6)

    plt.suptitle(f"Phase vs. Master Phase for PS candidates ({patch})")

    plt.tight_layout()

    plt.show()


def export_ps_geojson(patch: str) -> None:
    """Export the PS candidates to a GeoJSON file."""
    import json

    ps = stamps_load("ps1")
    assert isinstance(ps, dotdict)

    geojson = {"type": "FeatureCollection", "features": []}

    for i in range(len(ps.lonlat)):
        lon, lat = ps.lonlat[i]
        features = geojson.get("features")
        assert isinstance(features, list)
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "id": i,
                },
            }
        )

    with open("ps.geojson", "w") as f:
        json.dump(geojson, f, indent=4)


def fig_coh_histogram(patch: str) -> None:
    pm = stamps_load("pm1")
    select = stamps_load("select1")

    assert isinstance(pm, dotdict)
    assert isinstance(select, dotdict)

    plt.figure()
    plt.hist(
        pm.coh_ps,
        bins=50,
        color="magenta",
        alpha=0.4,
        edgecolor="darkmagenta",
    )

    plt.hist(
        select.coh_ps2,
        bins=50,
        color="cyan",
        alpha=0.4,
        edgecolor="darkblue",
    )

    for ct in select.coh_thresh:
        plt.axvline(x=ct, color="k", linestyle="--", label="coh_thresh")
        plt.text(ct, 1000, "coh_thresh", rotation=90)
    plt.title(f"Distribution of coherence values for PS candidates ({patch})")
    plt.show()


def fig_bperp(patch: str) -> None:
    ps = stamps_load("ps1")
    assert isinstance(ps, dotdict)

    # Plot 2 x 1 figure
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    mean_bperp = np.mean(ps.bperp)
    std_bperp = np.std(ps.bperp)

    # Convert timestamp64 to date strings
    dates = datestr(ps.day)
    # Draw dots and lines
    ax[0].scatter(dates, ps.bperp, color="black", s=8)
    ax[0].plot(dates, ps.bperp, color="black", linestyle="-", linewidth=0.5)
    ax[0].set_title(f"Time series of bperp for PS candidates ({patch})")
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("bperp")

    ax[0].axhline(
        y=mean_bperp,
        color="magenta",
        linestyle="--",
        label=f"Mean bperp: {mean_bperp:.2f}",
        linewidth=1.0,
    )

    # Plot lines for mean +/- 1 std
    ax[0].axhline(
        y=mean_bperp - std_bperp,
        color="magenta",
        linestyle="--",
        label=f"Mean - 1 std: {mean_bperp - std_bperp:.2f}",
        linewidth=0.5,
    )
    ax[0].axhline(
        y=mean_bperp + std_bperp,
        color="magenta",
        linestyle="--",
        label=f"Mean + 1 std: {mean_bperp + std_bperp:.2f}",
        linewidth=0.5,
    )
    # Plot labels
    ax[0].legend()

    # Set plot range to mean +/- 3 std
    ax[0].set_ylim(mean_bperp - 3 * std_bperp, mean_bperp + 3 * std_bperp)

    # Plot histogram of bperp
    ax[1].hist(ps.bperp, bins=50, color="magenta", alpha=0.4, edgecolor="darkmagenta")
    ax[1].set_title(f"Distribution of mean perpendicular baseline values ({patch})")
    ax[1].set_xlabel("bperp")
    ax[1].set_ylabel("Frequency")

    plt.show()


def fig_incidence_angles(patch: str) -> None:
    ps = stamps_load("ps1")
    assert isinstance(ps, dotdict)

    la = stamps_load("la1")
    assert isinstance(la, np.ndarray)

    min_la = np.min(la)
    max_la = np.max(la)

    print(f"Min incidence angle: {min_la}")
    print(f"Max incidence angle: {max_la}")

    N = 160
    bottom = 0
    max_height = 1

    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    width = (2 * np.pi) / N

    radii = np.histogram(la, bins=theta)[0]
    radii = radii / np.max(radii) * max_height

    ax = plt.subplot(111, polar=True)
    bars = ax.bar(theta[:-1], radii, width=width, bottom=bottom)
    for r, bar in zip(radii, bars):
        bar.set_facecolor(plt.get_cmap("Spectral")(r / 10.0))
        bar.set_alpha(0.8)

    ax.spines["polar"].set_visible(False)
    ax.set_yticklabels([])

    ax.set_title(f"Distribution of incidence angles for PS candidates ({patch})")

    plt.show()


def fig_unwrap_points(patch: str) -> None:
    ps = stamps_load("ps1")
    assert isinstance(ps, dotdict)

    phuw = stamps_load("phuw1")
    assert isinstance(phuw, dotdict)

    ph_uw = phuw.ph_uw

    lon = ps.lonlat[:, 0]
    lat = ps.lonlat[:, 1]

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(lon, lat, c=ph_uw[:, -1], cmap="Spectral", marker=".", s=0.5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Add a colorbar
    cbar = plt.colorbar(ax.collections[0], ax=ax, orientation="vertical")
    cbar.set_label("Unwrapped phase")

    plt.title(f"Unwrapped phase for PS candidates ({patch})")
    plt.show()


def fig_unwrap_mesh(patch: str) -> None:
    ps = stamps_load("ps1")
    assert isinstance(ps, dotdict)

    phuw = stamps_load("phuw1")
    assert isinstance(phuw, dotdict)

    dates = datestr(ps.day)
    assert isinstance(dates, np.ndarray)

    ph_uw = phuw.ph_uw
    msd = phuw.msd

    lon = ps.lonlat[:, 0]
    lat = ps.lonlat[:, 1]

    m, n = 5, 5

    pl, pu = np.nanpercentile(ph_uw, [2, 98])
    pl = min(pl, -pu)
    pu = max(pu, -pl)
    print(f"Percentiles: {pl}, {pu}")

    ph_uw[np.isnan(ph_uw)] = 0

    fig, ax = plt.subplots(figsize=(10, 10 + 1), ncols=m, nrows=n)

    for i in range(m):
        for j in range(n):
            k = i * m + j
            ax[i, j].tripcolor(lon, lat, ph_uw[:, k], cmap="PiYG", vmin=pl, vmax=pu)
            ax[i, j].set_title(f"{dates[k]} MSD={msd[k]:.3f}", fontsize=8)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

    cbar = plt.colorbar(ax[0, 0].collections[0], ax=ax, orientation="vertical")
    cbar.set_label("Unwrapped phase (radians)")

    plt.suptitle(f"Unwrapped phase for PS candidates ({patch})")

    show(f"unwrap_mesh_{patch}")


def fig_wrap_vs_unwrap(patch: str) -> None:
    ps = stamps_load("ps1")
    assert isinstance(ps, dotdict)

    ph = stamps_load("ph1")
    assert isinstance(ph, np.ndarray)

    phuw = stamps_load("phuw1")
    assert isinstance(phuw, dotdict)

    ph_uw = phuw.ph_uw

    pl, pu = np.nanpercentile(ph_uw, [10, 90])
    pl = min(pl, -pu)
    pu = max(pu, -pl)
    print(f"Percentiles: {pl}, {pu}")

    ph_uw[np.isnan(ph_uw)] = 0

    lon = ps.lonlat[:, 0]
    lat = ps.lonlat[:, 1]

    fig, ax = plt.subplots(figsize=(13, 5 + 1), nrows=1, ncols=2)

    # set axes background to light gray
    ax[0].set_facecolor("lightgray")
    ax[1].set_facecolor("lightgray")

    ax[0].scatter(lon, lat, c=np.angle(ph[:, 1]), cmap="jet_r", marker=".", s=0.5)
    ax[1].scatter(lon, lat, c=ph_uw[:, 1], cmap="jet_r", marker=".", s=0.5, vmin=pl, vmax=pu)

    # add colorbars

    cbar = plt.colorbar(ax[0].collections[0], ax=ax[0], orientation="vertical")
    cbar.set_label("Wrapped phase (radians)")
    cbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cbar.set_ticklabels([r"$-\pi$", r"$-\pi/2$", r"0", r"$\pi/2$", r"$\pi$"])

    cbar = plt.colorbar(ax[1].collections[0], ax=ax[1], orientation="vertical")
    cbar.set_label("Unwrapped phase (radians)")

    plt.suptitle(f"Wrapped vs. unwrapped phase ({patch})")

    plt.tight_layout()

    show("wrap_vs_unwrap")


def show(figname: str) -> None:
    plt.savefig(f"{figname}")
    plt.show()


def set_fig_params() -> None:
    plt.rcParams["savefig.format"] = "png"
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 8


def generate_figures() -> None:
    set_fig_params()

    for p in patchdirs():
        with chdir(p):
            # export_ps_geojson(str(p))
            # fig_initial_ps(str(p))
            # fig_coh_histogram(str(p))
            # fig_bperp(str(p))
            # fig_ps_height_vs_phase(str(p))
            # fig_ps_phase_vs_prev_phase(str(p))
            # fig_ps_phase_vs_master(str(p))
            # fig_incidence_angles(str(p))
            # fig_initial_ps_mesh(str(p))
            # fig_initial_dispersion_mesh(str(p))
            # fig_stage2(str(p))
            # fig_unwrap_mesh(str(p))
            fig_wrap_vs_unwrap(str(p))


if __name__ == "__main__":
    generate_figures()
