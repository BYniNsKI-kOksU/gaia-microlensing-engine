"""Interactive 3D Milky Way map from Gaia DR3 data.

This script queries a random Gaia DR3 sample, converts the stars to a
galactocentric Cartesian frame, bins them into 3D voxels, and exports a
self-contained Plotly HTML file for browser-based exploration.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import plotly.graph_objects as go
from astroquery.gaia import Gaia


SUN_GALACTOCENTRIC_X_KPC = -8.2
DEFAULT_TOP = 3_000_000
DEFAULT_BINS = 100
DEFAULT_MAX_VOXELS_TO_PLOT = 60_000
DEFAULT_RANDOM_SEED = 42


# ICRS (RA/DEC) to Galactic Cartesian rotation matrix.
# This keeps the transformation explicit and avoids an extra dependency.
ICRS_TO_GALACTIC = np.array(
    [
        [-0.0548755604, -0.8734370902, -0.4838350155],
        [0.4941094279, -0.4448296300, 0.7469822445],
        [-0.8676661490, -0.1980763734, 0.4559837762],
    ],
    dtype=float,
)


def _log(message: str) -> None:
    print(message, flush=True)


def _elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.2f} s"


def query_gaia(
    top: int = DEFAULT_TOP,
    random_index_min: float | None = None,
    random_index_max: float | None = None,
):
    """Query Gaia DR3 with a random sample.

    The optional random_index bounds are intentionally exposed so the sample
    can later be split into sectors without rewriting the query logic.
    """

    where_clauses = [
        "parallax IS NOT NULL",
        "parallax > 0",
        "ra IS NOT NULL",
        "dec IS NOT NULL",
    ]

    if random_index_min is not None:
        where_clauses.append(f"random_index >= {float(random_index_min)}")
    if random_index_max is not None:
        where_clauses.append(f"random_index < {float(random_index_max)}")

    query = f"""
    SELECT TOP {int(top)}
        ra,
        dec,
        parallax,
        phot_g_mean_mag
    FROM gaiadr3.gaia_source
    WHERE
        {" AND ".join(where_clauses)}
    ORDER BY random_index
    """

    _log("Querying Gaia DR3...")
    _log(f"Top rows requested: {int(top)}")
    if random_index_min is not None or random_index_max is not None:
        _log(
            "Random index sector: "
            f"[{random_index_min if random_index_min is not None else 0.0}, "
            f"{random_index_max if random_index_max is not None else 1.0})"
        )

    job = Gaia.launch_job_async(query=query, verbose=False)
    results = job.get_results()
    _log(f"Rows fetched from Gaia: {len(results)}")
    return results


def _icrs_to_galactic_xyz(x_helio: np.ndarray, y_helio: np.ndarray, z_helio: np.ndarray):
    stacked = np.vstack([x_helio, y_helio, z_helio])
    gal = ICRS_TO_GALACTIC @ stacked
    return gal[0], gal[1], gal[2]


def convert_coordinates(table):
    """Convert RA/DEC/parallax to galactocentric XYZ in kpc."""

    ra = np.asarray(table["ra"], dtype=float)
    dec = np.asarray(table["dec"], dtype=float)
    parallax = np.asarray(table["parallax"], dtype=float)

    if "phot_g_mean_mag" in table.colnames:
        g_mag = np.asarray(table["phot_g_mean_mag"], dtype=float)
    else:
        g_mag = np.full(len(ra), np.nan, dtype=float)

    before = len(ra)
    mask = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(parallax) & (parallax > 0)
    ra = ra[mask]
    dec = dec[mask]
    parallax = parallax[mask]
    g_mag = g_mag[mask]
    after = len(ra)

    if after == 0:
        raise ValueError("No valid Gaia rows remain after filtering.")

    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)

    # Parallax is in milliarcseconds, so the distance in kpc is simply 1/parallax.
    distance_kpc = 1.0 / parallax

    x_helio = distance_kpc * np.cos(dec_rad) * np.cos(ra_rad)
    y_helio = distance_kpc * np.cos(dec_rad) * np.sin(ra_rad)
    z_helio = distance_kpc * np.sin(dec_rad)

    x_gal, y_gal, z_gal = _icrs_to_galactic_xyz(x_helio, y_helio, z_helio)

    # Shift into a galactocentric frame with the Galactic center at (0, 0, 0)
    # and the Sun at X = -8.2 kpc.
    x_gc = x_gal - 8.2
    y_gc = y_gal
    z_gc = z_gal

    _log(f"Rows after coordinate filtering: {after} / {before}")
    _log(
        "Distance range: "
        f"{distance_kpc.min():.3f} to {distance_kpc.max():.3f} kpc"
    )

    return {
        "x": x_gc,
        "y": y_gc,
        "z": z_gc,
        "distance_kpc": distance_kpc,
        "g_mag": g_mag,
        "count": after,
    }


def create_density_voxels(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bins: int | tuple[int, int, int] = DEFAULT_BINS,
    max_voxels_to_plot: int = DEFAULT_MAX_VOXELS_TO_PLOT,
    random_seed: int = DEFAULT_RANDOM_SEED,
):
    """Bin stars into voxels and return voxel centers with density."""

    coords = np.column_stack([x, y, z])
    counts, edges = np.histogramdd(coords, bins=bins)

    nonzero_mask = counts > 0
    nonzero_indices = np.column_stack(np.nonzero(nonzero_mask))
    densities = counts[nonzero_mask].astype(float)
    nonzero_voxel_count = int(densities.size)

    if nonzero_voxel_count == 0:
        raise ValueError("No non-empty voxels were created.")

    selected = np.arange(nonzero_voxel_count)
    if nonzero_voxel_count > max_voxels_to_plot:
        log_density = np.log1p(densities)
        dense_cutoff = np.percentile(log_density, 85.0)
        dense_idx = np.flatnonzero(log_density >= dense_cutoff)

        if dense_idx.size >= max_voxels_to_plot:
            dense_densities = densities[dense_idx]
            keep_local = np.argsort(dense_densities)[-max_voxels_to_plot:]
            selected = dense_idx[keep_local]
        else:
            remaining_slots = max_voxels_to_plot - dense_idx.size
            sparse_idx = np.setdiff1d(
                np.arange(nonzero_voxel_count), dense_idx, assume_unique=False
            )
            rng = np.random.default_rng(random_seed)
            if sparse_idx.size > 0 and remaining_slots > 0:
                sparse_densities = densities[sparse_idx]
                weights = sparse_densities / sparse_densities.sum()
                sampled = rng.choice(
                    sparse_idx,
                    size=min(remaining_slots, sparse_idx.size),
                    replace=False,
                    p=weights,
                )
                selected = np.concatenate([dense_idx, sampled])
            else:
                selected = dense_idx

    selected = np.asarray(selected, dtype=int)
    selected_indices = nonzero_indices[selected]
    selected_densities = densities[selected]

    x_edges, y_edges, z_edges = edges
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    voxel_x = x_centers[selected_indices[:, 0]]
    voxel_y = y_centers[selected_indices[:, 1]]
    voxel_z = z_centers[selected_indices[:, 2]]

    log_density = np.log1p(selected_densities)

    _log(f"Non-empty voxels: {nonzero_voxel_count}")
    _log(f"Voxels plotted: {len(selected_densities)}")
    _log(f"Voxel grid shape: {counts.shape[0]} x {counts.shape[1]} x {counts.shape[2]}")

    return {
        "x": voxel_x,
        "y": voxel_y,
        "z": voxel_z,
        "density": selected_densities,
        "log_density": log_density,
        "counts": counts,
        "edges": edges,
        "nonzero_voxels": nonzero_voxel_count,
        "plotted_voxels": int(len(selected_densities)),
    }


def load_microlensing_events():
    """Placeholder for future microlensing overlay.

    For now this returns an empty list, but it is intentionally separate so
    future event tables can be converted to XYZ and drawn as a red Scatter3D
    layer without changing the Milky Way workflow.
    """

    return []


def _prepare_microlensing_xyz(events: Iterable[dict]):
    xs = []
    ys = []
    zs = []

    for event in events:
        if all(key in event for key in ("x", "y", "z")):
            xs.append(float(event["x"]))
            ys.append(float(event["y"]))
            zs.append(float(event["z"]))
            continue

        if all(key in event for key in ("ra", "dec", "distance")):
            ra_rad = math.radians(float(event["ra"]))
            dec_rad = math.radians(float(event["dec"]))
            distance_kpc = float(event["distance"])

            x_helio = distance_kpc * math.cos(dec_rad) * math.cos(ra_rad)
            y_helio = distance_kpc * math.cos(dec_rad) * math.sin(ra_rad)
            z_helio = distance_kpc * math.sin(dec_rad)

            x_gal, y_gal, z_gal = _icrs_to_galactic_xyz(
                np.array([x_helio]),
                np.array([y_helio]),
                np.array([z_helio]),
            )
            xs.append(float(x_gal[0] - 8.2))
            ys.append(float(y_gal[0]))
            zs.append(float(z_gal[0]))

    return xs, ys, zs


def create_milky_way_plot(
    density_data: dict,
    output_html: str | Path = "milky_way_3d.html",
    microlensing_events: Iterable[dict] | None = None,
):
    """Create and save the interactive Plotly visualization."""

    x = np.asarray(density_data["x"], dtype=float)
    y = np.asarray(density_data["y"], dtype=float)
    z = np.asarray(density_data["z"], dtype=float)
    density = np.asarray(density_data["density"], dtype=float)
    log_density = np.asarray(density_data["log_density"], dtype=float)

    if log_density.size == 0:
        raise ValueError("No density voxels available for plotting.")

    log_min = float(log_density.min())
    log_max = float(log_density.max())
    log_span = log_max - log_min if log_max > log_min else 1.0
    norm = (log_density - log_min) / log_span
    point_sizes = 1.2 + 3.3 * norm

    density_trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        name="Gaia density",
        marker=dict(
            size=point_sizes,
            color=log_density,
            colorscale="Inferno",
            opacity=0.78,
            cmin=log_min,
            cmax=log_max,
            colorbar=dict(
                title="log10(N+1)",
                thickness=15,
                len=0.75,
            ),
        ),
        hovertemplate=(
            "X: %{x:.2f} kpc<br>"
            "Y: %{y:.2f} kpc<br>"
            "Z: %{z:.2f} kpc<br>"
            "Voxel stars: %{customdata:.0f}<extra></extra>"
        ),
        customdata=density,
    )

    all_x = [x]
    all_y = [y]
    all_z = [z]
    traces = [density_trace]

    center_trace = go.Scatter3d(
        x=[0.0],
        y=[0.0],
        z=[0.0],
        mode="markers+text",
        name="Galactic center",
        text=["GC"],
        textposition="top center",
        marker=dict(size=7, color="red"),
        hovertemplate="Galactic center<extra></extra>",
    )
    sun_trace = go.Scatter3d(
        x=[SUN_GALACTOCENTRIC_X_KPC],
        y=[0.0],
        z=[0.0],
        mode="markers+text",
        name="Sun",
        text=["Sun"],
        textposition="top center",
        marker=dict(size=6, color="yellow"),
        hovertemplate="Sun<extra></extra>",
    )
    traces.extend([center_trace, sun_trace])
    all_x.extend([np.array([0.0]), np.array([SUN_GALACTOCENTRIC_X_KPC])])
    all_y.extend([np.array([0.0]), np.array([0.0])])
    all_z.extend([np.array([0.0]), np.array([0.0])])

    if microlensing_events is None:
        microlensing_events = load_microlensing_events()

    microlensing_events = list(microlensing_events)
    if microlensing_events:
        mx, my, mz = _prepare_microlensing_xyz(microlensing_events)
        microlensing_trace = go.Scatter3d(
            x=mx,
            y=my,
            z=mz,
            mode="markers",
            name="Microlensing",
            marker=dict(size=5, color="red", opacity=0.9),
            hovertemplate=(
                "Microlensing event<br>"
                "X: %{x:.2f} kpc<br>"
                "Y: %{y:.2f} kpc<br>"
                "Z: %{z:.2f} kpc<extra></extra>"
            ),
        )
        traces.append(microlensing_trace)
        all_x.append(np.asarray(mx, dtype=float))
        all_y.append(np.asarray(my, dtype=float))
        all_z.append(np.asarray(mz, dtype=float))

    finite_x = np.concatenate(all_x)
    finite_y = np.concatenate(all_y)
    finite_z = np.concatenate(all_z)

    axis_extent = float(
        max(
            np.max(np.abs(finite_x)),
            np.max(np.abs(finite_y)),
            np.max(np.abs(finite_z)),
            10.0,
        )
    )
    axis_extent *= 1.05

    axis_traces = [
        go.Scatter3d(
            x=[-axis_extent, axis_extent],
            y=[0.0, 0.0],
            z=[0.0, 0.0],
            mode="lines",
            name="X axis",
            line=dict(color="cyan", width=4),
            hoverinfo="skip",
            showlegend=False,
        ),
        go.Scatter3d(
            x=[0.0, 0.0],
            y=[-axis_extent, axis_extent],
            z=[0.0, 0.0],
            mode="lines",
            name="Y axis",
            line=dict(color="lime", width=4),
            hoverinfo="skip",
            showlegend=False,
        ),
        go.Scatter3d(
            x=[0.0, 0.0],
            y=[0.0, 0.0],
            z=[-axis_extent, axis_extent],
            mode="lines",
            name="Z axis",
            line=dict(color="deepskyblue", width=4),
            hoverinfo="skip",
            showlegend=False,
        ),
    ]

    traces.extend(axis_traces)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Gaia DR3 Milky Way 3D density map",
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        legend=dict(font=dict(color="white")),
        margin=dict(l=0, r=0, t=45, b=0),
        scene=dict(
            aspectmode="data",
            xaxis=dict(
                title="X [kpc]",
                range=[-axis_extent, axis_extent],
                showbackground=True,
                backgroundcolor="rgb(0,0,0)",
                gridcolor="rgba(255,255,255,0.08)",
                zerolinecolor="rgba(255,255,255,0.20)",
                color="white",
            ),
            yaxis=dict(
                title="Y [kpc]",
                range=[-axis_extent, axis_extent],
                showbackground=True,
                backgroundcolor="rgb(0,0,0)",
                gridcolor="rgba(255,255,255,0.08)",
                zerolinecolor="rgba(255,255,255,0.20)",
                color="white",
            ),
            zaxis=dict(
                title="Z [kpc]",
                range=[-axis_extent, axis_extent],
                showbackground=True,
                backgroundcolor="rgb(0,0,0)",
                gridcolor="rgba(255,255,255,0.08)",
                zerolinecolor="rgba(255,255,255,0.20)",
                color="white",
            ),
            camera=dict(
                eye=dict(x=1.45, y=1.45, z=0.95),
                center=dict(x=0, y=0, z=0),
            ),
        ),
    )

    output_html = Path(output_html)
    fig.write_html(
        str(output_html),
        include_plotlyjs=True,
        full_html=True,
        auto_open=False,
        config=dict(scrollZoom=True, isplaylogo=False, responsive=True),
    )
    _log(f"Saved interactive HTML: {output_html.resolve()}")
    return fig


def main():
    start_total = time.perf_counter()

    query_start = time.perf_counter()
    table = query_gaia(top=DEFAULT_TOP)
    _log(f"Gaia query time: {_elapsed(query_start)}")

    convert_start = time.perf_counter()
    coords = convert_coordinates(table)
    _log(f"Coordinate conversion time: {_elapsed(convert_start)}")

    voxel_start = time.perf_counter()
    density_voxels = create_density_voxels(
        coords["x"],
        coords["y"],
        coords["z"],
        bins=DEFAULT_BINS,
        max_voxels_to_plot=DEFAULT_MAX_VOXELS_TO_PLOT,
    )
    _log(f"Voxelization time: {_elapsed(voxel_start)}")

    plot_start = time.perf_counter()
    create_milky_way_plot(
        density_voxels,
        output_html="milky_way_3d.html",
        microlensing_events=load_microlensing_events(),
    )
    _log(f"Plot export time: {_elapsed(plot_start)}")

    _log(f"Total execution time: {_elapsed(start_total)}")


if __name__ == "__main__":
    main()
