"""
Creates a high-resolution microlensing animation based on Gaia DR3 data.

The program loads Gaia microlensing events, overlays their brightness changes
on a precomputed Milky Way all-sky map, renders animation frames, and encodes
the final video using FFmpeg.
"""
import os

# Limit numerical library threads to keep CPU usage predictable.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import multiprocessing as mp

MAX_RENDER_WORKERS = 4

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from pathlib import Path
import subprocess
from tqdm import tqdm


# -----------------------------
# Program settings
# -----------------------------

MICROLENS_FILE = "gaia_microlensing.fits"
MAP_FILE       = "gaia_allsky_hammer_16k.png"
OUT_DIR        = Path("frames_micro")
VIDEO          = "microlensing_animation.mp4"
PNG_CACHE      = Path("microlensing_events.png")
FRAMES         = 625 # target 25sec -> 625 
FPS            = 25


# -----------------------------
# Microlensing model
# -----------------------------

def amplification(t, t0, te, u0):
    u = np.sqrt(u0*u0 + ((t - t0) / te)**2)
    return (u*u + 2) / (u * np.sqrt(u*u + 4))


# -----------------------------
# Coordinate conversion
# -----------------------------

def galactic_to_pixel(l, b, img_w, img_h):
    """Converts Galactic coordinates to pixel positions on the Hammer projection map."""
    z     = np.sqrt(1.0 + np.cos(b) * np.cos(l * 0.5))
    x_ham = 2.0 * np.sqrt(2.0) * np.cos(b) * np.sin(l * 0.5) / z
    y_ham = np.sqrt(2.0) * np.sin(b) / z

    px = (x_ham / (2.0 * np.sqrt(2.0)) + 1.0) * 0.5 * img_w
    py = (1.0 - y_ham / np.sqrt(2.0))           * 0.5 * img_h
    return px, py


# -----------------------------
# Light amplification rendering
# -----------------------------

def draw_gaussians(canvas, px_arr, py_arr, sigma_arr, alpha_arr, color):
    """Draws smooth Gaussian brightness increases for microlensing events."""
    img_h, img_w = canvas.shape[:2]

    for cx, cy, sig, alph in zip(px_arr, py_arr, sigma_arr, alpha_arr):
        sig  = max(float(sig), 1.0)
        cut  = int(3.5 * sig) + 1

        x0 = max(0,     int(cx) - cut)
        x1 = min(img_w, int(cx) + cut + 1)
        y0 = max(0,     int(cy) - cut)
        y1 = min(img_h, int(cy) + cut + 1)

        if x0 >= x1 or y0 >= y1:
            continue

        # Use broadcasting to reduce memory usage during Gaussian calculations.
        yy, xx = np.ogrid[y0:y1, x0:x1]
        r2    = (xx - cx)**2 + (yy - cy)**2
        gauss = np.exp(-r2 / (2.0 * sig * sig))

        # Increase the visible bright area near the center of the microlensing event.
        flat_level = 0.5
        gauss = np.where(
            gauss > flat_level,
            1.0,
            gauss / flat_level
        )

        gauss *= float(alph)

        canvas[y0:y1, x0:x1] += gauss[:, :, np.newaxis] * color


# -----------------------------
# Image data conversion helpers
# -----------------------------

def _bg_to_float(bg):
    """Converts image data to float RGB format used during rendering."""
    rgb = bg[:, :, :3]
    if rgb.dtype == np.uint8:
        return rgb.astype(np.float32) / 255.0
    return rgb.astype(np.float32)


# -----------------------------
# Parallel rendering setup
# (loads background once per process instead of sending ~400 MB via pickle)
# -----------------------------

_bg_global = None

def _init_worker(bg_path):
    global _bg_global
    _bg_global = plt.imread(bg_path)


# -----------------------------
# Animation frame generation
# -----------------------------

def render_frame(task):
    frame, l, b, tmax, te, u0, mass = task

    bg     = _bg_global
    img_h, img_w = bg.shape[:2]

    # Calculate the simulation time represented by this frame.
    t = np.linspace(
        np.min(tmax)  - 3.0 * np.percentile(te, 95),
        np.max(tmax)  + 3.0 * np.percentile(te, 95),
        FRAMES
    )[frame]

    # Convert event positions to image pixel coordinates.
    px_all, py_all = galactic_to_pixel(l, b, img_w, img_h)

    # Calculate brightness amplification using the Paczynski model.
    amp     = amplification(t, tmax, te, u0)
    visible = np.abs(t - tmax) < te * 2.5

    # Combine the Milky Way background with microlensing flashes.
    bg_f = _bg_to_float(bg)
    glow = np.zeros((img_h, img_w, 3), dtype=np.float32)

    if np.any(visible):
        strength = np.clip(amp[visible] - 1.0, 0.0, None)
        mass_v   = np.clip(mass[visible], 0.3, 5.0)

        # --- Gaussian sigma in pixels ---
        # Equivalent: scatter size = 10 + 10000*log1p(strength) [pt²]
        # Conversion: σ_px = √(size/π) · (dpi/72) / 3 · mass · scale
        # dpi=200 as in original; scale normalizes to map resolution
        scale    = img_h / 4320.0
        size_pt2 = 10.0 + 10000.0 * np.log1p(strength)
        sigma_px = (
            np.sqrt(size_pt2 / np.pi)
            * (200.0 / 72.0) / 3.0
            * mass_v
            * scale
        )
        sigma_px = np.clip(sigma_px, 2.0, img_h * 0.08)

        # Intensity (as in original)
        alpha = np.clip(np.log1p(strength) / np.log(10.0), 0.05, 1.0)

        # Color: warm white (visible on dark Gaia map background)
        color = np.array([1.0, 0.95, 0.80], dtype=np.float32)

        draw_gaussians(
            glow,
            px_all[visible], py_all[visible],
            sigma_px, alpha,
            color
        )

    composite = np.clip(bg_f + glow, 0.0, 1.0)

    # plt.imsave instead of fig+savefig — zero matplotlib overhead
    plt.imsave(str(OUT_DIR / f"frame_{frame:04d}.png"), composite)


# -----------------------------
# Fallback background map generation
# -----------------------------

def create_background_map():
    print("Creating substitute Hammer map...")
    fig = plt.figure(figsize=(76.8, 43.2), dpi=100, facecolor="black")
    ax  = fig.add_subplot(111, projection="hammer")
    ax.set_facecolor("black")
    ax.axis("off")
    lon = np.linspace(-np.pi, np.pi, 400)
    ax.plot(lon, np.zeros_like(lon), color="gray", linewidth=0.5)
    fig.savefig(MAP_FILE, facecolor="black", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print("Saved map:", MAP_FILE)


# -----------------------------
# Load microlensing data
# -----------------------------

def load_events():
    print("Loading microlensing catalog...")
    tab = Table.read(MICROLENS_FILE).to_pandas()
    tab = tab.dropna(subset=["l", "b"])

    l = np.radians(tab["l"].values)
    b = np.radians(tab["b"].values)

    # Range [-π, π], negation → east to the left (galactic standard)
    l = -np.where(l > np.pi, l - 2.0 * np.pi, l)

    tmax = tab["paczynski0_tmax"].values
    te   = tab["paczynski0_te"].values
    u0   = tab["paczynski0_u0"].values

    mass = np.ones(len(tab))
    for col in ["paczynski0_mass", "paczynski_mass", "lens_mass", "mass"]:
        if col in tab.columns:
            mass = tab[col].values
            print("Using mass:", col)
            break

    good = (
        np.isfinite(tmax) & np.isfinite(te)
        & np.isfinite(u0) & (te > 0)
    )
    l, b     = l[good],    b[good]
    tmax, te = tmax[good], te[good]
    u0, mass = u0[good],   mass[good]

    tmax = tmax - np.min(tmax)
    return l, b, tmax, te, u0, mass


# -----------------------------
# Main program
# -----------------------------

def main():
    OUT_DIR.mkdir(exist_ok=True)

    l, b, tmax, te, u0, mass = load_events()
    print("Microlensing events:", len(l))

    if not Path(MAP_FILE).exists():
        raise FileNotFoundError(f"Missing ready map: {MAP_FILE}")
    print("Using sky map:", MAP_FILE)

    bg     = plt.imread(MAP_FILE)
    img_h, img_w = bg.shape[:2]
    print(f"Map resolution: {img_w}×{img_h} px")

    # Create a preview image using the same rendering method as the animation.
    if not PNG_CACHE.exists():
        px_all, py_all = galactic_to_pixel(l, b, img_w, img_h)
        amp_s  = amplification(np.median(tmax), tmax, te, u0)
        vis_s  = np.abs(np.median(tmax) - tmax) < te * 2.5

        bg_f   = _bg_to_float(bg)
        glow_s = np.zeros((img_h, img_w, 3), dtype=np.float32)

        if np.any(vis_s):
            strength_s = np.clip(amp_s[vis_s] - 1.0, 0.0, None)
            scale      = img_h / 4320.0
            size_pt2_s = 10.0 + 10000.0 * np.log1p(strength_s)
            sigma_s    = (
                np.sqrt(size_pt2_s / np.pi)
                * (200.0 / 72.0) / 3.0
                * scale
            )
            sigma_s = np.clip(sigma_s, 2.0, img_h * 0.08)

            draw_gaussians(
                glow_s,
                px_all[vis_s], py_all[vis_s],
                sigma_s,
                np.ones(np.sum(vis_s), dtype=np.float32),
                np.array([1.0, 0.95, 0.80], dtype=np.float32)
            )

        plt.imsave(str(PNG_CACHE), np.clip(bg_f + glow_s, 0.0, 1.0))
        print("Saved PNG:", PNG_CACHE)
    else:
        print("PNG already exists — skipping creation")

    # Render missing animation frames.
    existing_frames = set(OUT_DIR.glob("frame_*.png"))

    if len(existing_frames) >= FRAMES:
        print("Render frames already exist — skipping rendering and proceeding to ffmpeg")
    else:
        print("Missing complete frames — rendering")

    missing_frames = [
        i for i in range(FRAMES)
        if OUT_DIR / f"frame_{i:04d}.png" not in existing_frames
    ]

    # Background map is loaded once by each worker process.
    tasks = [
        (frame, l, b, tmax, te, u0, mass)
        for frame in missing_frames
    ]

    if tasks:
        print(f"Parallel rendering: {MAX_RENDER_WORKERS} processes")
        with mp.Pool(
            processes=MAX_RENDER_WORKERS,
            initializer=_init_worker,
            initargs=(MAP_FILE,)
        ) as pool:
            list(tqdm(
                pool.imap_unordered(render_frame, tasks),
                total=len(tasks),
                desc="Rendering frames",
                unit="frame"
            ))

    # Encode rendered frames into video files.
    print("Encoding mp4")

    if Path(VIDEO).exists():
        print("MP4 already exists — skipping assembly")
        return

    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", str(OUT_DIR / "frame_%04d.png"),
        "-vf", "scale=15998:9000",
        "-c:v", "libx265",
        "-pix_fmt", "yuv420p",
        VIDEO
    ])
    subprocess.run([
        "ffmpeg", "-y",
        "-i", VIDEO,
        "-c:v", "copy",
        "-tag:v", "hvc1",
        "microlensing_animation_git.mp4"
    ])

    convert_phone = input("Create 8K phone-compatible version? (y/n): ").lower().strip()

    if convert_phone == "y":
        subprocess.run([
            "ffmpeg", "-y",
            "-i", "microlensing_animation_git.mp4",
            "-vf", "scale=7680:4320:flags=lanczos",
            "-c:v", "libx265",
            "-tag:v", "hvc1",
            "-pix_fmt", "yuv420p10le",
            "-crf", "14",
            "-preset", "slow",
            "-movflags", "+faststart",
            "microa_8k_mobile.mp4"
        ])

    if not Path(VIDEO).exists():
        print("MP4 encoding error")
        return
    print("Done:", VIDEO)


if __name__ == "__main__":
    main()