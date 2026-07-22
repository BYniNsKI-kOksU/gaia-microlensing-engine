import os

# limiting multithreading of numerical libraries to 1 core
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
# CONFIGURATION
# -----------------------------

MICROLENS_FILE = "gaia_microlensing.fits"
MAP_FILE       = "gaia_allsky_hammer_16k.png"
OUT_DIR        = Path("frames_micro")
VIDEO          = "microlensing_animation.mp4"
PNG_CACHE      = Path("microlensing_events.png")
FRAMES         = 625 # target 25sec -> 625 
FPS            = 25


# -----------------------------
# PHYSICS: PACZYŃSKI (unchanged)
# -----------------------------

def amplification(t, t0, te, u0):
    u = np.sqrt(u0*u0 + ((t - t0) / te)**2)
    return (u*u + 2) / (u * np.sqrt(u*u + 4))


# -----------------------------
# COORDINATES: GALACTIC → HAMMER PIXELS
# -----------------------------

def galactic_to_pixel(l, b, img_w, img_h):
    """
    Converts galactic coordinates (l, b) [rad] to pixel coordinates (px, py)
    in a PNG map saved in Hammer projection.

    l must be already negated (east to the left, astronomical standard).
    Formulas: x_h = 2√2·cos(b)·sin(l/2) / z
              y_h =  √2·sin(b) / z
              z   = √(1 + cos(b)·cos(l/2))
    x_h ∈ [-2√2, 2√2],  y_h ∈ [-√2, √2]
    """
    z     = np.sqrt(1.0 + np.cos(b) * np.cos(l * 0.5))
    x_ham = 2.0 * np.sqrt(2.0) * np.cos(b) * np.sin(l * 0.5) / z
    y_ham = np.sqrt(2.0) * np.sin(b) / z

    px = (x_ham / (2.0 * np.sqrt(2.0)) + 1.0) * 0.5 * img_w
    py = (1.0 - y_ham / np.sqrt(2.0))           * 0.5 * img_h
    return px, py


# -----------------------------
# RENDERING: GAUSSIAN PROFILE  I(r) = exp(-r²/2σ²)
# -----------------------------

def draw_gaussians(canvas, px_arr, py_arr, sigma_arr, alpha_arr, color):
    """
    Overlays flashes with smooth Gaussian gradient on a float32 RGB canvas.

    One object per event — no rings, no layers.
    I(r) = exp(-r² / 2σ²) * alpha

    At the edge (r ≈ 2.15σ) brightness drops to ~0.1 of maximum.
    Clipping at 3.5σ gives I < 0.002 — invisible, no artifacts.

    canvas   : ndarray (H, W, 3) float32, modified in place
    px_arr   : X positions in pixels
    py_arr   : Y positions in pixels
    sigma_arr: Gaussian sigma in pixels for each event
    alpha_arr: peak intensity [0..1]
    color    : ndarray (3,) float32, RGB color
    """
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

        # e.g. np.ogrid → 1D arrays; broadcasting reduces memory usage
        yy, xx = np.ogrid[y0:y1, x0:x1]
        r2    = (xx - cx)**2 + (yy - cy)**2
        gauss = np.exp(-r2 / (2.0 * sig * sig))

        # Flattening the Gaussian peak:
        # values above flat_level create a larger bright area with intensity 1
        flat_level = 0.5
        gauss = np.where(
            gauss > flat_level,
            1.0,
            gauss / flat_level
        )

        gauss *= float(alph)

        canvas[y0:y1, x0:x1] += gauss[:, :, np.newaxis] * color


# -----------------------------
# HELPER: background as float32 [0,1]
# -----------------------------

def _bg_to_float(bg):
    """Handles both uint8 [0,255] (JPEG) and float32 [0,1] (PNG)."""
    rgb = bg[:, :, :3]
    if rgb.dtype == np.uint8:
        return rgb.astype(np.float32) / 255.0
    return rgb.astype(np.float32)


# -----------------------------
# WORKER PROCESS INITIALIZATION
# (loads background once per process instead of sending ~400 MB via pickle)
# -----------------------------

_bg_global = None

def _init_worker(bg_path):
    global _bg_global
    _bg_global = plt.imread(bg_path)


# -----------------------------
# FRAME RENDERING
# -----------------------------

def render_frame(task):
    frame, l, b, tmax, te, u0, mass = task

    bg     = _bg_global
    img_h, img_w = bg.shape[:2]

    # Time for this frame (identical time axis as original)
    t = np.linspace(
        np.min(tmax)  - 3.0 * np.percentile(te, 95),
        np.max(tmax)  + 3.0 * np.percentile(te, 95),
        FRAMES
    )[frame]

    # Galactic coordinates → PNG pixels (without matplotlib Hammer axes)
    px_all, py_all = galactic_to_pixel(l, b, img_w, img_h)

    # Paczyński physics (unchanged)
    amp     = amplification(t, tmax, te, u0)
    visible = np.abs(t - tmax) < te * 2.5

    # Composite: float32 background + flash layer
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
# CREATING A SUBSTITUTE BACKGROUND MAP (only if file missing)
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
# LOADING DATA (unchanged)
# -----------------------------

def load_events():
    print("Loading FITS...")
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
# MAIN
# -----------------------------

def main():
    OUT_DIR.mkdir(exist_ok=True)

    l, b, tmax, te, u0, mass = load_events()
    print("Events:", len(l))

    if not Path(MAP_FILE).exists():
        raise FileNotFoundError(f"Missing ready map: {MAP_FILE}")
    print("Using ready Milky Way map:", MAP_FILE)

    bg     = plt.imread(MAP_FILE)
    img_h, img_w = bg.shape[:2]
    print(f"Map resolution: {img_w}×{img_h} px")

    # --- Static PNG preview (same logic as animation) ---
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

    # --- Frame rendering ---
    existing_frames = set(OUT_DIR.glob("frame_*.png"))

    if len(existing_frames) >= FRAMES:
        print("Render frames already exist — skipping rendering and proceeding to ffmpeg")
    else:
        print("Missing complete frames — rendering")

    missing_frames = [
        i for i in range(FRAMES)
        if OUT_DIR / f"frame_{i:04d}.png" not in existing_frames
    ]

    # bg is NOT part of the task tuple — loaded once by _init_worker
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

    # --- Video encoding ---
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
            "-i", "microlensing_animation_fixed.mp4",
            "-vf", "scale=7680:4320:flags=lanczos",
            "-c:v", "libx265",
            "-tag:v", "hvc1",
            "-pix_fmt", "yuv420p10le",
            "-crf", "14",
            "-preset", "slow",
            "-movflags", "+faststart",
            "microlensing_a_8k_mobile.mp4"
        ])

    if not Path(VIDEO).exists():
        print("MP4 encoding error")
        return
    print("Done:", VIDEO)


if __name__ == "__main__":
    main()