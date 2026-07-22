import os

# ograniczenie wielowątkowości bibliotek numerycznych do 1 rdzenia
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
# KONFIGURACJA
# -----------------------------

MICROLENS_FILE = "gaia_microlensing.fits"
MAP_FILE       = "gaia_allsky_hammer_16k.png"
OUT_DIR        = Path("frames_micro")
VIDEO          = "microlensing_animation.mp4"
PNG_CACHE      = Path("microlensing_events.png")
FRAMES         = 100
FPS            = 25


# -----------------------------
# FIZYKA: PACZYŃSKI (bez zmian)
# -----------------------------

def amplification(t, t0, te, u0):
    u = np.sqrt(u0*u0 + ((t - t0) / te)**2)
    return (u*u + 2) / (u * np.sqrt(u*u + 4))


# -----------------------------
# WSPÓŁRZĘDNE: GALAKTYCZNE → PIKSELE HAMMERA
# -----------------------------

def galactic_to_pixel(l, b, img_w, img_h):
    """
    Przelicza współrzędne galaktyczne (l, b) [rad] na pikselowe (px, py)
    w mapie PNG zapisanej w projekcji Hammera.

    l musi być już zanegowane (wschód na lewo, standard astronomiczny).
    Wzory: x_h = 2√2·cos(b)·sin(l/2) / z
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
# RENDEROWANIE: PROFIL GAUSSA  I(r) = exp(-r²/2σ²)
# -----------------------------

def draw_gaussians(canvas, px_arr, py_arr, sigma_arr, alpha_arr, color):
    """
    Nakłada błyski z płynnym gradientem Gaussa na canvas float32 RGB.

    Jeden obiekt na zdarzenie — bez pierścieni, bez warstw.
    I(r) = exp(-r² / 2σ²) * alpha

    Na brzegu (r ≈ 2.15σ) jasność spada do ~0.1 maksimum.
    Obcięcie przy 3.5σ daje I < 0.002 — niewidoczne, brak artefaktów.

    canvas   : ndarray (H, W, 3) float32, modyfikowane w miejscu
    px_arr   : pozycje X w pikselach
    py_arr   : pozycje Y w pikselach
    sigma_arr: sigma Gaussa w pikselach dla każdego zdarzenia
    alpha_arr: szczytowa intensywność [0..1]
    color    : ndarray (3,) float32, kolor RGB
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

        # np.ogrid → tablice 1D; broadcasting redukuje zużycie pamięci
        yy, xx = np.ogrid[y0:y1, x0:x1]
        r2    = (xx - cx)**2 + (yy - cy)**2
        gauss = np.exp(-r2 / (2.0 * sig * sig)) * float(alph)

        canvas[y0:y1, x0:x1] += gauss[:, :, np.newaxis] * color


# -----------------------------
# POMOCNICZA: tło jako float32 [0,1]
# -----------------------------

def _bg_to_float(bg):
    """Obsługuje zarówno uint8 [0,255] (JPEG) jak i float32 [0,1] (PNG)."""
    rgb = bg[:, :, :3]
    if rgb.dtype == np.uint8:
        return rgb.astype(np.float32) / 255.0
    return rgb.astype(np.float32)


# -----------------------------
# INICJALIZACJA PROCESU ROBOCZEGO
# (wczytuje tło raz na proces zamiast przesyłać ~400 MB przez pickle)
# -----------------------------

_bg_global = None

def _init_worker(bg_path):
    global _bg_global
    _bg_global = plt.imread(bg_path)


# -----------------------------
# RENDER KLATKI
# -----------------------------

def render_frame(task):
    frame, l, b, tmax, te, u0, mass = task

    bg     = _bg_global
    img_h, img_w = bg.shape[:2]

    # Czas dla tej klatki (identyczna oś czasu jak w oryginale)
    t = np.linspace(
        np.min(tmax)  - 3.0 * np.percentile(te, 95),
        np.max(tmax)  + 3.0 * np.percentile(te, 95),
        FRAMES
    )[frame]

    # Współrzędne galaktyczne → piksele PNG (bez osi matplotlib Hammer)
    px_all, py_all = galactic_to_pixel(l, b, img_w, img_h)

    # Fizyka Paczyńskiego (bez zmian)
    amp     = amplification(t, tmax, te, u0)
    visible = np.abs(t - tmax) < te * 2.5

    # Kompozyt: tło float32 + warstwa błysków
    bg_f = _bg_to_float(bg)
    glow = np.zeros((img_h, img_w, 3), dtype=np.float32)

    if np.any(visible):
        strength = np.clip(amp[visible] - 1.0, 0.0, None)
        mass_v   = np.clip(mass[visible], 0.3, 5.0)

        # --- Sigma Gaussa w pikselach ---
        # Ekwiwalent: scatter size = 10 + 10000*log1p(strength) [pkt²]
        # Przeliczenie: σ_px = √(size/π) · (dpi/72) / 3 · mass · scale
        # dpi=200 jak w oryginale; scale normalizuje do rozdzielczości mapy
        scale    = img_h / 4320.0
        size_pt2 = 10.0 + 10000.0 * np.log1p(strength)
        sigma_px = (
            np.sqrt(size_pt2 / np.pi)
            * (200.0 / 72.0) / 3.0
            * mass_v
            * scale
        )
        sigma_px = np.clip(sigma_px, 2.0, img_h * 0.08)

        # Intensywność (jak w oryginale)
        alpha = np.clip(np.log1p(strength) / np.log(10.0), 0.05, 1.0)

        # Kolor: ciepła biel (widoczna na ciemnym tle mapy Gaia)
        color = np.array([1.0, 0.95, 0.80], dtype=np.float32)

        draw_gaussians(
            glow,
            px_all[visible], py_all[visible],
            sigma_px, alpha,
            color
        )

    composite = np.clip(bg_f + glow, 0.0, 1.0)

    # plt.imsave zamiast fig+savefig — zero narzutu matplotlib
    plt.imsave(str(OUT_DIR / f"frame_{frame:04d}.png"), composite)


# -----------------------------
# TWORZENIE ZASTĘPCZEJ MAPY TŁA (tylko jeśli brak pliku)
# -----------------------------

def create_background_map():
    print("Tworzenie zastępczej mapy Hammer...")
    fig = plt.figure(figsize=(76.8, 43.2), dpi=100, facecolor="black")
    ax  = fig.add_subplot(111, projection="hammer")
    ax.set_facecolor("black")
    ax.axis("off")
    lon = np.linspace(-np.pi, np.pi, 400)
    ax.plot(lon, np.zeros_like(lon), color="gray", linewidth=0.5)
    fig.savefig(MAP_FILE, facecolor="black", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print("Zapisano mapę:", MAP_FILE)


# -----------------------------
# WCZYTYWANIE DANYCH (bez zmian)
# -----------------------------

def load_events():
    print("Wczytywanie FITS...")
    tab = Table.read(MICROLENS_FILE).to_pandas()
    tab = tab.dropna(subset=["l", "b"])

    l = np.radians(tab["l"].values)
    b = np.radians(tab["b"].values)

    # Zakres [-π, π], negacja → wschód na lewo (standard galaktyczny)
    l = -np.where(l > np.pi, l - 2.0 * np.pi, l)

    tmax = tab["paczynski0_tmax"].values
    te   = tab["paczynski0_te"].values
    u0   = tab["paczynski0_u0"].values

    mass = np.ones(len(tab))
    for col in ["paczynski0_mass", "paczynski_mass", "lens_mass", "mass"]:
        if col in tab.columns:
            mass = tab[col].values
            print("Używam masy:", col)
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
    print("Zdarzenia:", len(l))

    if not Path(MAP_FILE).exists():
        raise FileNotFoundError(f"Brak gotowej mapy: {MAP_FILE}")
    print("Używam gotowej mapy Drogi Mlecznej:", MAP_FILE)

    bg     = plt.imread(MAP_FILE)
    img_h, img_w = bg.shape[:2]
    print(f"Rozdzielczość mapy: {img_w}×{img_h} px")

    # --- Statyczny PNG podglądu (ta sama logika co animacja) ---
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
        print("Zapisano PNG:", PNG_CACHE)
    else:
        print("PNG już istnieje — pomijam tworzenie")

    # --- Rendering klatek ---
    existing_frames = set(OUT_DIR.glob("frame_*.png"))

    if len(existing_frames) >= FRAMES:
        print("Klatki renderu już istnieją — pomijam rendering i przechodzę do ffmpeg")
    else:
        print("Brak kompletu klatek — renderowanie")

    missing_frames = [
        i for i in range(FRAMES)
        if OUT_DIR / f"frame_{i:04d}.png" not in existing_frames
    ]

    # bg NIE jest częścią krotki task — ładowane raz przez _init_worker
    tasks = [
        (frame, l, b, tmax, te, u0, mass)
        for frame in missing_frames
    ]

    if tasks:
        print(f"Renderowanie równoległe: {MAX_RENDER_WORKERS} procesy")
        with mp.Pool(
            processes=MAX_RENDER_WORKERS,
            initializer=_init_worker,
            initargs=(MAP_FILE,)
        ) as pool:
            list(tqdm(
                pool.imap_unordered(render_frame, tasks),
                total=len(tasks),
                desc="Renderowanie klatek",
                unit="klatka"
            ))

    # --- Kodowanie wideo ---
    print("Kodowanie mp4")

    if Path(VIDEO).exists():
        print("MP4 już istnieje — pomijam składanie")
        return

    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", str(OUT_DIR / "frame_%04d.png"),
        "-c:v", "libx265",
        "-pix_fmt", "yuv420p",
        VIDEO
    ])

    print("Gotowe:", VIDEO)


if __name__ == "__main__":
    main()