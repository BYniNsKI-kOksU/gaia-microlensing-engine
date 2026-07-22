"""
allsky.py — wersja zoptymalizowana pod kątem szybkości renderingu.

2. RÓWNOLEGŁY RENDERING KLATEK (multiprocessing)
   Renderowanie klatki (Agg → PNG) jest zadaniem czysto CPU-bound i
   jednowątkowym w obrębie jednej figury matplotlib. Zamiast renderować
   600 klatek jedna po drugiej w jednym procesie, praca jest dzielona
   na wszystkie rdzenie CPU — każdy proces roboczy buduje własną figurę
   RAZ (tło + puste kolekcje punktów), a potem tylko podmienia
   offsets/sizes/colors i zapisuje PNG. To jest największy realny zysk
   czasowy na wielordzeniowej maszynie.

WAŻNE — inwalidacja cache:
- Jeśli zmienisz plik z danymi mikrosoczewkowania (gaia_microlensing.fits)
  albo parametry FRAMES / ANIMATION_DPI / ANIMATION_FPS, usuń ręcznie
  frame_cache.pkl oraz katalog frames_tmp/ przed ponownym uruchomieniem —
  inaczej skrypt użyje nieaktualnego cache.
- To samo dotyczy mapy: żeby przeliczyć mapę od nowa, usuń
  gaia_allsky_hammer_16k.png i .npz.
"""

import matplotlib
# WYMUSZENIE backendu Agg (czysto rastrowy, bez GUI) — MUSI być wykonane
# przed `import matplotlib.pyplot`. Bez tego matplotlib na macOS potrafi
# automatycznie wybrać backend GUI (np. MacOSX), który przy przeskalowanym
# ekranie Retina/5K mnoży żądane dpi przez współczynnik skalowania ekranu.
matplotlib.use("Agg")

import os
import pickle
import subprocess
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap
from astropy.table import Table
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
#  KONFIGURACJA
# ─────────────────────────────────────────────────────────────
INPUT_FILE = "gaia_90m_allsky.fits"
MAP_FILE = Path("gaia_allsky_hammer_32k.png")
MAP_LAYOUT_FILE = Path("gaia_allsky_hammer_32k_layout.npz")
MICROLENS_FILE = "gaia_microlensing.fits"

FRAME_CACHE_FILE = Path("frame_cache.pkl")
FRAMES_DIR = Path("frames_tmp")
FRAMES_META_FILE = FRAMES_DIR / "_render_meta.pkl"
ANIMATION_OUT = "gaia_microlensing_animation.mp4"

BINS_L = 32768
BINS_B = 16384

FRAMES = 100
ANIMATION_DPI = 200
ANIMATION_FPS = 25
FRAME_CACHE_VERSION = 3

FIG_W_IN, FIG_H_IN = 109.227, 61.44
X264_MAX_DIM = 16384


CPU_COUNT = os.cpu_count() or 1
# Nie wykorzystuj wszystkich rdzeni – zostaw część systemowi.
# Maksymalnie 8 procesów i około 60% dostępnych rdzeni.
N_WORKERS = max(1, min(8, int(CPU_COUNT * 0.6)))


# ─────────────────────────────────────────────────────────────
#  PALETA — budowana raz (moduł jest ładowany raz w każdym procesie)
# ─────────────────────────────────────────────────────────────
_COLORS = [
    (0.000, (0.000, 0.000, 0.000)),   # kosmiczna czerń
    (0.085, (0.008, 0.012, 0.030)),   # bardzo ciemny granat
    (0.180, (0.015, 0.032, 0.060)),   # chłodny halo
    (0.300, (0.035, 0.070, 0.110)),   # niebiesko-szary dysk
    (0.440, (0.090, 0.150, 0.185)),   # jaśniejsze struktury
    (0.575, (0.180, 0.230, 0.220)),   # kremowo-szare gwiazdy i pył
    (0.700, (0.360, 0.330, 0.255)),   # strefa bogata w gwiazdy
    (0.820, (0.640, 0.520, 0.290)),   # ciepły złoty dysk
    (0.920, (0.890, 0.760, 0.470)),   # żółto-pomarańczowe centrum
    (1.000, (0.985, 0.945, 0.835)),   # bardzo jasne jądro bez bieli
]


def build_colormap():
    return LinearSegmentedColormap.from_list("milky_way_realistic", _COLORS, N=2048)


def paczynski_amplification(t, t0, te, u0):
    u = np.sqrt(u0 ** 2 + ((t - t0) / te) ** 2)
    return (u * u + 2) / (u * np.sqrt(u * u + 4))


def _smoothstep(edge0, edge1, value):
    if edge1 == edge0:
        return np.clip(value, 0.0, 1.0)
    x = np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _rgba_stack(rgb, alpha):
    rgb = np.asarray(rgb, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    if alpha.size == 0:
        return np.empty((0, 4), dtype=float)
    if alpha.ndim == 0:
        alpha = np.full(1, float(alpha), dtype=float)
    if rgb.ndim == 1:
        rgb_stack = np.repeat(rgb[None, :], alpha.size, axis=0)
    elif rgb.ndim == 2 and rgb.shape[0] == alpha.size:
        rgb_stack = rgb
    elif rgb.ndim == 2 and rgb.shape[0] == 1:
        rgb_stack = np.repeat(rgb, alpha.size, axis=0)
    else:
        raise ValueError("RGB array must be a single color or one color per point.")
    return np.column_stack([rgb_stack, alpha])


# ─────────────────────────────────────────────────────────────
#  MAPA TŁA — liczona i cache'owana do PNG + layout .npz (bez zmian
#  koncepcyjnych względem oryginału, tylko wydzielona do funkcji)
# ─────────────────────────────────────────────────────────────
def build_and_save_sky_map(fig, ax, cmap_mw):
    print(f"Wczytywanie katalogu Gaia: {INPUT_FILE}")
    table = Table.read(INPUT_FILE)
    print(f"Wczytano: {len(table):,} rekordów")

    sky = table.to_pandas().dropna(subset=["l", "b"])
    print(f"Po filtracji:    {len(sky):,} gwiazd")

    l_deg = sky["l"].values
    b_deg = sky["b"].values

    l_deg_centered = np.where(l_deg > 180, l_deg - 360, l_deg)
    l = np.radians(l_deg_centered)
    b = np.radians(b_deg)
    l = -l  # centrum galaktyki (l=0) w środku, l=90 w prawo

    progress = tqdm(total=1, desc="Tworzenie mapy 2D", unit="etap")
    hist, lon_edges, lat_edges = np.histogram2d(
        l, b,
        bins=[BINS_L, BINS_B],
        range=[[-np.pi, np.pi], [-np.pi / 2, np.pi / 2]],
    )

    hist = gaussian_filter(hist.T, sigma=1.8)
    hist = np.log1p(hist)

    p_low = np.percentile(hist, 1.0)
    p_high = np.percentile(hist, 99.85)
    hist = (hist - p_low) / (p_high - p_low)
    hist = np.clip(hist, 0, 1)
    hist = np.arcsinh(3.8 * hist) / np.arcsinh(3.8)
    hist = hist ** 1.12
    hist = np.clip(hist, 0, 1)
    hist = hist.astype(np.float32, copy=False)

    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    LON, LAT = np.meshgrid(lon_centers, lat_centers)

    ax.pcolormesh(LON, LAT, hist, shading="auto", cmap=cmap_mw)
    progress.update(1)
    progress.close()

    ax.set_axis_off()
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    np.savez(
        MAP_LAYOUT_FILE,
        axes_bounds=np.array(ax.get_position().bounds, dtype=np.float32),
        fig_size=np.array(fig.get_size_inches(), dtype=np.float32),
        dpi=np.array([fig.dpi], dtype=np.float32),
    )
    plt.savefig(str(MAP_FILE), dpi=300, facecolor="black")
    print(f"Zapisano mapę: {MAP_FILE}")


def load_cached_background(fig, ax):
    print(f"Użyto gotowej mapy PNG bez ponownego liczenia histogramu: {MAP_FILE}")
    background_img = mpimg.imread(str(MAP_FILE))
    bg_ax = fig.add_axes([0, 0, 1, 1], zorder=-10)
    bg_ax.imshow(background_img, origin="upper", interpolation="nearest")
    bg_ax.axis("off")

    if MAP_LAYOUT_FILE.exists():
        with np.load(MAP_LAYOUT_FILE) as layout:
            ax.set_position(layout["axes_bounds"])
    ax.set_zorder(10)
    ax.set_facecolor("none")
    ax.patch.set_alpha(0.0)
    ax.set_axis_off()


# ─────────────────────────────────────────────────────────────
#  DANE MIKROSOCZEWKOWANIA
# ─────────────────────────────────────────────────────────────
def prepare_microlensing_data():
    print(f"Wczytywanie mikrosoczewkowania: {MICROLENS_FILE}")
    microlensing = Table.read(MICROLENS_FILE).to_pandas()
    microlensing = microlensing.dropna(subset=["l", "b"])
    print(f"Zdarzenia mikrosoczewkowania: {len(microlensing):,}")

    ml_l = np.radians(microlensing["l"].values)
    ml_b = np.radians(microlensing["b"].values)
    ml_l = -np.where(ml_l > np.pi, ml_l - 2 * np.pi, ml_l)
    print("Zakres l:", ml_l.min(), ml_l.max())
    print("Zakres b:", ml_b.min(), ml_b.max())

    ml_tmax = microlensing["paczynski0_tmax"].values
    ml_te = microlensing["paczynski0_te"].values
    ml_u0 = microlensing["paczynski0_u0"].values
    ml_mass = None
    for mass_col in ("paczynski0_mass", "paczynski_mass", "lens_mass", "mass"):
        if mass_col in microlensing.columns:
            ml_mass = microlensing[mass_col].values
            break

    valid = np.isfinite(ml_tmax) & np.isfinite(ml_te) & np.isfinite(ml_u0) & (ml_te > 0)
    if ml_mass is not None:
        valid &= np.isfinite(ml_mass) & (ml_mass > 0)
    ml_l, ml_b = ml_l[valid], ml_b[valid]
    ml_tmax, ml_te, ml_u0 = ml_tmax[valid], ml_te[valid], ml_u0[valid]
    if ml_mass is not None:
        ml_mass = ml_mass[valid]

    tmax_lo = np.nanpercentile(ml_tmax, 2)
    tmax_hi = np.nanpercentile(ml_tmax, 98)
    core = (ml_tmax >= tmax_lo) & (ml_tmax <= tmax_hi)
    ml_l, ml_b = ml_l[core], ml_b[core]
    ml_tmax, ml_te, ml_u0 = ml_tmax[core], ml_te[core], ml_u0[core]
    if ml_mass is not None:
        ml_mass = ml_mass[core]

    te_lo = np.nanpercentile(ml_te, 5)
    te_hi = np.nanpercentile(ml_te, 95)
    te_log_span = max(np.log10(te_hi) - np.log10(te_lo), 1e-6)
    ml_te_norm = np.clip((np.log10(ml_te) - np.log10(te_lo)) / te_log_span, 0, 1)

    if ml_mass is not None:
        mass_lo = np.nanpercentile(ml_mass, 5)
        mass_hi = np.nanpercentile(ml_mass, 95)
        mass_log_span = max(np.log10(mass_hi) - np.log10(mass_lo), 1e-6)
        ml_mass_norm = np.clip((np.log10(ml_mass) - np.log10(mass_lo)) / mass_log_span, 0, 1)
    else:
        ml_mass_norm = np.zeros_like(ml_te_norm)

    ml_order = np.argsort(ml_tmax)
    ml_activation = np.empty_like(ml_te_norm)
    if len(ml_activation) > 1:
        ml_activation[ml_order] = np.linspace(0.0, 1.0, len(ml_activation), endpoint=True)
    else:
        ml_activation[:] = 1.0

    # Czas animacji jest rozciągany na dokładnie FRAMES klatek.
    # Nie odsłaniamy zdarzeń dodatkową falą; każde zdarzenie pojawia się
    # wyłącznie wtedy, gdy wynika to z jego krzywej Paczyńskiego.
    event_window = 2.75 * ml_te
    anim_start = float(np.nanmin(ml_tmax - event_window))
    anim_end = float(np.nanmax(ml_tmax + event_window))

    return dict(
        l=ml_l, b=ml_b, tmax=ml_tmax, te=ml_te, u0=ml_u0,
        te_norm=ml_te_norm, mass_norm=ml_mass_norm, activation=ml_activation,
        anim_start=anim_start, anim_end=anim_end,
    )


# ─────────────────────────────────────────────────────────────
#  CACHE: dane WSZYSTKICH klatek liczone RAZ, wektorowo
# ─────────────────────────────────────────────────────────────
def precompute_all_frames(m, frames):
    """
    Liczy dla wszystkich klatek naraz (macierz frames × zdarzenia):
    widoczność, amplifikację Paczyńskiego, rozmiary, kolory i alfy.
    Zwraca current_times oraz listę długości `frames`, gdzie każdy
    element to None (nic widocznego) albo słownik gotowych tablic
    numpy do bezpośredniego wrzucenia w set_offsets/set_sizes/
    set_facecolors — bez żadnych obliczeń w pętli renderującej.
    """
    current_times = np.linspace(m["anim_start"], m["anim_end"], frames)

    T = current_times[:, None]                 # (F, 1)
    TMAX = m["tmax"][None, :]                   # (1, N)
    TE = m["te"][None, :]
    U0 = m["u0"][None, :]
    EVENT_WINDOW = (2.75 * m["te"])[None, :]

    delta = T - TMAX
    visible = np.abs(delta) <= EVENT_WINDOW
    print("Widoczne zdarzenia:")
    print(np.sum(visible, axis=1))

    u = np.sqrt(U0 ** 2 + (delta / TE) ** 2)
    amp = (u * u + 2) / (u * np.sqrt(u * u + 4))

    contrast = np.clip(amp - 1.0, 0.0, None)
    amp_boost = np.log1p(contrast * 9.0) / np.log1p(9.0)

    mass_factor = np.clip(0.85 + 3.4 * m["mass_norm"][None, :], 0.85, 4.25)
    brightness_factor = np.clip(0.55 + 1.15 * amp_boost, 0.55, 1.70)
    core_size = 18.0 * mass_factor * brightness_factor
    inner_size = core_size * 7.0
    outer_size = core_size * 22.0

    weak_rgb = np.array([0.42, 0.91, 1.00])
    strong_rgb = np.array([0.97, 0.99, 1.00])
    blend = np.clip(0.18 + 0.82 * amp_boost, 0.0, 1.0)
    rgb = weak_rgb[None, None, :] * (1 - blend[..., None]) + strong_rgb[None, None, :] * blend[..., None]

    fade = np.clip(1.0 - np.abs(delta) / EVENT_WINDOW, 0.0, 1.0)
    fade = fade * fade * (3.0 - 2.0 * fade)
    pulse = fade * np.clip(0.35 + 0.65 * amp_boost, 0.0, 1.0)

    outer_alpha = np.clip(0.04 + 0.24 * pulse, 0.0, 0.32)
    inner_alpha = np.clip(0.10 + 0.50 * pulse, 0.0, 0.62)
    core_alpha = np.clip(0.38 + 0.62 * pulse, 0.0, 1.00)

    frame_cache = []
    for f in tqdm(range(frames), desc="Liczenie danych klatek", unit="klatka"):
        idx = np.nonzero(visible[f])[0]
        if idx.size == 0:
            frame_cache.append(None)
            continue
        offsets = np.column_stack((m["l"][idx], m["b"][idx])).astype(np.float32)
        frame_cache.append({
            "offsets": offsets,
            "outer_size": outer_size[f, idx].astype(np.float32),
            "outer_rgba": _rgba_stack([0.30, 0.88, 1.00], outer_alpha[f, idx]).astype(np.float32),
            "inner_size": inner_size[f, idx].astype(np.float32),
            "inner_rgba": _rgba_stack([0.70, 0.96, 1.00], inner_alpha[f, idx]).astype(np.float32),
            "core_size": core_size[f, idx].astype(np.float32),
            "core_rgba": _rgba_stack(rgb[f, idx], core_alpha[f, idx]).astype(np.float32),
        })

    return current_times, frame_cache


# ─────────────────────────────────────────────────────────────
#  RENDEROWANIE RÓWNOLEGŁE (proces roboczy buduje figurę RAZ)
# ─────────────────────────────────────────────────────────────
_W_FIG = _W_OUTER = _W_INNER = _W_CORE = _W_CACHE = None


def _init_worker(map_file, layout_file, cache_file, animation_dpi):
    global _W_FIG, _W_OUTER, _W_INNER, _W_CORE, _W_CACHE

    plt.rcParams.update({"figure.facecolor": "black", "axes.facecolor": "black"})

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=animation_dpi, facecolor="black")
    ax = fig.add_subplot(111, projection="hammer", facecolor="black")

    background_img = mpimg.imread(map_file)
    bg_ax = fig.add_axes([0, 0, 1, 1], zorder=-10)
    bg_ax.imshow(background_img, origin="upper", interpolation="nearest")
    bg_ax.axis("off")

    if os.path.exists(layout_file):
        with np.load(layout_file) as layout:
            ax.set_position(layout["axes_bounds"])
    ax.set_zorder(10)
    ax.set_facecolor("none")
    ax.patch.set_alpha(0.0)
    ax.set_axis_off()

    outer = ax.scatter([], [], s=[], c=[], marker="o", edgecolors="none", zorder=4)
    inner = ax.scatter([], [], s=[], c=[], marker="o", edgecolors="none", zorder=5)
    core = ax.scatter([], [], s=[], c=[], marker="o", edgecolors="none", zorder=6)

    with open(cache_file, "rb") as fh:
        cache = pickle.load(fh)

    _W_FIG, _W_OUTER, _W_INNER, _W_CORE, _W_CACHE = fig, outer, inner, core, cache


def _render_one_frame(task):
    frame_idx, out_path = task
    entry = _W_CACHE[frame_idx]
    empty2 = np.empty((0, 2))
    empty4 = np.empty((0, 4))

    if entry is None:
        _W_OUTER.set_offsets(empty2); _W_OUTER.set_sizes([]); _W_OUTER.set_facecolors(empty4)
        _W_INNER.set_offsets(empty2); _W_INNER.set_sizes([]); _W_INNER.set_facecolors(empty4)
        _W_CORE.set_offsets(empty2); _W_CORE.set_sizes([]); _W_CORE.set_facecolors(empty4)
    else:
        print("Render frame", frame_idx, "punkty:", len(entry["offsets"]))
        print(entry["offsets"][:5])
        _W_OUTER.set_offsets(entry["offsets"]); _W_OUTER.set_sizes(entry["outer_size"]); _W_OUTER.set_facecolors(entry["outer_rgba"])
        _W_INNER.set_offsets(entry["offsets"]); _W_INNER.set_sizes(entry["inner_size"]); _W_INNER.set_facecolors(entry["inner_rgba"])
        _W_CORE.set_offsets(entry["offsets"]); _W_CORE.set_sizes(entry["core_size"]); _W_CORE.set_facecolors(entry["core_rgba"])

    _W_FIG.savefig(out_path, dpi=_W_FIG.dpi, facecolor="black")
    return frame_idx


def render_animation_parallel(frame_cache, frames):
    FRAMES_DIR.mkdir(exist_ok=True)

    render_meta = {
        "version": FRAME_CACHE_VERSION,
        "frames": frames,
        "dpi": ANIMATION_DPI,
        "fps": ANIMATION_FPS,
        "fig_size": (FIG_W_IN, FIG_H_IN),
    }
    previous_meta = None
    if FRAMES_META_FILE.exists():
        with open(FRAMES_META_FILE, "rb") as fh:
            previous_meta = pickle.load(fh)

    if previous_meta != render_meta:
        removed = 0
        for old_frame in FRAMES_DIR.glob("frame_*.png"):
            old_frame.unlink()
            removed += 1
        if removed:
            print(f"Usunięto {removed} starych klatek renderu — parametry animacji się zmieniły.")

    worker_cache_file = FRAMES_DIR / "_frame_cache_for_workers.pkl"
    with open(worker_cache_file, "wb") as fh:
        pickle.dump(frame_cache, fh, protocol=pickle.HIGHEST_PROTOCOL)

    tasks = []
    for i in range(frames):
        out_path = FRAMES_DIR / f"frame_{i:06d}.png"
        if out_path.exists():
            continue  # cache: ta klatka jest już wyrenderowana — pomijamy
        tasks.append((i, str(out_path)))

    if tasks:
        print(f"Wykryto {CPU_COUNT} rdzeni CPU. Używam {N_WORKERS} procesów roboczych.")
        print(f"Renderowanie {len(tasks)}/{frames} brakujących klatek na {N_WORKERS} rdzeniach...")
        with mp.Pool(
            processes=N_WORKERS,
            initializer=_init_worker,
            initargs=(str(MAP_FILE), str(MAP_LAYOUT_FILE), str(worker_cache_file), ANIMATION_DPI),
        ) as pool:
            for _ in tqdm(pool.imap_unordered(_render_one_frame, tasks),
                          total=len(tasks), desc="Renderowanie klatek", unit="klatka"):
                pass
    else:
        print("Wszystkie klatki już wyrenderowane wcześniej (cache) — pomijam rendering.")

    print("Kodowanie wideo (ffmpeg, jeden przebieg, wielowątkowo)...")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(ANIMATION_FPS),
        "-i", str(FRAMES_DIR / "frame_%06d.png"),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-threads", str(N_WORKERS),
        ANIMATION_OUT,
    ]
    subprocess.run(cmd, check=True)
    with open(FRAMES_META_FILE, "wb") as fh:
        pickle.dump(render_meta, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Zapisano animację: {ANIMATION_OUT}")


# ─────────────────────────────────────────────────────────────
#  GŁÓWNY PRZEBIEG PROGRAMU
# ─────────────────────────────────────────────────────────────
def main():
    program_start = time.perf_counter()

    plt.rcParams.update({
        "figure.facecolor": "black",
        "axes.facecolor": "black",
        "text.color": "white",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
    })

    use_cached_map = MAP_FILE.exists()
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=300, facecolor="black")
    ax = fig.add_subplot(111, projection="hammer", facecolor="black")

    if use_cached_map:
        load_cached_background(fig, ax)
    else:
        cmap_mw = build_colormap()
        build_and_save_sky_map(fig, ax, cmap_mw)

    # Figura mapy statycznej nie jest już potrzebna — każdy proces roboczy
    # buduje własną, lekką figurę tylko z tłem PNG + kolekcjami punktów.
    plt.close(fig)

    microlens = prepare_microlensing_data()

    frame_cache = None
    if FRAME_CACHE_FILE.exists():
        print(f"Wczytano gotowy cache danych animacji: {FRAME_CACHE_FILE}")
        with open(FRAME_CACHE_FILE, "rb") as fh:
            cached_payload = pickle.load(fh)
        if (
            isinstance(cached_payload, dict)
            and cached_payload.get("version") == FRAME_CACHE_VERSION
            and cached_payload.get("frames") == FRAMES
            and cached_payload.get("dpi") == ANIMATION_DPI
            and cached_payload.get("fps") == ANIMATION_FPS
        ):
            frame_cache = cached_payload["frame_cache"]
        else:
            print("Cache danych jest nieaktualny — przeliczam błyski od nowa.")

    if frame_cache is None:
        print("Liczenie danych animacji dla wszystkich klatek naraz (wektorowo)...")
        _current_times, frame_cache = precompute_all_frames(microlens, FRAMES)
        cache_payload = {
            "version": FRAME_CACHE_VERSION,
            "frames": FRAMES,
            "dpi": ANIMATION_DPI,
            "fps": ANIMATION_FPS,
            "frame_cache": frame_cache,
        }
        with open(FRAME_CACHE_FILE, "wb") as fh:
            pickle.dump(cache_payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Zapisano cache danych animacji: {FRAME_CACHE_FILE}")

    render_animation = input("Czy wyrenderować animację mikrosoczewkowania? (t/n): ").strip().lower()

    if render_animation in ("t", "tak", "y", "yes"):
        expected_w = round(FIG_W_IN * ANIMATION_DPI)
        expected_h = round(FIG_H_IN * ANIMATION_DPI)
        print(f"Rozmiar klatki animacji: {expected_w}x{expected_h} px "
              f"(figura {FIG_W_IN:.2f}x{FIG_H_IN:.2f}\" @ {ANIMATION_DPI} dpi)")
        if expected_w > X264_MAX_DIM or expected_h > X264_MAX_DIM:
            raise RuntimeError(
                f"Rozmiar klatki {expected_w}x{expected_h} px przekracza limit "
                f"kodeka x264 ({X264_MAX_DIM} px). Obniż ANIMATION_DPI."
            )
        render_animation_parallel(frame_cache, FRAMES)
    else:
        print("Pominięto renderowanie animacji")

    elapsed = time.perf_counter() - program_start
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    print(f"\nCałkowity czas działania programu: {hours:02d}:{minutes:02d}:{seconds:05.2f}")


if __name__ == "__main__":
    # Guard konieczny dla multiprocessing (szczególnie start method
    # "spawn", domyślny na macOS/Windows) — bez niego procesy robocze
    # próbowałyby wykonać cały skrypt od nowa (w tym ciężkie wczytywanie
    # katalogu Gaia), zamiast tylko funkcji renderującej.
    main()
