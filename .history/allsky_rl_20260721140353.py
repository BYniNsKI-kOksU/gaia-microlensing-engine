"""
allsky_rl.py — mapa Drogi Mlecznej (8K) z animacją mikrosoczewkowania Gaia.

Zmiany względem oryginału:
  1. Mapa nieba (PNG) jest generowana raz i cache'owana.  Kolejne uruchomienia
     wczytują gotowy plik — histogram nie jest przeliczany od nowa.
  2. Animacja jest renderowana równolegle na N_WORKERS = 4 rdzeniach CPU.
     Każdy proces roboczy buduje własną figurę raz (tło PNG + kolekcja punktów),
     a potem tylko podmienia dane i zapisuje PNG klatki — bez ponownego rysowania
     mapy w każdej klatce.
  3. Logika zjawisk mikrosoczewkowania jest identyczna jak w allsky.py:
       – filtracja valid + percentylowa filtracja tmax (2–98 %)
       – okno czasowe event_window = 2.75 × t_E
       – anim_start / anim_end wyznaczane z min/max(t_max ± event_window)
       – widoczność: |Δt| ≤ event_window  (zanik wynikający z krzywej Paczyńskiego)
  4. Styl wizualny animacji zachowany bez zmian: pojedynczy scatter,
     paleta 'cool', alpha = 0.9, białe obramowanie (edgecolors='white'), linewidth 0.25.
  5. N_WORKERS = 4 — na stałe; nie dobieramy automatycznie.
"""

import matplotlib
# Musi być przed importem pyplot; wymusza backend rastrowy (bez GUI).
matplotlib.use("Agg")

import os
import pickle
import subprocess
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
INPUT_FILE      = "gaia_60m_allsky.fits"
MAP_FILE        = Path("gaia_allsky_hammer_8k.png")
MAP_LAYOUT_FILE = Path("gaia_allsky_hammer_8k_layout.npz")
MICROLENS_FILE  = "gaia_microlensing.fits"

FRAME_CACHE_FILE = Path("frame_cache_rl.pkl")
FRAMES_DIR       = Path("frames_tmp_rl")
ANIMATION_OUT    = "gaia_microlensing_animation.mp4"

BINS_L = 16384
BINS_B = 8192

FRAMES        = 750
ANIMATION_DPI = 200
ANIMATION_FPS = 25
FRAME_CACHE_VERSION = 1

FIG_W_IN, FIG_H_IN = 32.0, 18.0

# Liczba procesów roboczych — na stałe 4 (wymaganie 6).
# Nie dobieramy automatycznie na podstawie os.cpu_count().
N_WORKERS = 4

# ─────────────────────────────────────────────────────────────
#  PALETA — identyczna jak w oryginale
# ─────────────────────────────────────────────────────────────
_PALETTE = [
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


def _build_cmap():
    return LinearSegmentedColormap.from_list(
        "milky_way_visible", [(v, c) for v, c in _PALETTE], N=512
    )


# ─────────────────────────────────────────────────────────────
#  AMPLIFIKACJA PACZYŃSKIEGO
# ─────────────────────────────────────────────────────────────
def paczynski_amplification(t, t0, te, u0):
    u = np.sqrt(u0 ** 2 + ((t - t0) / te) ** 2)
    return (u * u + 2) / (u * np.sqrt(u * u + 4))


# ─────────────────────────────────────────────────────────────
#  GENEROWANIE MAPY TŁA
#  Logika identyczna jak w oryginale — zmiana tylko w sposobie
#  zapisu: brak bbox_inches="tight" (PNG ma dokładnie dpi×figsize px)
#  i dodatkowy zapis układu osi do .npz (potrzebny do wyrównania warstw).
# ─────────────────────────────────────────────────────────────
def generate_sky_map():
    plt.rcParams.update({
        "figure.facecolor": "black",
        "axes.facecolor":   "black",
        "text.color":       "white",
        "axes.labelcolor":  "white",
        "xtick.color":      "white",
        "ytick.color":      "white",
    })

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
    l = -l   # centrum galaktyki (l=0) w środku, l=90 w prawo

    hist, lon_edges, lat_edges = np.histogram2d(
        l, b,
        bins=[BINS_L, BINS_B],
        range=[[-np.pi, np.pi], [-np.pi / 2, np.pi / 2]],
    )

    hist = gaussian_filter(hist.T, sigma=2)
    hist = np.log10(hist + 1)

    p_low  = np.percentile(hist, 5)
    p_high = np.percentile(hist, 98.5)
    hist = (hist - p_low) / (p_high - p_low)
    hist = np.clip(hist, 0, 1)
    hist = hist ** 0.55

    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    LON, LAT = np.meshgrid(lon_centers, lat_centers)

    cmap_mw = _build_cmap()

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=300, facecolor="black")
    ax  = fig.add_subplot(111, projection="hammer", facecolor="black")

    ax.pcolormesh(LON, LAT, hist, shading="auto", cmap=cmap_mw)

    # Siatka
    ax.grid(True, linewidth=0.25, alpha=0.20, color="white", linestyle="--")

    # Szerokość galaktyczna b — etykiety po lewej
    b_ticks = np.array([-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75])
    ax.set_yticks(np.radians(b_ticks))
    ax.set_yticklabels(
        [f"{v}°" for v in b_ticks],
        fontsize=6.5, color="white", alpha=0.75,
    )
    ax.set_xticklabels([])

    # Adnotacje l pod dolnym łukiem elipsy
    for deg in range(-150, 180, 30):
        rad   = np.radians(deg)
        label = "0°" if deg == 0 else f"{abs(deg)}°"
        ax.annotate(
            label,
            xy=(rad, -1.62), xycoords="data",
            ha="center", va="top",
            fontsize=6, color="white", alpha=0.65,
            annotation_clip=False,
        )

    # Opisy osi
    ax.annotate(
        "Długość galaktyczna  l",
        xy=(0.5, -0.07), xycoords="axes fraction",
        ha="center", va="top", fontsize=9, color="white",
        annotation_clip=False,
    )
    ax.annotate(
        "Szerokość galaktyczna  b",
        xy=(-0.055, 0.5), xycoords="axes fraction",
        va="center", rotation=90, fontsize=9, color="white",
        annotation_clip=False,
    )

    # Ramka
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
        spine.set_linewidth(0.4)

    plt.tight_layout(pad=0.5)

    # Zapisz pozycję osi w układzie znormalizowanym figury (0–1).
    # Dane te są potrzebne workerom do prawidłowego nakładania punktów
    # na obraz PNG (wyrównanie warstw).
    np.savez(
        MAP_LAYOUT_FILE,
        axes_bounds=np.array(ax.get_position().bounds, dtype=np.float64),
    )

    # Zapis bez bbox_inches="tight": PNG ma DOKŁADNIE dpi×figsize pikseli,
    # więc ax.get_position().bounds dokładnie opisuje, gdzie jest oś w obrazie.
    plt.savefig(str(MAP_FILE), dpi=300, facecolor="black")
    print(f"Zapisano mapę:        {MAP_FILE}")
    print(f"Zapisano układ osi:   {MAP_LAYOUT_FILE}")

    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  POZYCJA OSI HAMMER
# ─────────────────────────────────────────────────────────────
def _get_axes_bounds():
    """
    Zwraca granice osi Hammer jako (x0, y0, w, h) w układzie figury [0, 1].
    Czyta z pliku .npz (zapisanego przez generate_sky_map); jeśli nie istnieje,
    wyznacza przez odtworzenie identycznej konfiguracji figury (fallback).
    """
    if MAP_LAYOUT_FILE.exists():
        with np.load(MAP_LAYOUT_FILE) as lay:
            return tuple(float(v) for v in lay["axes_bounds"])

    # Fallback: odtwórz konfigurację identyczną z generate_sky_map()
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=300, facecolor="black")
    ax  = fig.add_subplot(111, projection="hammer", facecolor="black")
    b_ticks = np.array([-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75])
    ax.set_yticks(np.radians(b_ticks))
    ax.set_yticklabels([f"{v}°" for v in b_ticks], fontsize=6.5)
    ax.set_xticklabels([])
    plt.tight_layout(pad=0.5)
    bounds = tuple(float(v) for v in ax.get_position().bounds)
    plt.close(fig)
    return bounds


# ─────────────────────────────────────────────────────────────
#  DANE MIKROSOCZEWKOWANIA
#  Identyczna logika jak w allsky.py: filtracja valid, percentylowa
#  filtracja tmax (2–98 %), okno czasowe 2.75 × t_E.
# ─────────────────────────────────────────────────────────────
def prepare_microlensing_data():
    print(f"Wczytywanie mikrosoczewkowania: {MICROLENS_FILE}")
    microlensing = Table.read(MICROLENS_FILE).to_pandas()
    microlensing = microlensing.dropna(subset=["l", "b"])
    print(f"Zdarzenia mikrosoczewkowania: {len(microlensing):,}")

    ml_l = np.radians(microlensing["l"].values)
    ml_b = np.radians(microlensing["b"].values)
    # Identyczne odbicie jak dla mapy gwiazd
    ml_l = -np.where(ml_l > np.pi, ml_l - 2 * np.pi, ml_l)

    ml_tmax = microlensing["paczynski0_tmax"].values
    ml_te   = microlensing["paczynski0_te"].values
    ml_u0   = microlensing["paczynski0_u0"].values

    # — Filtracja identyczna jak w allsky.py —
    # Krok 1: odrzuć nieskończone i niefizyczne wartości
    valid = (
        np.isfinite(ml_tmax) & np.isfinite(ml_te)
        & np.isfinite(ml_u0) & (ml_te > 0)
    )
    ml_l, ml_b       = ml_l[valid], ml_b[valid]
    ml_tmax, ml_te, ml_u0 = ml_tmax[valid], ml_te[valid], ml_u0[valid]

    # Krok 2: ogranicz zakres t_max do 2–98 percentyla (usuwa outliers)
    tmax_lo = np.nanpercentile(ml_tmax, 2)
    tmax_hi = np.nanpercentile(ml_tmax, 98)
    core = (ml_tmax >= tmax_lo) & (ml_tmax <= tmax_hi)
    ml_l, ml_b       = ml_l[core], ml_b[core]
    ml_tmax, ml_te, ml_u0 = ml_tmax[core], ml_te[core], ml_u0[core]

    print(f"Po filtracji: {len(ml_l):,} zdarzeń")

    # — Okno czasowe i zakres animacji — identyczne jak w allsky.py —
    event_window = 2.75 * ml_te
    anim_start   = float(np.nanmin(ml_tmax - event_window))
    anim_end     = float(np.nanmax(ml_tmax + event_window))

    return dict(
        l=ml_l, b=ml_b,
        tmax=ml_tmax, te=ml_te, u0=ml_u0,
        anim_start=anim_start, anim_end=anim_end,
    )


# ─────────────────────────────────────────────────────────────
#  PREKALKULACJA DANYCH WSZYSTKICH KLATEK
#  Logika czasowa (widoczność, zanik) identyczna jak w allsky.py.
#  Styl wizualny zachowany jak w oryginale allsky_rl.py:
#    – pojedynczy scatter, paleta 'cool', alpha = 0.9,
#    – rozmiar = (A − 1) × 500, białe krawędzie.
# ─────────────────────────────────────────────────────────────
def precompute_all_frames(m, frames):
    from matplotlib import colormaps
    cool = colormaps["cool"]

    current_times = np.linspace(m["anim_start"], m["anim_end"], frames)

    T            = current_times[:, None]         # (F, 1)
    TMAX         = m["tmax"][None, :]              # (1, N)
    TE           = m["te"][None, :]
    U0           = m["u0"][None, :]
    EVENT_WINDOW = (2.75 * m["te"])[None, :]      # identyczne jak w allsky.py

    delta   = T - TMAX                            # (F, N)
    # Warunek widoczności identyczny jak w allsky.py: |Δt| ≤ event_window
    visible = np.abs(delta) <= EVENT_WINDOW        # (F, N)

    u   = np.sqrt(U0 ** 2 + (delta / TE) ** 2)
    amp = (u * u + 2) / (u * np.sqrt(u * u + 4)) # (F, N)

    # Globalne clim dla spójnego odwzorowania kolorów między klatkami.
    # Clamp w 99. percentylu, żeby skrajne wartości nie wyprały kolorów.
    amp_vis_all   = amp[visible]
    amp_clim_max  = float(np.nanpercentile(amp_vis_all, 99)) if amp_vis_all.size > 0 else 2.0

    frame_cache = []
    for f in tqdm(range(frames), desc="Liczenie danych klatek", unit="klatka"):
        idx = np.nonzero(visible[f])[0]
        if idx.size == 0:
            frame_cache.append(None)
            continue

        amp_vis = amp[f, idx]                                          # (K,)
        sizes   = np.clip((amp_vis - 1.0) * 500.0, 0.0, None).astype(np.float32)

        # Odwzorowanie amplifikacji → kolor w palecie 'cool'
        # (identyczny styl jak oryginalny set_array(amp[visible]) z cmap='cool')
        norm = np.clip(
            (amp_vis - 1.0) / max(amp_clim_max - 1.0, 1e-10),
            0.0, 1.0,
        )
        rgba       = cool(norm).astype(np.float32)
        rgba[:, 3] = 0.9   # stały alfa — identyczny jak alpha=0.9 w oryginalnym scatter

        offsets = np.column_stack((m["l"][idx], m["b"][idx])).astype(np.float32)
        frame_cache.append({
            "offsets": offsets,
            "sizes":   sizes,
            "rgba":    rgba,
        })

    return current_times, frame_cache


# ─────────────────────────────────────────────────────────────
#  WORKER — buduje figurę RAZ, renderuje przydzielone klatki
# ─────────────────────────────────────────────────────────────
# Globalne zmienne procesu roboczego (wypełniane przez _init_worker)
_W_FIG     = None
_W_SCATTER = None
_W_CACHE   = None


def _init_worker(map_file, ax_bounds, cache_file, animation_dpi):
    """
    Inicjalizator procesu roboczego: buduje figurę z tłem PNG i kolekcją punktów.
    Wywoływany raz na proces — konfiguracja jest współdzielona przez wszystkie
    klatki przydzielone do tego procesu.
    """
    global _W_FIG, _W_SCATTER, _W_CACHE

    plt.rcParams.update({"figure.facecolor": "black", "axes.facecolor": "black"})

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=animation_dpi, facecolor="black")

    # ── Warstwa 1: mapa nieba jako nieruchome tło ──
    bg_ax = fig.add_axes([0, 0, 1, 1], zorder=-10)
    bg_img = mpimg.imread(map_file)
    bg_ax.imshow(bg_img, origin="upper", interpolation="nearest", aspect="auto")
    bg_ax.axis("off")

    # ── Warstwa 2: przezroczysta projekcja Hammer z animowanymi punktami ──
    ax = fig.add_subplot(111, projection="hammer", facecolor="none")
    ax.set_position(ax_bounds)   # wyrównanie z układem mapy w PNG
    ax.set_zorder(10)
    ax.patch.set_alpha(0.0)
    ax.set_axis_off()            # ukrywa osie/ticki/ramkę (są w PNG); scatter pozostaje

    # Scatter z identycznym stylem jak w oryginalnym allsky_rl.py
    scatter = ax.scatter(
        [], [], s=[], c=[],
        edgecolors="white", linewidths=0.25,
        zorder=15,
    )

    with open(cache_file, "rb") as fh:
        cache = pickle.load(fh)

    _W_FIG, _W_SCATTER, _W_CACHE = fig, scatter, cache


def _render_one_frame(task):
    """Renderuje jedną klatkę i zapisuje ją jako PNG."""
    frame_idx, out_path = task
    entry   = _W_CACHE[frame_idx]
    empty2  = np.empty((0, 2))
    empty4  = np.empty((0, 4))

    if entry is None:
        _W_SCATTER.set_offsets(empty2)
        _W_SCATTER.set_sizes([])
        _W_SCATTER.set_facecolors(empty4)
    else:
        _W_SCATTER.set_offsets(entry["offsets"])
        _W_SCATTER.set_sizes(entry["sizes"])
        _W_SCATTER.set_facecolors(entry["rgba"])

    _W_FIG.savefig(out_path, dpi=_W_FIG.dpi, facecolor="black")
    return frame_idx


# ─────────────────────────────────────────────────────────────
#  RENDEROWANIE RÓWNOLEGŁE + KODOWANIE WIDEO
# ─────────────────────────────────────────────────────────────
def render_animation_parallel(frame_cache, frames, ax_bounds):
    FRAMES_DIR.mkdir(exist_ok=True)

    # Zapisz cache klatek do pliku tymczasowego dostępnego dla workerów
    worker_cache_file = FRAMES_DIR / "_frame_cache_rl_workers.pkl"
    with open(worker_cache_file, "wb") as fh:
        pickle.dump(frame_cache, fh, protocol=pickle.HIGHEST_PROTOCOL)

    # Zbierz klatki do wyrenderowania (istniejące pomijamy — wznawianie renderu)
    tasks = [
        (i, str(FRAMES_DIR / f"frame_{i:06d}.png"))
        for i in range(frames)
        if not (FRAMES_DIR / f"frame_{i:06d}.png").exists()
    ]

    if tasks:
        print(
            f"Renderowanie {len(tasks)}/{frames} klatek "
            f"na {N_WORKERS} rdzeniach (N_WORKERS na stałe = {N_WORKERS})..."
        )
        with mp.Pool(
            processes=N_WORKERS,
            initializer=_init_worker,
            initargs=(str(MAP_FILE), ax_bounds, str(worker_cache_file), ANIMATION_DPI),
        ) as pool:
            for _ in tqdm(
                pool.imap_unordered(_render_one_frame, tasks),
                total=len(tasks), desc="Renderowanie klatek", unit="klatka",
            ):
                pass
    else:
        print("Wszystkie klatki już wyrenderowane (cache) — pomijam rendering.")

    # Jedno wywołanie ffmpeg koduje całość wielowątkowo
    print("Kodowanie wideo (ffmpeg)...")
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
    print(f"Zapisano animację: {ANIMATION_OUT}")


# ─────────────────────────────────────────────────────────────
#  GŁÓWNY PRZEBIEG
# ─────────────────────────────────────────────────────────────
def main():
    # 1. Mapa nieba — wczytaj istniejącą lub wygeneruj nową
    if MAP_FILE.exists() and MAP_LAYOUT_FILE.exists():
        print(f"Używam istniejącej mapy: {MAP_FILE}")
    else:
        if MAP_FILE.exists():
            print(
                f"Znaleziono {MAP_FILE}, ale brak pliku układu {MAP_LAYOUT_FILE}.\n"
                "Mapa zostanie wygenerowana ponownie, aby zapisać układ osi."
            )
        else:
            print(f"Brak pliku mapy {MAP_FILE} — generowanie od nowa...")
        generate_sky_map()

    # 2. Wyznacz pozycję osi Hammer w układzie figury
    ax_bounds = _get_axes_bounds()
    print(f"Granice osi Hammer (x0, y0, w, h): {tuple(f'{v:.4f}' for v in ax_bounds)}")

    # 3. Dane mikrosoczewkowania z identyczną logiką jak w allsky.py
    microlens = prepare_microlensing_data()

    # 4. Cache danych klatek (prekalkulacja wektorowa)
    frame_cache = None
    if FRAME_CACHE_FILE.exists():
        with open(FRAME_CACHE_FILE, "rb") as fh:
            cached = pickle.load(fh)
        if (
            isinstance(cached, dict)
            and cached.get("version") == FRAME_CACHE_VERSION
            and cached.get("frames") == FRAMES
        ):
            frame_cache = cached["frame_cache"]
            print(f"Wczytano gotowy cache danych animacji: {FRAME_CACHE_FILE}")
        else:
            print("Cache danych jest nieaktualny — przeliczam od nowa.")

    if frame_cache is None:
        print("Obliczanie danych animacji dla wszystkich klatek (wektorowo)...")
        _, frame_cache = precompute_all_frames(microlens, FRAMES)
        with open(FRAME_CACHE_FILE, "wb") as fh:
            pickle.dump(
                {
                    "version": FRAME_CACHE_VERSION,
                    "frames":  FRAMES,
                    "frame_cache": frame_cache,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"Zapisano cache danych animacji: {FRAME_CACHE_FILE}")

    # 5. Renderowanie równoległe + kodowanie wideo
    render_animation_parallel(frame_cache, FRAMES, ax_bounds)

    print("\nGotowe.")


if __name__ == "__main__":
    # Guard konieczny dla multiprocessing na macOS/Windows (metoda 'spawn'):
    # bez niego procesy robocze próbowałyby wykonać cały skrypt od nowa.
    main()
