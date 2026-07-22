"""
allsky_animation.py — Animacja mikrosoczewkowania na pustej mapie allsky (projekcja Hammer).

Wymagane pliki w katalogu roboczym:
  - gaia_microlensing.fits   (dane zdarzeń mikrosoczewkowania)

Wyjście:
  - gaia_microlensing_animation.mp4

Cache (automatycznie tworzone/usuwane):
  - frame_cache.pkl          (dane wszystkich klatek wektorowo)
  - frames_tmp/              (renderowane klatki PNG)
"""

import matplotlib
matplotlib.use("Agg")

import os
import pickle
import subprocess
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
#  KONFIGURACJA
# ─────────────────────────────────────────────────────────────
MICROLENS_FILE = "gaia_microlensing.fits"

FRAME_CACHE_FILE = Path("frame_cache.pkl")
FRAMES_DIR = Path("frames_tmp")
FRAMES_META_FILE = FRAMES_DIR / "_render_meta.pkl"
ANIMATION_OUT = "gaia_microlensing_animation.mp4"

FRAMES = 100
ANIMATION_DPI = 300
ANIMATION_FPS = 25
FRAME_CACHE_VERSION = 4

FIG_W_IN, FIG_H_IN = 53.333, 30.0
X264_MAX_DIM = 16384

CPU_COUNT = os.cpu_count() or 1
N_WORKERS = max(1, min(8, int(CPU_COUNT * 0.6)))


# ─────────────────────────────────────────────────────────────
#  POMOCNICZE
# ─────────────────────────────────────────────────────────────
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

    # Nie usuwamy skrajnych zdarzeń czasowych, ponieważ mogą one
    # zawierać jedyne widoczne błyski w krótkiej animacji.
    core = np.ones_like(ml_tmax, dtype=bool)
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
    current_times = np.linspace(m["anim_start"], m["anim_end"], frames)

    T = current_times[:, None]
    TMAX = m["tmax"][None, :]
    TE = m["te"][None, :]
    U0 = m["u0"][None, :]
    EVENT_WINDOW = (5.0 * m["te"])[None, :]

    delta = T - TMAX
    visible = np.abs(delta) <= EVENT_WINDOW
    visible_counts = np.sum(visible, axis=1)
    print("Widoczne zdarzenia:")
    print(visible_counts)
    print("Łącznie aktywnych punktów w cache:", np.sum(visible_counts))
    if np.max(visible_counts) == 0:
        raise RuntimeError("Żadne zdarzenie nie mieści się w czasie animacji. Sprawdź paczynski0_tmax/te.")

    u = np.sqrt(U0 ** 2 + (delta / TE) ** 2)
    amp = (u * u + 2) / (u * np.sqrt(u * u + 4))

    contrast = np.clip(amp - 1.0, 0.0, None)
    amp_boost = np.log1p(contrast * 9.0) / np.log1p(9.0)

    mass_factor = np.clip(0.85 + 3.4 * m["mass_norm"][None, :], 0.85, 4.25)
    brightness_factor = np.clip(0.55 + 1.15 * amp_boost, 0.55, 1.70)
    core_size = 40.0 * mass_factor * brightness_factor
    inner_size = core_size * 5.0
    outer_size = core_size * 15.0

    # biały błysk gwiazdy: wszystkie pierścienie neutralnie białe
    rgb = np.ones((frames, len(m["l"]), 3), dtype=np.float32)

    fade = np.clip(1.0 - np.abs(delta) / EVENT_WINDOW, 0.0, 1.0)
    fade = fade * fade * (3.0 - 2.0 * fade)
    pulse = fade * np.clip(0.35 + 0.65 * amp_boost, 0.0, 1.0)

    outer_alpha = np.clip(0.20 + 0.50 * pulse, 0.0, 0.70)
    inner_alpha = np.clip(0.40 + 0.60 * pulse, 0.0, 0.90)
    core_alpha = np.ones_like(pulse)

    print("Maksymalna liczba aktywnych zdarzeń w klatce:", np.max(np.sum(visible, axis=1)))
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
            "outer_rgba": _rgba_stack([1.0, 1.0, 1.0], outer_alpha[f, idx]).astype(np.float32),
            "inner_size": inner_size[f, idx].astype(np.float32),
            "inner_rgba": _rgba_stack([1.0, 1.0, 1.0], inner_alpha[f, idx]).astype(np.float32),
            "core_size": core_size[f, idx].astype(np.float32),
            "core_rgba": _rgba_stack([1.0, 1.0, 1.0], core_alpha[f, idx]).astype(np.float32),
        })

    return current_times, frame_cache


# ─────────────────────────────────────────────────────────────
#  RENDEROWANIE RÓWNOLEGŁE (pusta mapa Hammer)
# ─────────────────────────────────────────────────────────────
_W_FIG = _W_OUTER = _W_INNER = _W_CORE = _W_CACHE = None


def _init_worker(animation_dpi):
    global _W_FIG, _W_OUTER, _W_INNER, _W_CORE, _W_CACHE

    plt.rcParams.update({
        "figure.facecolor": "black",
        "axes.facecolor": "black",
        "text.color": "white",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
    })

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=animation_dpi, facecolor="black")
    ax = fig.add_subplot(111, projection="hammer", facecolor="black")
    ax.set_facecolor("black")
    ax.set_axis_off()

    outer = ax.scatter([], [], s=[], c=[], marker="o", edgecolors="none", zorder=4)
    inner = ax.scatter([], [], s=[], c=[], marker="o", edgecolors="none", zorder=5)
    core = ax.scatter([], [], s=[], c=[], marker="o", edgecolors="none", zorder=6)

    _W_FIG, _W_OUTER, _W_INNER, _W_CORE = fig, outer, inner, core


def _render_one_frame(task):
    frame_idx, out_path, worker_cache_file = task

    global _W_CACHE
    if _W_CACHE is None:
        with open(worker_cache_file, "rb") as fh:
            _W_CACHE = pickle.load(fh)

    entry = _W_CACHE[frame_idx]
    empty2 = np.empty((0, 2))
    empty4 = np.empty((0, 4))

    if entry is None:
        _W_OUTER.set_offsets(empty2); _W_OUTER.set_sizes([]); _W_OUTER.set_facecolors(empty4)
        _W_INNER.set_offsets(empty2); _W_INNER.set_sizes([]); _W_INNER.set_facecolors(empty4)
        _W_CORE.set_offsets(empty2); _W_CORE.set_sizes([]); _W_CORE.set_facecolors(empty4)
    else:
        if frame_idx == 0:
            print("Pierwsza renderowana klatka, liczba błysków:", len(entry["offsets"]))
            print("Zakres l:", entry["offsets"][:,0].min(), entry["offsets"][:,0].max())
            print("Zakres b:", entry["offsets"][:,1].min(), entry["offsets"][:,1].max())
        _W_OUTER.set_offsets(entry["offsets"]); _W_OUTER.set_sizes(entry["outer_size"]); _W_OUTER.set_facecolors(entry["outer_rgba"])
        _W_INNER.set_offsets(entry["offsets"]); _W_INNER.set_sizes(entry["inner_size"]); _W_INNER.set_facecolors(entry["inner_rgba"])
        _W_CORE.set_offsets(entry["offsets"]); _W_CORE.set_sizes(entry["core_size"]); _W_CORE.set_facecolors(entry["core_rgba"])

    _W_FIG.savefig(out_path, dpi=_W_FIG.dpi, facecolor="black", bbox_inches=None)
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
            continue
        tasks.append((i, str(out_path), str(worker_cache_file)))

    if tasks:
        print(f"Wykryto {CPU_COUNT} rdzeni CPU. Używam {N_WORKERS} procesów roboczych.")
        print(f"Renderowanie {len(tasks)}/{frames} brakujących klatek na {N_WORKERS} rdzeniach...")
        with mp.Pool(
            processes=N_WORKERS,
            initializer=_init_worker,
            initargs=(ANIMATION_DPI,),
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

    elapsed = time.perf_counter() - program_start
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    print(f"\nCałkowity czas działania programu: {hours:02d}:{minutes:02d}:{seconds:05.2f}")


if __name__ == "__main__":
    main()