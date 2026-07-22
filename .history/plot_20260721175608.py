import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from astropy.table import Table
from pathlib import Path
import subprocess
import os
from tqdm import tqdm


# -----------------------------
# KONFIGURACJA
# -----------------------------

MICROLENS_FILE = "gaia_microlensing.fits"

MAP_FILE = "gaia_allsky_hammer_16k.png"

OUT_DIR = Path("frames_micro")
VIDEO = "microlensing_animation.mp4"
PNG_CACHE = Path("microlensing_events.png")

FRAMES = 750
FPS = 25


# -----------------------------
# PACZYNSKI
# -----------------------------

def amplification(t, t0, te, u0):

    u = np.sqrt(
        u0*u0 +
        ((t-t0)/te)**2
    )

    return (
        (u*u+2)
        /
        (u*np.sqrt(u*u+4))
    )


def create_background_map():

    print("Tworzenie mapy Hammer...")

    fig = plt.figure(
        figsize=(16, 9),
        dpi=150,
        facecolor="white"
    )

    ax = fig.add_subplot(
        111,
        projection="hammer"
    )

    ax.set_facecolor("white")
    ax.axis("off")

    lon = np.linspace(-np.pi, np.pi, 400)
    lat = np.zeros_like(lon)

    ax.plot(
        lon,
        lat,
        color="lightgray",
        linewidth=0.5
    )

    fig.savefig(
        MAP_FILE,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close(fig)

    print("Zapisano mapę:", MAP_FILE)


# -----------------------------
# WCZYTYWANIE DANYCH
# -----------------------------

def load_events():

    print("Wczytywanie FITS...")

    tab = Table.read(
        MICROLENS_FILE
    ).to_pandas()


    tab = tab.dropna(
        subset=["l","b"]
    )


    l = np.radians(tab["l"].values)
    b = np.radians(tab["b"].values)


    l = -np.where(
        l > np.pi,
        l-2*np.pi,
        l
    )


    tmax = tab["paczynski0_tmax"].values
    te = tab["paczynski0_te"].values
    u0 = tab["paczynski0_u0"].values


    mass = np.ones(len(tab))


    for col in [
        "paczynski0_mass",
        "paczynski_mass",
        "lens_mass",
        "mass"
    ]:
        if col in tab.columns:
            mass = tab[col].values
            print("Używam masy:",col)
            break


    good = (
        np.isfinite(tmax)
        &
        np.isfinite(te)
        &
        np.isfinite(u0)
        &
        (te>0)
    )

    l = l[good]
    b = b[good]
    tmax = tmax[good]
    te = te[good]
    u0 = u0[good]
    mass = mass[good]

    # Normalizacja czasów względem początku
    tmax = tmax - np.min(tmax)

    return (
        l,
        b,
        tmax,
        te,
        u0,
        mass
    )



# -----------------------------
# RENDER
# -----------------------------


def main():

    OUT_DIR.mkdir(
        exist_ok=True
    )


    l,b,tmax,te,u0,mass = load_events()


    print(
        "Zdarzenia:",
        len(l)
    )


    te_max = np.percentile(te, 95)
    duration = 6 * te_max

    start = np.min(tmax) - duration / 2
    end = np.max(tmax) + duration / 2

    times = np.linspace(
        start,
        end,
        FRAMES
    )

    if not Path(MAP_FILE).exists():
        raise FileNotFoundError(f"Brak gotowej mapy: {MAP_FILE}")
    else:
        print("Używam gotowej mapy Drogi Mlecznej:", MAP_FILE)

    bg = plt.imread(MAP_FILE)

    # Generowanie statycznego PNG jeśli nie istnieje
    if not PNG_CACHE.exists():
        fig_png = plt.figure(
            figsize=(16, 9),
            dpi=150,
            facecolor="black"
        )

        ax_png = fig_png.add_subplot(
            111,
            projection="hammer"
        )

        ax_png.set_facecolor("white")
        ax_png.axis("off")

        amp_png = amplification(
            np.median(tmax),
            tmax,
            te,
            u0
        )

        visible_png = np.abs(np.median(tmax) - tmax) < te * 2.5

        if np.any(visible_png):
            strength_png = np.clip(
                amp_png[visible_png] - 1,
                0,
                None
            )

            size_png = 10 + 150 * np.log1p(strength_png)

            ax_png.scatter(
                l[visible_png],
                b[visible_png],
                s=size_png,
                c="red",
                alpha=1,
                edgecolors="none"
            )

        fig_png.savefig(
            PNG_CACHE,
            facecolor="black"
        )

        plt.close(fig_png)
        print("Zapisano PNG:", PNG_CACHE)
    else:
        print("PNG już istnieje — pomijam tworzenie")

    existing_frames = list(OUT_DIR.glob("frame_*.png"))

    if len(existing_frames) >= FRAMES:
        print("Klatki renderu już istnieją — pomijam rendering i przechodzę do ffmpeg")
    else:
        print("Brak kompletu klatek — renderowanie")

    render_tasks = range(FRAMES)
    if len(existing_frames) < FRAMES:
        render_tasks = tqdm(render_tasks, total=FRAMES, desc="Renderowanie klatek", unit="klatka")

    for frame in render_tasks:
        if len(existing_frames) >= FRAMES:
            break

        t = times[frame]

        fig = plt.figure(
            figsize=(16, 9),
            dpi=150,
            facecolor="black"
        )

        bg_ax = fig.add_axes([0, 0, 1, 1], zorder=-10)
        bg_ax.imshow(
            bg,
            origin="upper"
        )
        bg_ax.axis("off")

        ax = fig.add_subplot(
            111,
            projection="hammer"
        )

        ax.set_facecolor(
            "none"
        )
        ax.patch.set_alpha(0.0)

        ax.axis("off")

        amp = amplification(
            t,
            tmax,
            te,
            u0
        )

        visible = np.abs(t - tmax) < te * 2.5

        if np.any(visible):

            strength = amp[visible] - 1
            strength = np.clip(
                strength,
                0,
                None
            )

            size = (
                10
                +
                150 * np.log1p(
                    strength
                )
            )

            # masa wpływa na wielkość błysku
            size *= np.clip(
                mass[visible],
                0.3,
                5
            )

            alpha = np.clip(
                np.log1p(strength) / np.log(10),
                0.05,
                1
            )

            ax.scatter(
                l[visible],
                b[visible],
                s=size,
                c="red",
                alpha=alpha,
                edgecolors="none"
            )

        fig.savefig(
            OUT_DIR /
            f"frame_{frame:04d}.png",
            facecolor="black"
        )

        plt.close(fig)


    print("Kodowanie mp4")

    if Path(VIDEO).exists():
        print("MP4 już istnieje — pomijam składanie")
        return

    subprocess.run([
        "ffmpeg",
        "-y",
        "-framerate",
        str(FPS),
        "-i",
        str(OUT_DIR/"frame_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        VIDEO
    ])

    print(
        "Gotowe:",
        VIDEO
    )



if __name__=="__main__":
    main()