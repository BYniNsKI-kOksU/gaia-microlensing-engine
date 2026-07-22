import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from pathlib import Path
import matplotlib.image as mpimg
import subprocess
import os


# -----------------------------
# KONFIGURACJA
# -----------------------------

MICROLENS_FILE = "gaia_microlensing.fits"

MAP_FILE = "microlensing_hammer.png"

OUT_DIR = Path("frames_micro")
VIDEO = "microlensing_animation.mp4"

FRAMES = 200
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


    return (
        l[good],
        b[good],
        tmax[good],
        te[good],
        u0[good],
        mass[good]
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


    start = np.min(
        tmax-3*te
    )

    end = np.max(
        tmax+3*te
    )


    times = np.linspace(
        start,
        end,
        FRAMES
    )


    bg = mpimg.imread(
        MAP_FILE
    )


    for frame,t in enumerate(times):

        fig = plt.figure(
            figsize=(16,9),
            dpi=150,
            facecolor="black"
        )


        # tło
        axbg = fig.add_axes(
            [0,0,1,1]
        )

        axbg.imshow(
            bg,
            origin="upper"
        )

        axbg.axis("off")


        ax = fig.add_subplot(
            111,
            projection="hammer"
        )


        ax.set_facecolor(
            "none"
        )

        ax.axis("off")


        amp = amplification(
            t,
            tmax,
            te,
            u0
        )


        visible = np.abs(
            t-tmax
        ) < 3*te


        if np.any(visible):

            strength = amp[visible]-1


            strength = np.clip(
                strength,
                0,
                None
            )


            size = (
                10
                +
                150*np.log1p(
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
                strength/5,
                0.15,
                1
            )


            ax.scatter(
                l[visible],
                b[visible],
                s=size,
                c="cyan",
                alpha=alpha,
                edgecolors="none"
            )


        fig.savefig(
            OUT_DIR /
            f"frame_{frame:04d}.png",
            facecolor="black"
        )


        plt.close(fig)


        print(
            frame+1,
            "/",
            FRAMES
        )



    print("Kodowanie mp4")


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