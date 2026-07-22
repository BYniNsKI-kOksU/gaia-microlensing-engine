"""
allsky.py — generowanie statycznej mapy nieba allsky.

Aby przeliczyć mapę od nowa, usuń gaia_allsky_hammer_16k.png i .npz.
"""

import matplotlib
# WYMUSZENIE backendu Agg (czysto rastrowy, bez GUI) — MUSI być wykonane
# przed `import matplotlib.pyplot`. Bez tego matplotlib na macOS potrafi
# automatycznie wybrać backend GUI (np. MacOSX), który przy przeskalowanym
# ekranie Retina/5K mnoży żądane dpi przez współczynnik skalowania ekranu.
matplotlib.use("Agg")

import os
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap
from astropy.table import Table

# ─────────────────────────────────────────────────────────────
#  KONFIGURACJA
# ─────────────────────────────────────────────────────────────
INPUT_FILE = "gaia_60m_allsky.fits"
MAP_FILE = Path("gaia_allsky_hammer_16k.png")
MAP_LAYOUT_FILE = Path("gaia_allsky_hammer_16k_layout.npz")
MAP_HIST_FILE = Path("gaia_allsky_hist_16k.npz")

BINS_L = 16384
BINS_B = 8192

FIG_W_IN, FIG_H_IN = 54.613, 30.72


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
    if MAP_HIST_FILE.exists():
        print(f"Wczytywanie gotowego histogramu mapy: {MAP_HIST_FILE}")
        with np.load(MAP_HIST_FILE) as cache:
            hist = cache["hist"]
            lon_edges = cache["lon_edges"]
            lat_edges = cache["lat_edges"]

        lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
        lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
        LON, LAT = np.meshgrid(lon_centers, lat_centers)

        ax.pcolormesh(LON, LAT, hist, shading="auto", cmap=cmap_mw)
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
        plt.savefig(str(MAP_FILE), dpi=100, facecolor="black")
        print(f"Zapisano mapę: {MAP_FILE}")
        return

    print(f"Wczytywanie katalogu Gaia: {INPUT_FILE}")
    table = Table.read(INPUT_FILE)
    print(f"Wczytano: {len(table):,} rekordów")

    mask = np.isfinite(table["l"]) & np.isfinite(table["b"])

    l_deg = np.asarray(table["l"][mask], dtype=np.float32)
    b_deg = np.asarray(table["b"][mask], dtype=np.float32)

    print(f"Po filtracji:    {len(l_deg):,} gwiazd")

    l_deg_centered = np.where(l_deg > 180, l_deg - 360, l_deg)
    l = np.radians(l_deg_centered)
    b = np.radians(b_deg)
    l = -l  # centrum galaktyki (l=0) w środku, l=90 w prawo

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

    np.savez_compressed(
        MAP_HIST_FILE,
        hist=hist,
        lon_edges=lon_edges,
        lat_edges=lat_edges,
    )
    print(f"Zapisano histogram mapy: {MAP_HIST_FILE}")

    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    LON, LAT = np.meshgrid(lon_centers, lat_centers)

    ax.pcolormesh(LON, LAT, hist, shading="auto", cmap=cmap_mw)

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
    plt.savefig(str(MAP_FILE), dpi=100, facecolor="black")
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
#  GŁÓWNY PRZEBIEG PROGRAMU
# ─────────────────────────────────────────────────────────────
def main():
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
        plt.savefig(str(MAP_FILE), dpi=300, facecolor="black")
        plt.close(fig)
        return

    plt.savefig(str(MAP_FILE), dpi=300, facecolor="black")
    plt.close(fig)


if __name__ == "__main__":
    main()
