"""
allsky.py — generowanie statycznej mapy nieba (projekcja Hammer, 16K) z katalogu Gaia.
Mapa tła jest cache'owana do PNG + layout .npz.
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
from matplotlib.path import Path as MplPath
from matplotlib.collections import PathCollection
from tqdm import tqdm
from astropy.table import Table

# ─────────────────────────────────────────────────────────────
#  KONFIGURACJA
# ─────────────────────────────────────────────────────────────
INPUT_FILE = "gaia_60m_allsky.fits"
MAP_FILE = Path("gaia_allsky_hammer_16k.png")
MAP_LAYOUT_FILE = Path("gaia_allsky_hammer_16k_layout.npz")

BINS_L = 16384
BINS_B = 8192
FIG_W_IN, FIG_H_IN = 53.333, 30.0


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


def _sparkle_path(spike_ratio, minor_ratio, valley_r=1.0, n_arms=4):
    """Znormalizowany kształt "błysku-gwiazdy" z promieniami dyfrakcyjnymi."""
    n = n_arms * 4
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    radii = np.empty(n)
    radii[0::4] = spike_ratio
    radii[1::4] = valley_r
    radii[2::4] = minor_ratio
    radii[3::4] = valley_r
    verts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    verts = np.vstack([verts, verts[0]])
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
    return MplPath(verts, codes)


# Kształt gwiazdy jest identyczny dla wszystkich zdarzeń i klatek —
# liczony raz przy imporcie modułu (a nie w każdej klatce jak w oryginale).
_DEFAULT_STAR_PATH = _sparkle_path(3.0, 1.4)




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

    ax.grid(True, linewidth=0.25, alpha=0.20, color="white", linestyle="--")

    b_ticks = np.array([-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75])
    ax.set_yticks(np.radians(b_ticks))
    ax.set_yticklabels([f"{v}°" for v in b_ticks], fontsize=6.5, color="white", alpha=0.75)
    ax.set_xticklabels([])

    for deg in range(-150, 180, 30):
        rad = np.radians(deg)
        label = "0°" if deg == 0 else f"{abs(deg)}°"
        ax.annotate(
            label, xy=(rad, -1.62), xycoords="data",
            ha="center", va="top", fontsize=6, color="white", alpha=0.65,
            annotation_clip=False,
        )

    ax.annotate(
        "Długość galaktyczna  l", xy=(0.5, -0.07), xycoords="axes fraction",
        ha="center", va="top", fontsize=9, color="white", annotation_clip=False,
    )
    ax.annotate(
        "Szerokość galaktyczna  b", xy=(-0.055, 0.5), xycoords="axes fraction",
        va="center", rotation=90, fontsize=9, color="white", annotation_clip=False,
    )

    for spine in ax.spines.values():
        spine.set_edgecolor("white")
        spine.set_linewidth(0.4)

    plt.tight_layout(pad=0.5)

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
    print(f"Statyczna mapa nieba została zapisana do pliku: {MAP_FILE}")


if __name__ == "__main__":
    main()