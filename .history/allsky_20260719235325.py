from astroquery.gaia import Gaia
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ─────────────────────────────────────────────────────────────
#  POBIERANIE DANYCH
#
#  Podział po ZAKRESACH random_index (nie po l, nie po MOD).
#  random_index w Gaia DR3 jest losowo przypisany do każdej
#  gwiazdy w całym katalogu → każdy z 10 przedziałów wartości
#  daje przestrzennie jednorodną próbkę, zero artefaktów.
#
#  Gaia DR3: ~1 811 709 771 źródeł, random_index ∈ [0, N-1].
# ─────────────────────────────────────────────────────────────
print("Pobieranie danych Gaia DR3 – 10 przedziałów random_index...")

N_TOTAL = 1_811_709_771          # liczba źródeł DR3
CHUNK   = N_TOTAL // 10          # ~181 mln na przedział

tables = []

for i in range(10):
    ri_lo = i * CHUNK
    ri_hi = (i + 1) * CHUNK if i < 9 else N_TOTAL

    query = f"""
    SELECT TOP 3000000
        l,
        b
    FROM gaiadr3.gaia_source
    WHERE
        random_index >= {ri_lo}
        AND random_index < {ri_hi}
        AND l IS NOT NULL
        AND b IS NOT NULL
    """

    print(f"  Przedział {i + 1}/10  ri ∈ [{ri_lo:,}, {ri_hi:,})")
    job = Gaia.launch_job_async(query)
    result = job.get_results()
    print(f"    → {len(result):,} gwiazd")
    tables.append(result)

from astropy.table import vstack

table = vstack(tables)
print(f"\nŁącznie pobrano: {len(table):,} rekordów")

# ─────────────────────────────────────────────────────────────
#  PRZYGOTOWANIE DANYCH
# ─────────────────────────────────────────────────────────────
sky = table.to_pandas().dropna(subset=["l", "b"])
print(f"Po filtracji:    {len(sky):,} gwiazd")

l_deg = sky["l"].values   # [0, 360)
b_deg = sky["b"].values   # [-90, 90]

# Konwersja do radianów i przejście do układu [-π, π]
# Gaia l ∈ [0, 360) → przenosimy do [-180, 180) przed odbiciem
l_deg_centered = np.where(l_deg > 180, l_deg - 360, l_deg)  # [-180, 180)
l = np.radians(l_deg_centered)
b = np.radians(b_deg)

# Odbicie: wzrost l idzie w lewo (standard astronomiczny widoku z zewnątrz)
l = -l   # teraz centrum galaktyki (l=0) zostaje w środku, l=90 idzie w prawo

# ─────────────────────────────────────────────────────────────
#  BINOWANIE 8K  (pełny zakres [-π, π] × [-π/2, π/2])
# ─────────────────────────────────────────────────────────────
bins_l = 7680
bins_b = 4320

hist, lon_edges, lat_edges = np.histogram2d(
    l, b,
    bins=[bins_l, bins_b],
    range=[[-np.pi, np.pi], [-np.pi / 2, np.pi / 2]]   # ← jawny zakres
)

# Wygładzenie
hist = gaussian_filter(hist.T, sigma=2)

# Skala logarytmiczna gęstości
hist = np.log10(hist + 1)

lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
LON, LAT = np.meshgrid(lon_centers, lat_centers)

# ─────────────────────────────────────────────────────────────
#  PALETA – przybliżenie światła widzialnego
# ─────────────────────────────────────────────────────────────
_colors = [
    (0.00, (0.000, 0.000, 0.000)),
    (0.18, (0.030, 0.025, 0.065)),
    (0.38, (0.110, 0.090, 0.190)),
    (0.55, (0.340, 0.290, 0.430)),
    (0.70, (0.600, 0.560, 0.660)),
    (0.84, (0.860, 0.840, 0.800)),
    (1.00, (1.000, 0.985, 0.940)),
]

cmap_mw = LinearSegmentedColormap.from_list(
    "milky_way_visible",
    [(v, c) for v, c in _colors],
    N=512
)

# ─────────────────────────────────────────────────────────────
#  WYKRES MOLLWEIDE  16:9  300 dpi
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "black",
    "axes.facecolor":   "black",
    "text.color":       "white",
    "axes.labelcolor":  "white",
    "xtick.color":      "white",
    "ytick.color":      "white",
})

fig = plt.figure(figsize=(16, 9), dpi=300, facecolor="black")
ax  = fig.add_subplot(111, projection="mollweide", facecolor="black")

ax.pcolormesh(LON, LAT, hist, shading="auto", cmap=cmap_mw)

# Delikatna siatka
ax.grid(True, linewidth=0.25, alpha=0.20, color="white", linestyle="--")

# Szerokość galaktyczna b – etykiety po lewej
b_ticks = np.array([-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75])
ax.set_yticks(np.radians(b_ticks))
ax.set_yticklabels(
    [f"{v}°" for v in b_ticks],
    fontsize=6.5, color="white", alpha=0.75,
)

# Długość galaktyczna l – usuń wbudowane etykiety z mapy
ax.set_xticklabels([])

# Adnotacje l pod dolnym łukiem elipsy
for deg in range(-150, 180, 30):
    rad   = np.radians(deg)          # po odbiciu: -deg → ale oś już odwrócona w danych
    label = "0°" if deg == 0 else f"{abs(deg)}°"
    ax.annotate(
        label,
        xy=(rad, -1.62),
        xycoords="data",
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

out = "gaia_allsky_mollweide_8k.png"
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="black")
print(f"\nZapisano: {out}")
plt.show()