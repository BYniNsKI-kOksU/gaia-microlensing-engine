from astroquery.gaia import Gaia
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ─────────────────────────────────────────────────────────────
#  POBIERANIE DANYCH
#  Podział po MOD(random_index, 10) zamiast po zakresach l.
#  Każda z 10 grup ma jednolity rozkład przestrzenny →
#  żadnych granic gęstości na mapie.
# ─────────────────────────────────────────────────────────────
print("Pobieranie danych Gaia DR3 – 10 losowych grup...")

tables = []

for i in range(10):
    query = f"""
    SELECT TOP 3000000
        l,
        b
    FROM gaiadr3.gaia_source
    WHERE
        MOD(random_index, 10) = {i}
        AND l IS NOT NULL
        AND b IS NOT NULL
    """

    print(f"  Grupa {i + 1}/10  (random_index mod 10 = {i})")
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

l = np.radians(sky["l"].values)
b = np.radians(sky["b"].values)

# Odbicie osi l – centrum Galaktyki pośrodku, wzrost l w lewo (standard astronomiczny)
l = -l
l[l < -np.pi] += 2 * np.pi
l[l > np.pi] -= 2 * np.pi

# ─────────────────────────────────────────────────────────────
#  BINOWANIE  8K
# ─────────────────────────────────────────────────────────────
bins_l = 7680
bins_b = 4320

hist, lon_edges, lat_edges = np.histogram2d(l, b, bins=[bins_l, bins_b])

# Lekkie wygładzenie (sigma=2 piksele)
hist = gaussian_filter(hist.T, sigma=2)

# Skala logarytmiczna gęstości
hist = np.log10(hist + 1)

lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
LON, LAT = np.meshgrid(lon_centers, lat_centers)

# ─────────────────────────────────────────────────────────────
#  PALETA – przybliżenie światła widzialnego
#  Gwiazdy Drogi Mlecznej: czarne tło → ciemny fiolet →
#  niebieskawa szarość → ciepła biel (centrum galaktyki)
# ─────────────────────────────────────────────────────────────
_colors = [
    (0.00, (0.000, 0.000, 0.000)),   # czerń (tło)
    (0.18, (0.030, 0.025, 0.065)),   # głęboki granat
    (0.38, (0.110, 0.090, 0.190)),   # ciemny fiolet
    (0.55, (0.340, 0.290, 0.430)),   # fiolet/szary
    (0.70, (0.600, 0.560, 0.660)),   # jasna lawa / srebrnoszary
    (0.84, (0.860, 0.840, 0.800)),   # ciepłobiały
    (1.00, (1.000, 0.985, 0.940)),   # prawie biały z ciepłym odcieniem
]

cmap_mw = LinearSegmentedColormap.from_list(
    "milky_way_visible",
    [(v, c) for v, c in _colors],
    N=512
)

# ─────────────────────────────────────────────────────────────
#  WYKRES HAMMER  – 16:9,  300 dpi  →  ~4800 × 2700 px
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
ax = fig.add_subplot(111, projection="hammer", facecolor="black")

ax.pcolormesh(LON, LAT, hist, shading="auto", cmap=cmap_mw)

# Delikatna siatka
ax.grid(True, linewidth=0.25, alpha=0.20, color="white", linestyle="--")

# ── Szerokość galaktyczna b – etykiety po lewej, pionowo ─────
b_tick_vals_deg = np.array([-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75])
ax.set_yticks(np.radians(b_tick_vals_deg))
ax.set_yticklabels(
    [f"{v}°" for v in b_tick_vals_deg],
    fontsize=6.5,
    color="white",
    alpha=0.75,
)

# ── Długość galaktyczna l – USUŃ etykiety z mapy ─────────────
ax.set_xticklabels([])

# ── Adnotacja osi l pod mapą ──────────────────────────────────
# Generujemy własne oznaczenia stopniowe wzdłuż dolnej osi
l_ticks_deg = np.arange(-150, 180, 30)   # -150 … +150 co 30°
for deg in l_ticks_deg:
    rad = np.radians(-deg)               # odwrócone (jak na mapie)
    # projekcja punktu na krawędzi b≈ -π/2 jest poza elipsą,
    # więc użyjemy annotate z transform=ax.transData
    label = f"{abs(deg)}°"
    if deg == 0:
        label = "0°"
    # przybliżona pozycja pod mapą wzdłuż dolnego łuku
    ax.annotate(
        label,
        xy=(rad, -1.58),          # poniżej dolnej krawędzi elipsy
        xycoords="data",
        ha="center",
        va="top",
        fontsize=6,
        color="white",
        alpha=0.65,
        annotation_clip=False,
    )

# Opis osi
ax.annotate(
    "Długość galaktyczna  l",
    xy=(0.5, -0.07),
    xycoords="axes fraction",
    ha="center",
    va="top",
    fontsize=9,
    color="white",
    annotation_clip=False,
)

ax.annotate(
    "Szerokość galaktyczna  b",
    xy=(-0.055, 0.5),
    xycoords="axes fraction",
    va="center",
    rotation=90,
    fontsize=9,
    color="white",
    annotation_clip=False,
)

# Ramka elipsy – biała, cienka
for spine in ax.spines.values():
    spine.set_edgecolor("white")
    spine.set_linewidth(0.4)

plt.tight_layout(pad=0.5)

out = "gaia_allsky_mollweide_8k.png"
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="black")
print(f"\nZapisano: {out}")
plt.show()