from astroquery.gaia import Gaia
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt

# Pobieranie danych Gaia DR3 w 10 sektorach...
print("Pobieranie danych Gaia DR3 w 10 sektorach...")

tables = []
sector_width = 360 / 10

for i in range(10):
    l_min = i * sector_width
    l_max = (i + 1) * sector_width

    query = f"""
    SELECT TOP 3000000
        l,
        b
    FROM gaiadr3.gaia_source
    WHERE
        l >= {l_min}
        AND l < {l_max}
        AND l IS NOT NULL
        AND b IS NOT NULL
    """

    print(f"Sektor {i+1}/10: l={l_min:.1f} - {l_max:.1f} deg")

    job = Gaia.launch_job_async(query)
    result = job.get_results()

    print(f"  Pobrano: {len(result):,} gwiazd")

    tables.append(result)

from astropy.table import vstack

table = vstack(tables)

print(f"Łącznie pobrano rekordów Gaia: {len(table):,}")

sky = table.to_pandas().dropna(subset=["l", "b"])

print(f"Po usunięciu braków współrzędnych: {len(sky):,} gwiazd")

l = np.radians(sky["l"].values)
b = np.radians(sky["b"].values)

# Lustrzane odbicie osi długości galaktycznej
# Centrum Galaktyki pozostaje w środku, a kierunek wzrostu l jest odwrócony
l = -l
l[l < -np.pi] += 2 * np.pi
l[l > np.pi] -= 2 * np.pi

# Binowanie gęstości gwiazd - większa rozdzielczość
bins_l = 7680
bins_b = 4320

hist, lon_edges, lat_edges = np.histogram2d(
    l,
    b,
    bins=[bins_l, bins_b]
)

# Wygładzenie mapy przed renderem 8K
hist = gaussian_filter(hist.T, sigma=2)

# Logarytmiczna skala gęstości gwiazd
hist = np.log10(hist + 1)

# Środki pikseli
lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2

LON, LAT = np.meshgrid(lon_centers, lat_centers)

# Wykres Mollweide
fig = plt.figure(figsize=(16, 9), dpi=300)
ax = fig.add_subplot(111, projection="mollweide")

im = ax.pcolormesh(
    LON,
    LAT,
    hist,
    shading="auto",
    cmap="inferno"
)

ax.grid(True, linewidth=0.3, alpha=0.4)

ax.tick_params(
    axis="both",
    which="major",
    labelsize=8,
    pad=10
)

ax.set_xlabel("")
ax.set_ylabel("")

# Etykiety współrzędnych poza obszarem mapy
ax.annotate(
    "Długość galaktyczna l [deg]",
    xy=(0.5, -0.08),
    xycoords="axes fraction",
    ha="center",
    fontsize=10
)

ax.annotate(
    "Szerokość galaktyczna b [deg]",
    xy=(-0.08, 0.5),
    xycoords="axes fraction",
    va="center",
    rotation=90,
    fontsize=10
)

plt.colorbar(im, ax=ax, label="log10 liczby gwiazd")

plt.title("Mapa allsky Drogi Mlecznej - Gaia DR3")
plt.tight_layout()

plt.savefig(
    "gaia_allsky_mollweide_8k.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()