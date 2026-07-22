from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt

# Plik VOTable z Gaia Archive
# Wymagane kolumny: l, b
INPUT_FILE = "gaia-allsky.vot"

# Wczytanie danych Gaia VOTable
print("Wczytywanie danych...")
table = Table.read(INPUT_FILE, format="votable")

# Usunięcie braków
sky = table[["l", "b"]].to_pandas().dropna(subset=["l", "b"])

l = np.radians(sky["l"].values)
b = np.radians(sky["b"].values)

# Mollweide wymaga zakresu długości -pi do pi
l[l > np.pi] -= 2 * np.pi

# Binowanie gęstości gwiazd
bins_l = 720
bins_b = 360

hist, lon_edges, lat_edges = np.histogram2d(
    l,
    b,
    bins=[bins_l, bins_b]
)

# Logarytmiczna skala jasności mapy
hist = np.log10(hist.T + 1)

# Środki pikseli
lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2

LON, LAT = np.meshgrid(lon_centers, lat_centers)

# Wykres Mollweide
fig = plt.figure(figsize=(12, 6), dpi=300)
ax = fig.add_subplot(111, projection="mollweide")

im = ax.pcolormesh(
    LON,
    LAT,
    hist,
    shading="auto"
)

ax.grid(True)
ax.set_xlabel("Długość galaktyczna l")
ax.set_ylabel("Szerokość galaktyczna b")

plt.colorbar(im, ax=ax, label="log10 liczby gwiazd")

plt.title("Mapa allsky Drogi Mlecznej - Gaia DR3")
plt.tight_layout()

plt.savefig("gaia_allsky_mollweide.png", dpi=300)
plt.show()