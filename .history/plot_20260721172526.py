"""
plot_microlensing_hammer.py

Tworzy mapę allsky w projekcji Hammer
i nanosi zdarzenia mikrosoczewkowania Gaia.

Wymaga:
pip install astropy matplotlib pandas
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table


# -----------------------------
# KONFIGURACJA
# -----------------------------

MICROLENS_FILE = "gaia_microlensing.fits"

OUTPUT = "microlensing_hammer.png"

DPI = 300


# -----------------------------
# WCZYTANIE DANYCH
# -----------------------------

print("Wczytywanie danych...")

data = Table.read(MICROLENS_FILE).to_pandas()

print(f"Liczba zdarzeń: {len(data)}")


# usunięcie braków
data = data.dropna(subset=["l", "b"])


l = np.array(data["l"])
b = np.array(data["b"])


# -----------------------------
# KONWERSJA DO HAMMER
# -----------------------------

# galaktyczna długość:
# matplotlib Hammer oczekuje:
# -pi ... pi

l_rad = np.radians(l)

# przesunięcie 180 stopni,
# aby centrum Galaktyki było na środku
l_rad = np.where(
    l_rad > np.pi,
    l_rad - 2*np.pi,
    l_rad
)

# odwrócenie kierunku osi
l_rad = -l_rad


b_rad = np.radians(b)


print("Zakres:")
print("l:", l_rad.min(), l_rad.max())
print("b:", b_rad.min(), b_rad.max())


# -----------------------------
# PLOT
# -----------------------------

plt.figure(
    figsize=(16,9),
    dpi=DPI,
    facecolor="black"
)

ax = plt.subplot(
    111,
    projection="hammer",
    facecolor="black"
)


# gwiazdy / zdarzenia
ax.scatter(
    l_rad,
    b_rad,
    s=8,
    c="cyan",
    alpha=0.7,
    edgecolors="none"
)


# linia płaszczyzny Drogi Mlecznej
lon = np.linspace(-np.pi, np.pi, 1000)
lat = np.zeros_like(lon)

ax.plot(
    lon,
    lat,
    color="white",
    linewidth=0.5,
    alpha=0.4
)


# ustawienia wyglądu

ax.grid(
    color="gray",
    alpha=0.25
)

ax.set_title(
    "Gaia DR3 — zdarzenia mikrosoczewkowania\nprojekcja Hammer allsky",
    color="white",
    fontsize=16
)


# białe napisy osi
ax.tick_params(
    colors="white"
)


plt.tight_layout()


plt.savefig(
    OUTPUT,
    dpi=DPI,
    facecolor="black"
)

print(f"Zapisano: {OUTPUT}")

plt.show()