from astroquery.gaia import Gaia
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
import time

program_start = time.perf_counter()

# ─────────────────────────────────────────────────────────────
#  WCZYTYWANIE GOTOWEGO KATALOGU GAIA
# ─────────────────────────────────────────────────────────────
from astropy.table import Table

INPUT_FILE = "gaia_60m_allsky.fits"

print(f"Wczytywanie katalogu Gaia: {INPUT_FILE}")

table = Table.read(INPUT_FILE)


print(f"Wczytano: {len(table):,} rekordów")

# ─────────────────────────────────────────────────────────────
#  WCZYTYWANIE ZDARZEŃ MIKROSOCZEWKOWANIA GAIA
# ─────────────────────────────────────────────────────────────
MICROLENS_FILE = "gaia_microlensing.fits"

print(f"Wczytywanie mikrosoczewkowania: {MICROLENS_FILE}")

microlensing = Table.read(MICROLENS_FILE).to_pandas()
microlensing = microlensing.dropna(subset=["l", "b"])

print(f"Zdarzenia mikrosoczewkowania: {len(microlensing):,}")

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

plot_progress = tqdm(total=5, desc="Tworzenie mapy 2D", unit="etap")

bins_l = 15360
bins_b = 8640

hist, lon_edges, lat_edges = np.histogram2d(
    l,b,
    bins=[bins_l, bins_b],
    range=[[-np.pi, np.pi], [-np.pi / 2, np.pi / 2]]   # ← jawny zakres
)

# Wygładzenie
hist = gaussian_filter(hist.T, sigma=2)

# Skala logarytmiczna gęstości
hist = np.log10(hist + 1) 

# Normalizacja względem słabszych struktur, aby centrum nie dominowało
p_low = np.percentile(hist, 5)
p_high = np.percentile(hist, 98)   # 2% najjaśniejszych pikseli zostaje przyciętych

hist = (hist - p_low) / (p_high - p_low)
hist = np.clip(hist, 0, 1) 

# Mocniejsza kompresja jasnego dysku i centrum,
# # zachowanie widoczności halo oraz zewnętrznych struktur
hist = hist ** 0.55 

lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
LON, LAT = np.meshgrid(lon_centers, lat_centers)

# ────────────────────────────────────────────────────────────
# PALETA – fioletowo-złota, styl klasycznych wizualizacji astronomicznych
# Przejście: głęboka czerń → ciemny fiolet → jasny fiolet/lila
#            → ciepłe złoto → jasna biel dla centrum galaktyki
# ─────────────────────────────────────────────────────────────
_colors = [
    (0.000, (0.000, 0.000, 0.000)),   # czysta czerń – tło przestrzeni
    (0.120, (0.020, 0.008, 0.055)),   # bardzo ciemny, granatowo-fioletowy
    (0.260, (0.100, 0.040, 0.220)),   # ciemny fiolet – halo i słabe struktury
    (0.420, (0.320, 0.080, 0.480)),   # nasycony fiolet – zewnętrzny dysk
    (0.570, (0.620, 0.200, 0.520)),   # ciepły purpurowo-malinowy – wewnętrzny dysk
    (0.700, (0.860, 0.480, 0.180)),   # złoto-pomarańczowy – ramiona spiralne
    (0.840, (0.970, 0.790, 0.200)),   # jasne złoto – pobliże centrum
    (0.930, (1.000, 0.960, 0.650)),   # kremowo-złoty – jądro
    (1.000, (1.000, 1.000, 1.000)),   # czysta biel – centrum galaktyki
]

cmap_mw = LinearSegmentedColormap.from_list(
    "milky_way_purple_gold",
    [(v, c) for v, c in _colors],
    N=1024
)

# ─────────────────────────────────────────────────────────────
#  WYKRES HAMMER  16:9  300 dpi
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "black",
    "axes.facecolor":   "black",
    "text.color":       "white",
    "axes.labelcolor":  "white",
    "xtick.color":      "white",
    "ytick.color":      "white",
})

fig = plt.figure(figsize=(53.333, 30), dpi=300, facecolor="black")
ax  = fig.add_subplot(111, projection="hammer", facecolor="black")


ax.pcolormesh(LON, LAT, hist, shading="auto", cmap=cmap_mw)
plot_progress.update(1)
plot_progress.close()

# ─────────────────────────────────────────────────────────────
#  PUNKTY MIKROSOCZEWKOWANIA GAIA
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
#  ANIMACJA MIKROSOCZEWKOWANIA GAIA
# ─────────────────────────────────────────────────────────────

ml_l = np.radians(microlensing["l"].values)
ml_b = np.radians(microlensing["b"].values)

# identyczne odbicie jak dla mapy gwiazd
ml_l = -np.where(ml_l > np.pi, ml_l - 2*np.pi, ml_l)

ml_tmax = microlensing["paczynski0_tmax"].values
ml_te = microlensing["paczynski0_te"].values
ml_u0 = microlensing["paczynski0_u0"].values

valid = np.isfinite(ml_tmax) & np.isfinite(ml_te) & np.isfinite(ml_u0) & (ml_te > 0)
ml_l = ml_l[valid]
ml_b = ml_b[valid]
ml_tmax = ml_tmax[valid]
ml_te = ml_te[valid]
ml_u0 = ml_u0[valid]

te_med = np.nanmedian(ml_te)
tmax_lo = np.nanpercentile(ml_tmax, 2)
tmax_hi = np.nanpercentile(ml_tmax, 98)

core = (ml_tmax >= tmax_lo) & (ml_tmax <= tmax_hi)
ml_l = ml_l[core]
ml_b = ml_b[core]
ml_tmax = ml_tmax[core]
ml_te = ml_te[core]
ml_u0 = ml_u0[core]

ml_te_vis = ml_te * 3.0

frames = 750
animation_duration = 30.0

anim_start = tmax_lo - 3 * te_med
anim_end = tmax_hi + 3 * te_med

current_times = np.linspace(
    anim_start,
    anim_end,
    frames
)

microlens_points = ax.scatter(
    [],
    [],
    s=[],
    c=[],
    cmap="cool",
    alpha=0.9,
    edgecolors="white",
    linewidths=0.25
)

def paczynski_amplification(t, t0, te, u0):
    u = np.sqrt(
        u0**2 + ((t - t0) / te)**2
    )

    return (u*u + 2) / (u * np.sqrt(u*u + 4))


def update(frame):

    # aktualny moment w czasie Gaia (MJD)
    t = current_times[frame]

    visible = np.abs(t - ml_tmax) < 3 * ml_te_vis

    if np.any(visible):
        amp = paczynski_amplification(
            t,
            ml_tmax[visible],
            ml_te_vis[visible],
            ml_u0[visible]
        )

        sizes = (amp - 1) * 500
        colors = amp

        microlens_points.set_offsets(
            np.column_stack((
                ml_l[visible],
                ml_b[visible]
            ))
        )

        microlens_points.set_sizes(sizes)
        microlens_points.set_array(colors)

    else:
        microlens_points.set_offsets(
            np.empty((0,2))
        )
        microlens_points.set_sizes([])

    return microlens_points,


def init():
    microlens_points.set_offsets(np.empty((0, 2)))
    microlens_points.set_sizes([])
    return microlens_points,

ani = None

render_animation = input("Czy wyrenderować animację mikrosoczewkowania? (t/n): ").strip().lower()

if render_animation in ("t", "tak", "y", "yes"):
    ani = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=frames,
        interval=40,
        blit=True
    )

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

out = "gaia_allsky_hammer_16k.png"
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="black")

animation_out = "gaia_microlensing_animation.mp4"

class ProgressWriter:
    def __init__(self, writer):
        self.writer = writer
        self.progress = None

    def setup(self, fig, outfile, dpi=None):
        return self.writer.setup(fig, outfile, dpi)

    def grab_frame(self, **savefig_kwargs):
        if self.progress:
            self.progress.update(1)
        return self.writer.grab_frame(**savefig_kwargs)

    def finish(self):
        return self.writer.finish()

    def _supports_transparency(self):
        return self.writer._supports_transparency()

    def saving(self, fig, outfile, dpi):
        self.writer.setup(fig, outfile, dpi)
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.finish()


writer = ProgressWriter(
    FFMpegWriter(fps=25)
)

writer.progress = tqdm(
    total=frames,
    desc="Renderowanie animacji",
    unit="klatka"
)

if ani is not None:
    ani.save(
        animation_out,
        writer=writer,
        dpi=200
    )
    writer.progress.close()
    print(f"Zapisano animację: {animation_out}")
else:
    print("Pominięto renderowanie animacji")

plt.close(fig)

print(f"\nZapisano: {out}")

program_end = time.perf_counter()
elapsed = program_end - program_start
hours = int(elapsed // 3600)
minutes = int((elapsed % 3600) // 60)
seconds = elapsed % 60
print(f"Całkowity czas działania programu: {hours:02d}:{minutes:02d}:{seconds:05.2f}")