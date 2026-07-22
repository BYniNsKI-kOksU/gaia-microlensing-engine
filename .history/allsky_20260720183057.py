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
bins_b = 864

hist, lon_edges, lat_edges = np.histogram2d(
    l,b,
    bins=[bins_l, bins_b],
    range=[[-np.pi, np.pi], [-np.pi / 2, np.pi / 2]]   # ← jawny zakres
)

# Wygładzenie
hist = gaussian_filter(hist.T, sigma=1.8)

# Miękka kompresja jasności:
# logarytm + percentyle + asinh daje lepszy kompromis między jądrem
# a słabszym dyskiem niż twarde przycinanie lub pojedyncza potęga.
hist = np.log1p(hist)

p_low = np.percentile(hist, 1.0)
p_high = np.percentile(hist, 99.85)

hist = (hist - p_low) / (p_high - p_low)
hist = np.clip(hist, 0, 1)
hist = np.arcsinh(3.8 * hist) / np.arcsinh(3.8)
hist = hist ** 1.12
hist = np.clip(hist, 0, 1)

lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
LON, LAT = np.meshgrid(lon_centers, lat_centers)

# ────────────────────────────────────────────────────────────
# PALETA inspirowana zdjęciami Drogi Mlecznej:
# tło: głęboka czerń / granat
# słabe struktury: chłodne niebiesko-szare
# dysk: przygaszone kremy
# centrum: ciepłe żółcie i pomarańcze bez czystej bieli
# ─────────────────────────────────────────────────────────────
_colors = [
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

cmap_mw = LinearSegmentedColormap.from_list(
    "milky_way_realistic",
    [(v, c) for v, c in _colors],
    N=2048
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

# fig = plt.figure(figsize=(53.333, 30), dpi=300, facecolor="black")
# bins_l = 15360
# bins_b = 8640
# 16k 
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
ml_mass = None
for mass_col in (
    "paczynski0_mass",
    "paczynski_mass",
    "lens_mass",
    "mass",
):
    if mass_col in microlensing.columns:
        ml_mass = microlensing[mass_col].values
        break

valid = np.isfinite(ml_tmax) & np.isfinite(ml_te) & np.isfinite(ml_u0) & (ml_te > 0)
if ml_mass is not None:
    valid &= np.isfinite(ml_mass) & (ml_mass > 0)
ml_l = ml_l[valid]
ml_b = ml_b[valid]
ml_tmax = ml_tmax[valid]
ml_te = ml_te[valid]
ml_u0 = ml_u0[valid]
if ml_mass is not None:
    ml_mass = ml_mass[valid]

te_med = np.nanmedian(ml_te)
tmax_lo = np.nanpercentile(ml_tmax, 2)
tmax_hi = np.nanpercentile(ml_tmax, 98)

core = (ml_tmax >= tmax_lo) & (ml_tmax <= tmax_hi)
ml_l = ml_l[core]
ml_b = ml_b[core]
ml_tmax = ml_tmax[core]
ml_te = ml_te[core]
ml_u0 = ml_u0[core]
if ml_mass is not None:
    ml_mass = ml_mass[core]

te_lo = np.nanpercentile(ml_te, 5)
te_hi = np.nanpercentile(ml_te, 95)
te_log_span = max(np.log10(te_hi) - np.log10(te_lo), 1e-6)
ml_te_norm = np.clip((np.log10(ml_te) - np.log10(te_lo)) / te_log_span, 0, 1)

if ml_mass is not None:
    mass_lo = np.nanpercentile(ml_mass, 5)
    mass_hi = np.nanpercentile(ml_mass, 95)
    mass_log_span = max(np.log10(mass_hi) - np.log10(mass_lo), 1e-6)
    ml_mass_norm = np.clip((np.log10(ml_mass) - np.log10(mass_lo)) / mass_log_span, 0, 1)
else:
    ml_mass_norm = np.zeros_like(ml_te_norm)

ml_order = np.argsort(ml_tmax)
ml_activation = np.empty_like(ml_te_norm)
if len(ml_activation) > 1:
    ml_activation[ml_order] = np.linspace(0.0, 1.0, len(ml_activation), endpoint=True)
else:
    ml_activation[:] = 1.0

frames = 750
animation_duration = 30.0

event_window = 2.45 * ml_te
time_margin = 0.08 * max(tmax_hi - tmax_lo, 1.0)
anim_start = float(np.nanmin(ml_tmax - event_window) - time_margin)
anim_end = float(np.nanmax(ml_tmax + event_window) + 1.5 * time_margin)

current_times = np.linspace(
    anim_start,
    anim_end,
    frames
)

microlens_outer = ax.scatter(
    [],
    [],
    s=[],
    c=[],
    alpha=0.0,
    edgecolors="none",
    zorder=4,
)

microlens_inner = ax.scatter(
    [],
    [],
    s=[],
    c=[],
    alpha=0.0,
    edgecolors="none",
    zorder=5,
)

microlens_core = ax.scatter(
    [],
    [],
    s=[],
    c=[],
    alpha=0.0,
    edgecolors="white",
    linewidths=0.28,
    zorder=6,
)

def paczynski_amplification(t, t0, te, u0):
    u = np.sqrt(
        u0**2 + ((t - t0) / te)**2
    )

    return (u*u + 2) / (u * np.sqrt(u*u + 4))


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


def _event_layers(amp, te_norm, mass_norm):
    contrast = np.clip(amp - 1.0, 0.0, None)
    amp_boost = np.log1p(contrast * 9.0) / np.log1p(9.0)
    structure = 0.68 * te_norm + 0.32 * mass_norm

    core_size = np.clip(4.0 + 8.0 * amp_boost + 4.5 * structure, 3.5, 18.0)
    inner_size = np.clip(12.0 + 24.0 * amp_boost + 14.0 * structure, 10.0, 60.0)
    outer_size = np.clip(26.0 + 34.0 * amp_boost + 24.0 * structure, 18.0, 120.0)

    weak_rgb = np.array([0.42, 0.91, 1.00], dtype=float)
    strong_rgb = np.array([0.97, 0.99, 1.00], dtype=float)
    blend = np.clip(0.18 + 0.82 * amp_boost, 0.0, 1.0)
    rgb = weak_rgb[None, :] * (1.0 - blend[:, None]) + strong_rgb[None, :] * blend[:, None]

    outer_alpha = np.clip(0.03 + 0.16 * amp_boost, 0.0, 0.28)
    inner_alpha = np.clip(0.10 + 0.34 * amp_boost, 0.0, 0.48)
    core_alpha = np.clip(0.45 + 0.50 * amp_boost, 0.0, 0.96)

    return {
        "rgb": rgb,
        "outer_size": outer_size,
        "inner_size": inner_size,
        "core_size": core_size,
        "outer_alpha": outer_alpha,
        "inner_alpha": inner_alpha,
        "core_alpha": core_alpha,
    }


def update(frame):

    # aktualny moment w czasie Gaia (MJD)
    t = current_times[frame]

    reveal = _smoothstep(0.08, 0.72, frame / max(frames - 1, 1))
    visible = np.abs(t - ml_tmax) <= event_window
    visible &= ml_activation <= reveal

    if np.any(visible):
        amp = paczynski_amplification(
            t,
            ml_tmax[visible],
            ml_te[visible],
            ml_u0[visible]
        )

        layer = _event_layers(
            amp,
            ml_te_norm[visible],
            ml_mass_norm[visible],
        )
        offsets = np.column_stack((
            ml_l[visible],
            ml_b[visible]
        ))

        microlens_outer.set_offsets(offsets)
        microlens_outer.set_sizes(layer["outer_size"])
        microlens_outer.set_facecolors(
            _rgba_stack([0.30, 0.88, 1.00], layer["outer_alpha"])
        )

        microlens_inner.set_offsets(offsets)
        microlens_inner.set_sizes(layer["inner_size"])
        microlens_inner.set_facecolors(
            _rgba_stack([0.70, 0.96, 1.00], layer["inner_alpha"])
        )

        microlens_core.set_offsets(offsets)
        microlens_core.set_sizes(layer["core_size"])
        microlens_core.set_facecolors(
            _rgba_stack(layer["rgb"], layer["core_alpha"])
        )

    else:
        empty = np.empty((0, 2))
        microlens_outer.set_offsets(empty)
        microlens_outer.set_sizes([])
        microlens_outer.set_facecolors(np.empty((0, 4)))

        microlens_inner.set_offsets(empty)
        microlens_inner.set_sizes([])
        microlens_inner.set_facecolors(np.empty((0, 4)))

        microlens_core.set_offsets(empty)
        microlens_core.set_sizes([])
        microlens_core.set_facecolors(np.empty((0, 4)))

    return microlens_outer, microlens_inner, microlens_core,


def init():
    empty = np.empty((0, 2))
    microlens_outer.set_offsets(empty)
    microlens_outer.set_sizes([])
    microlens_outer.set_facecolors(np.empty((0, 4)))

    microlens_inner.set_offsets(empty)
    microlens_inner.set_sizes([])
    microlens_inner.set_facecolors(np.empty((0, 4)))

    microlens_core.set_offsets(empty)
    microlens_core.set_sizes([])
    microlens_core.set_facecolors(np.empty((0, 4)))

    return microlens_outer, microlens_inner, microlens_core,

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

out = "gaia_allsky_hammer_8k.png"
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
