> **This README contains two language versions:**
> - 🇵🇱 Polish (first)
> - 🇬🇧 English (second)
>
> **Ten plik README zawiera dwie wersje językowe:**
> - 🇵🇱 Polska (pierwsza)
> - 🇬🇧 English (druga)

---

# 🌌 Silnik Generowania Mapy Całego Nieba Gaia i Animacji Mikrosoczewkowania

> **Polska wersja dokumentacji**

Zaawansowany pipeline przetwarzania danych astronomicznych, służący do pobierania danych, generowania wielkoformatowej mapy Drogi Mlecznej w projekcji Hammera (16K) na podstawie katalogu gwiazd **Gaia** oraz renderowania fotorealistycznych symulacji zjawisk **grawitacyjnego mikrosoczewkowania**.

---

## 📋 Spis Treści
1. [O Projekcie](#-o-projekcie)
2. [Główne Funkcje](#-główne-funkcje)
3. [Architektura i Struktura Kodu](#-architektura-i-struktura-kodu)
4. [Podstawy Fizyczne i Matematyczne](#-podstawy-fizyczne-i-matematyczne)
5. [Wymagania i Zależności](#-wymagania-i-zależności)
6. [Format Danych Wejściowych](#-format-danych-wejściowych)
7. [Instrukcja Użytkowania](#-instrukcja-użytkowania)
8. [Konfiguracja i Parametry](#-konfiguracja-i-parametry)
9. [Optymalizacja Wydajności](#-optymalizacja-wydajności)
10. [Licencja i Źródła Danych](#-licencja-i-źródła-danych)

---

## 🔭 O Projekcie

Projekt składa się z trzech zintegrowanych modułów Pythona:
1. **`download_gaia_catalog.py`**: Automatycznie wykonuje zapytanie ADQL do archiwum **Gaia DR3** przy użyciu `astroquery.gaia`. Pobiera dane o zjawiskach mikrosoczewkowania z tabeli `gaiadr3.vari_microlensing` połączonej z `gaiadr3.gaia_source` i zapisuje wyniki do pliku FITS.
2. **`gaia_allsky_map.py`**: Generuje statyczną mapę całego nieba o wysokiej rozdzielczości ($16384 \times 8192$ pikseli) w układzie Galaktycznym $(l, b)$ z użyciem projekcji Hammera. Oblicza całkowity strumień fotonów z obiektów katalogu Gaia i stosuje autorską 10-kolorową paletę, aby odtworzyć wygląd Drogi Mlecznej.
3. **`microlensing_animation.py`**: Odczytuje wygenerowaną mapę bazową i zjawiska mikrosoczewkowania, a następnie tworzy sekwencję animowanych klatek przedstawiających dynamiczne pojaśnienia grawitacyjnego mikrosoczewkowania wg modelu **Paczyńskiego**. Silnik wykorzystuje zoptymalizowane operacje na tablicach NumPy float32 oraz multiprocessing, omijając narzut rysowania obiektów Matplotlib.

---

## ✨ Główne Funkcje

* **Automatyczne pobieranie danych Gaia TAP/ADQL:** Pobiera zweryfikowane zjawiska mikrosoczewkowania bezpośrednio z serwerów ESA Gaia, filtrując paralaksę ($0.03 < \varpi < 21$) i czas Einsteina ($t_e$).
* **Mapa główna w rozdzielczości 16K:** Oblicza 2-wymiarowy histogram sferyczny na siatce $16384 \times 8192$ binów.
* **Fizyczny model jasności gwiazd:** Przekształca jasności w paśmie Gaia $G$ na fizyczny strumień fotonów ($F \propto 10^{-0.4 G}$) z obcięciem wartości ekstremalnych na poziomie 99.9 percentyla.
* **Zaawansowana rozciągliwość zakresu dynamicznego (HDR Tone Mapping):** Łączy filtrację Gaussa ($\sigma=1.8$), transformację logarytmiczną $\log(1+x)$ i odwrotną sinus hiperboliczną $\text{arcsinh}(x)$ w celu uwypuklenia pasm pyłowych i słabych struktur dysku.
* **Analityczna dynamika mikrosoczewkowania Paczyńskiego:** Dokładnie odtwarza teoretyczne krzywe blasku dla zjawisk soczewkowania grawitacyjnego.
* **Ultraszybki silnik renderujący bezpośrednio do tablicy:** Rasteryzuje profile Gaussa pojaśnień na 3-kanałowym płótnie RGB float32, eliminując narzut Matplotlib.
* **Równoległość multiprocessing:** Wydajne renderowanie klatek przy użyciu `multiprocessing.Pool` z izolowanym ładowaniem obrazu tła w każdym procesie, aby uniknąć transferu dużych danych Pickle.
* **Podwójny eksport wideo (16K/8K H.265 HEVC):** Klatki są kodowane przez `ffmpeg` do formatu H.265 z automatycznym tagowaniem `hvc1` dla zgodności Apple/QuickTime, z opcjonalnym przeskalowaniem do 8K dla urządzeń mobilnych.

---

## 🏗 Architektura i Struktura Kodu

Projekt jest podzielony na trzy główne moduły odpowiedzialne za osobne etapy przetwarzania danych Gaia DR3:

```text
.
├── download_gaia_catalog.py          # Pobieranie katalogu gwiazd Gaia DR3 lub danych wymaganych przez pipeline
├── download_gaia_microlensing.py     # Pobieranie katalogu zdarzeń mikrosoczewkowania Gaia DR3
├── gaia_allsky_map.py                # Generowanie mapy całego nieba Drogi Mlecznej w projekcji Hammera
├── microlensing_animation.py         # Renderowanie animacji mikrosoczewkowania i kodowanie wideo
├── README.md                         # Dokumentacja projektu
├── LICENSE                           # Licencja MIT
├── requirements.txt                  # Lista wymaganych bibliotek Python
├── gaia_150m_allsky.fits             # Lokalny katalog gwiazd Gaia (dane użytkownika)
├── gaia_microlensing.fits            # Katalog zdarzeń mikrosoczewkowania Gaia
├── gaia_allsky_hammer_16k.png        # Wygenerowana mapa bazowa całego nieba
├── gaia_allsky_hammer_16k_layout.npz # Dane pomocnicze mapy i układu współrzędnych
├── frames_micro/                     # Tymczasowe klatki animacji
└── microlensing_animation_*.mp4      # Wygenerowane pliki wideo
```

### Zakres odpowiedzialności modułów:

#### `download_gaia_catalog.py`
Odpowiada za pobieranie danych katalogowych Gaia DR3 wymaganych do stworzenia mapy gwiazd. Wykorzystuje zapytania ADQL przez `astroquery.gaia` i zapisuje wynik w formacie FITS.

#### `download_gaia_microlensing.py`
Pobiera z archiwum Gaia DR3 informacje o potwierdzonych zdarzeniach mikrosoczewkowania z tabeli `gaiadr3.vari_microlensing`, łącząc je z podstawowymi parametrami gwiazd.

#### `gaia_allsky_map.py`
Przetwarza katalog gwiazd Gaia, przelicza współrzędne galaktyczne oraz jasności gwiazd na mapę powierzchniową Drogi Mlecznej. Generuje wysokorozdzielczy obraz PNG używany jako tło animacji.

#### `microlensing_animation.py`
Wczytuje mapę bazową i katalog zdarzeń mikrosoczewkowania, oblicza przebieg jasności według modelu Paczyńskiego, renderuje błyski oraz tworzy końcowe pliki wideo przy pomocy FFmpeg.

---

## 📐 Podstawy Fizyczne i Matematyczne

### 1. Teoretyczna Krzywa Blasku Paczyńskiego
Wzmocnienie strumienia fotonów gwiazdy $A(t)$ w chwili $t$ opisuje wzór:

$$A(t) = \frac{u(t)^2 + 2}{u(t) \sqrt{u(t)^2 + 4}}$$

gdzie znormalizowane rozdzielenie kątowe $u(t)$ na płaszczyźnie soczewki:

$$u(t) = \sqrt{u_0^2 + \left(\frac{t - t_0}{t_e}\right)^2}$$

* $t_0$ – moment maksymalnego wyrównania (szczyt pojaśnienia),
* $t_e$ – czas Einsteina (czas przejścia przez promień Einsteina),
* $u_0$ – minimalny parametr zbliżenia w jednostkach promienia Einsteina.

### 2. Przekształcenie geometryczne projekcji Hammera
Transformacja współrzędnych sferycznych galaktycznych $(l, b)$ w radianach do płaskich współrzędnych projekcji Hammera $(x_H, y_H)$:

$$z = \sqrt{1 + \cos(b) \cdot \cos\left(\frac{l}{2}\right)}$$

$$x_H = \frac{2\sqrt{2} \cdot \cos(b) \cdot \sin\left(\frac{l}{2}\right)}{z}, \quad y_H = \frac{\sqrt{2} \cdot \sin(b)}{z}$$

Mapowanie ciągłych $x_H \in [-2\sqrt{2}, 2\sqrt{2}]$ oraz $y_H \in [-\sqrt{2}, \sqrt{2}]$ na dyskretną macierz pikseli $(W \times H)$:

$$p_x = \left(\frac{x_H}{2\sqrt{2}} + 1\right) \cdot \frac{W}{2}, \quad p_y = \left(1 - \frac{y_H}{\sqrt{2}}\right) \cdot \frac{H}{2}$$

### 3. Profil nasycenia i rasteryzacja błysków
Każde pojaśnienie mikrosoczewkowe reprezentowane jest przez 2D spłaszczony profil Gaussa. Promień dysku $\sigma_{\text{px}}$ i szczytowa intensywność $\alpha$ zależą od wzmocnienia $\Delta A = A(t) - 1$:

$$\sigma_{\text{px}} = \text{clip}\left( \sqrt{\frac{S_{\text{pt2}}}{\pi}} \cdot \frac{\text{DPI}}{72} \cdot \frac{1}{3} \cdot M \cdot \text{scale}, \, 2.0, \, 0.08 \cdot H \right)$$

gdzie $S_{\text{pt2}} = 10.0 + 10000.0 \cdot \ln(1 + \Delta A)$, a $M$ to masa soczewki.

---

## 📦 Wymagania i Zależności

Projekt wymaga **Pythona 3.9+** oraz następujących pakietów:

```bash
pip install numpy scipy matplotlib astropy pandas tqdm astroquery healpy opencv-python
```

Zewnętrzne narzędzie systemowe:
* **FFmpeg** (ze wsparciem `libx265` / H.265).

### Instalacja FFmpeg:
* **macOS:** `brew install ffmpeg`
* **Linux (Ubuntu/Debian):** `sudo apt update && sudo apt install ffmpeg`
* **Windows (Chocolatey):** `choco install ffmpeg`

---

## 📄 Format Danych Wejściowych

### 1. Katalog Gwiazd Gaia (`gaia_150m_allsky.fits`)
Plik FITS zawierający pozycje gwiazd oraz fotometrię. Wymagane kolumny:
* `l`: długość galaktyczna w stopniach $[0^\circ, 360^\circ]$.
* `b`: szerokość galaktyczna w stopniach $[-90^\circ, +90^\circ]$.
* `phot_g_mean_mag`: obserwowana jasność w paśmie Gaia $G$.

### 2. Katalog Zjawisk Mikrosoczewkowania (`gaia_microlensing.fits`)
Tabela FITS (generowana automatycznie przez `download_gaia_catalog.py` lub dostarczona ręcznie). Wymagane kolumny:
* `source_id`: identyfikator obiektu Gaia.
* `l`, `b`: współrzędne galaktyczne zdarzenia [stopnie].
* `paczynski0_tmax`: czas maksimum $t_0$ [dni].
* `paczynski0_te`: czas Einsteina $t_e$ [dni].
* `paczynski0_u0`: minimalny parametr zbliżenia $u_0$.
* `parallax`: paralaksa gwiazdy [mas].
* `phot_g_mean_mag`: średnia jasność w paśmie $G$.
* `paczynski0_mass` (opcjonalnie): masa soczewki [M$_\odot$].

---

## 🚀 Instrukcja Użytkowania

### Krok 1: Pobierz dane mikrosoczewkowania
Uruchom skrypt `download_gaia_catalog.py`, aby pobrać najnowszy katalog zjawisk mikrosoczewkowania z Gaia DR3:

```bash
python download_gaia_catalog.py
```
*Skrypt sprawdza czy `gaia_microlensing.fits` istnieje i pobiera dane z archiwum Gaia tylko jeśli to konieczne.*

### Krok 2: Wygeneruj statyczną mapę całego nieba (16K)
Uruchom moduł `gaia_allsky_map.py`. Przetwarza on katalog gwiazd Gaia i tworzy pliki mapy w formacie PNG oraz `.npz`:

```bash
python gaia_allsky_map.py
```

*Pliki wynikowe:*
* `gaia_allsky_hammer_16k.png`
* `gaia_allsky_hammer_16k_layout.npz`

### Krok 3: Wygeneruj animację i filmy wideo
Po zakończeniu kroków 1 i 2 uruchom symulację mikrosoczewkowania:

```bash
python microlensing_animation.py
```

*Przebieg procesu:*
1. Odczytuje zjawiska mikrosoczewkowania z `gaia_microlensing.fits`.
2. Tworzy statyczny obraz przeglądowy `microlensing_events.png`.
3. Renderuje sekwencje klatek animacji równolegle do katalogu `frames_micro/`.
4. Automatycznie koduje klatki do wideo 16K `microlensing_animation.mp4`.
5. Konwertuje do kompatybilnego z Apple/QuickTime wideo H.265 `microlensing_animation_git.mp4` z tagiem `hvc1`.
6. (Opcjonalnie) Po potwierdzeniu użytkownika generuje zoptymalizowane wideo 8K `microlensing_animation_8k.mp4` dla urządzeń mobilnych.

---

## ⚙️ Konfiguracja i Parametry

### W `download_gaia_catalog.py`:
* `OUTPUT = "gaia_microlensing.fits"`: Docelowa nazwa pliku FITS.
* `query`: Zapytanie ADQL definiujące kryteria filtrowania (np. zakres paralaksy `parallax > 0.03 AND parallax < 21`).

### W `gaia_allsky_map.py`:
* `BINS_L = 16384`, `BINS_B = 8192`: Rozdzielczość histogramu sferycznego 2D.
* `FIG_W_IN = 53.333`, `FIG_H_IN = 30.0` (przy DPI=300): Wymiary płótna.
* `_COLORS`: Autorska paleta RGB definiująca przejścia tonalne Drogi Mlecznej.

### W `microlensing_animation.py`:
* `MAX_RENDER_WORKERS = 4`: Liczba równoległych procesów renderujących.
* `FRAMES = 625`, `FPS = 25`: Długość sekwencji (625 klatek = 25 sekund przy 25 FPS).
* `OMP_NUM_THREADS = "1"`: Ogranicza wątki C/Fortran, by zapobiec przełączaniu kontekstu i walce o rdzenie.

---

## ⚡ Optymalizacja Wydajności

Kod implementuje szereg rozwiązań inżynierskich dla wydajnego przetwarzania ogromnych zbiorów danych:

1. **Asynchroniczne pobieranie danych TAP:** Użycie `Gaia.launch_job_async` w `download_gaia_catalog.py` umożliwia złożone zapytania SQL bez problemów z timeoutami HTTP.
2. **Wymuszenie trybu headless:** Wywołanie `matplotlib.use("Agg")` przed importem `pyplot` zapobiega automatycznemu wyborowi GUI na macOS, co pozwala uniknąć nadmiernego skalowania DPI i wyczerpania RAM na ekranach Retina/5K.
3. **Pamięć współdzielona i inicjalizator workerów:** Funkcja `_init_worker(bg_path)` w `multiprocessing.Pool` ładuje duży obraz tła PNG (~setki MB) do RAM każdego procesu, eliminując kosztowny transfer Pickle.
4. **Bezpośrednie płótno float32:** Unika powolnych obiektów Matplotlib `Axes` podczas renderowania klatek, operując bezpośrednio na 3-kanałowej tablicy `numpy.ndarray` float32.
5. **Przestrzenne przycinanie bounding box:** Obliczenia profilu Gaussa ograniczone są do okna $3.5\sigma$ wokół każdego zdarzenia, minimalizując niepotrzebne operacje zmiennoprzecinkowe poza mapą.

---

## 📜 Licencja i Źródła Danych

* **Źródło danych astronomicznych:** Dane pochodzą z misji kosmicznej ESA Gaia ([https://www.cosmos.esa.int/gaia](https://www.cosmos.esa.int/gaia)) oraz tabeli `gaiadr3.vari_microlensing`.
* **Licencja oprogramowania:** Projekt udostępniany jest na licencji open-source MIT.

---


> **English documentation**
# 🌌 Gaia All-Sky Map & Microlensing Animation Engine

An advanced astronomical data processing pipeline designed to download data, generate a large-format Hammer projection (16K) Milky Way map based on the **Gaia** star catalog, and render photorealistic simulations of **gravitational microlensing** events.

---

## 📋 Table of Contents
1. [About the Project](#-about-the-project)
2. [Main Features](#-main-features)
3. [Architecture and Code Structure](#-architecture-and-code-structure)
4. [Physical and Mathematical Foundations](#-physical-and-mathematical-foundations)
5. [Requirements and Dependencies](#-requirements-and-dependencies)
6. [Input Data Format](#-input-data-format)
7. [Usage Instructions](#-usage-instructions)
8. [Configuration and Parameters](#-configuration-and-parameters)
9. [Performance Optimization](#-performance-optimization)
10. [License and Data Sources](#-license-and-data-sources)

---

## 🔭 About the Project

The project consists of three integrated Python modules:
1. **`download_gaia_catalog.py`**: Automatically executes an ADQL query against the **Gaia DR3** archive using `astroquery.gaia`. It downloads microlensing event data from the `gaiadr3.vari_microlensing` table joined with `gaiadr3.gaia_source` and saves the results to a FITS file.
2. **`gaia_allsky_map.py`**: Generates a high-resolution ($16384 \times 8192$ pixels) static all-sky map in Galactic coordinates $(l, b)$ using the Hammer projection. It computes the total photon flux from Gaia catalog objects and applies a custom 10-color palette to recreate the appearance of the Milky Way.
3. **`microlensing_animation.py`**: Reads the generated base map and microlensing events, then produces an animated frame sequence illustrating dynamic gravitational microlensing brightening according to the **Paczynski** model. The engine uses optimized NumPy float32 array operations and multiprocessing, avoiding the overhead of Matplotlib's object drawing.

---

## ✨ Main Features

* **Automated Gaia TAP/ADQL data download:** Retrieves verified microlensing events directly from ESA Gaia servers, filtering parallax ($0.03 < \varpi < 21$) and Einstein timescale ($t_e$).
* **Master 16K resolution map:** Computes a 2D spherical histogram on a $16384 \times 8192$ bin grid.
* **Physical stellar brightness model:** Converts Gaia $G$-band magnitudes to physical photon flux ($F \propto 10^{-0.4 G}$) with clipping of extreme values at the 99.9th percentile.
* **Advanced dynamic range stretching (HDR Tone Mapping):** Combines Gaussian filtering ($\sigma=1.8$), logarithmic transform $\log(1+x)$, and inverse hyperbolic sine $\text{arcsinh}(x)$ to enhance dust lanes and faint disk structures.
* **Analytical Paczynski microlensing dynamics:** Accurately reproduces theoretical light curves for gravitational lensing events.
* **Ultra-fast direct-to-array rendering engine:** Rasterizes Gaussian profiles of brightening events onto a 3-channel RGB float32 canvas, eliminating Matplotlib overhead.
* **Multiprocessing parallelization:** Efficient frame rendering using `multiprocessing.Pool` with isolated background image loading to avoid large Pickle data transfers.
* **Dual video export (16K/8K H.265 HEVC):** Frames are encoded via `ffmpeg` into H.265 format with automatic `hvc1` tagging for Apple/QuickTime compatibility, plus optional 8K Lanczos downscale for mobile devices.

---

## 🏗 Architecture and Code Structure

The project is divided into three main Python modules, each responsible for a different stage of the Gaia DR3 processing pipeline:

```text
.
├── download_gaia_catalog.py          # Downloads Gaia DR3 stellar catalog data
├── download_gaia_microlensing.py     # Downloads Gaia DR3 microlensing event catalog
├── gaia_allsky_map.py                # Generates the Galactic all-sky Hammer projection map
├── microlensing_animation.py         # Renders microlensing animation and creates videos
├── README.md                         # Project documentation
├── LICENSE                           # MIT license
├── requirements.txt                  # Python dependencies
├── gaia_150m_allsky.fits             # Local Gaia star catalog (user data)
├── gaia_microlensing.fits            # Gaia microlensing events catalog
├── gaia_allsky_hammer_16k.png        # Generated all-sky background map
├── gaia_allsky_hammer_16k_layout.npz # Cached map layout metadata
├── frames_micro/                     # Temporary rendered animation frames
└── microlensing_animation_*.mp4      # Generated video outputs
```

### Module responsibilities:

#### `download_gaia_catalog.py`
Downloads Gaia DR3 catalog data required for stellar map generation using ADQL queries through `astroquery.gaia` and saves results as FITS files.

#### `download_gaia_microlensing.py`
Downloads confirmed Gaia DR3 microlensing events from the `gaiadr3.vari_microlensing` table and combines them with relevant stellar parameters.

#### `gaia_allsky_map.py`
Processes Gaia stellar data, converts Galactic coordinates and brightness values into a surface brightness map of the Milky Way, and generates the high-resolution PNG background used by the animation engine.

#### `microlensing_animation.py`
Loads the generated map and microlensing catalog, calculates Paczynski light curves, renders brightness events, and produces final video files using FFmpeg.

---

## 📐 Physical and Mathematical Foundations

### 1. Theoretical Paczynski Light Curve
The amplification of photon flux from a source star $A(t)$ at time $t$ is given by:

$$A(t) = \frac{u(t)^2 + 2}{u(t) \sqrt{u(t)^2 + 4}}$$

where the normalized angular separation $u(t)$ on the lens plane is:

$$u(t) = \sqrt{u_0^2 + \left(\frac{t - t_0}{t_e}\right)^2}$$

* $t_0$ – time of maximum alignment (peak magnification),
* $t_e$ – Einstein timescale (crossing time of Einstein radius),
* $u_0$ – minimum impact parameter in Einstein radius units.

### 2. Hammer Projection Geometric Transformation
Conversion from spherical Galactic coordinates $(l, b)$ in radians to flat Hammer projection coordinates $(x_H, y_H)$:

$$z = \sqrt{1 + \cos(b) \cdot \cos\left(\frac{l}{2}\right)}$$

$$x_H = \frac{2\sqrt{2} \cdot \cos(b) \cdot \sin\left(\frac{l}{2}\right)}{z}, \quad y_H = \frac{\sqrt{2} \cdot \sin(b)}{z}$$

Mapping continuous $x_H \in [-2\sqrt{2}, 2\sqrt{2}]$ and $y_H \in [-\sqrt{2}, \sqrt{2}]$ to discrete pixel matrix coordinates $(W \times H)$:

$$p_x = \left(\frac{x_H}{2\sqrt{2}} + 1\right) \cdot \frac{W}{2}, \quad p_y = \left(1 - \frac{y_H}{\sqrt{2}}\right) \cdot \frac{H}{2}$$

### 3. Saturation Profile and Flash Rasterization
Each microlensing flash is represented by a 2D flattened Gaussian profile. The disk radius $\sigma_{\text{px}}$ and peak intensity $\alpha$ depend on the amplification $\Delta A = A(t) - 1$:

$$\sigma_{\text{px}} = \text{clip}\left( \sqrt{\frac{S_{\text{pt2}}}{\pi}} \cdot \frac{\text{DPI}}{72} \cdot \frac{1}{3} \cdot M \cdot \text{scale}, \, 2.0, \, 0.08 \cdot H \right)$$

where $S_{\text{pt2}} = 10.0 + 10000.0 \cdot \ln(1 + \Delta A)$, and $M$ is the lens mass.

---

## 📦 Requirements and Dependencies

The project requires **Python 3.9+** and the following Python packages:

```bash
pip install numpy scipy matplotlib astropy pandas tqdm astroquery healpy opencv-python
```

External system tool:
* **FFmpeg** (with built-in `libx265` / H.265 encoder).

### FFmpeg Installation:
* **macOS:** `brew install ffmpeg`
* **Linux (Ubuntu/Debian):** `sudo apt update && sudo apt install ffmpeg`
* **Windows (Chocolatey):** `choco install ffmpeg`

---

## 📄 Input Data Format

### 1. Gaia Star Catalog (`gaia_150m_allsky.fits`)
A FITS file containing star positions and photometry. Required columns:
* `l`: Galactic longitude in degrees $[0^\circ, 360^\circ]$.
* `b`: Galactic latitude in degrees $[-90^\circ, +90^\circ]$.
* `phot_g_mean_mag`: Observed magnitude in Gaia $G$ band.

### 2. Microlensing Events Catalog (`gaia_microlensing.fits`)
A FITS table (automatically generated by `download_gaia_catalog.py` or provided manually). Required columns:
* `source_id`: Gaia object identifier.
* `l`, `b`: Galactic coordinates of the event [degrees].
* `paczynski0_tmax`: Peak time $t_0$ [days].
* `paczynski0_te`: Einstein timescale $t_e$ [days].
* `paczynski0_u0`: Minimum impact parameter $u_0$.
* `parallax`: Star parallax [mas].
* `phot_g_mean_mag`: Mean $G$-band magnitude.
* `paczynski0_mass` (optional): Lens mass [M$_\odot$].

---

## 🚀 Usage Instructions

### Step 1: Download Microlensing Data
Run the `download_gaia_catalog.py` script to fetch the latest microlensing event catalog from Gaia DR3:

```bash
python download_gaia_catalog.py
```
*The script checks if `gaia_microlensing.fits` exists and queries the Gaia archive if needed.*

### Step 2: Generate Static All-Sky Map (16K)
Run the `gaia_allsky_map.py` module. It processes the Gaia star catalog and creates cached map files in PNG and `.npz` formats:

```bash
python gaia_allsky_map.py
```

*Generated files:*
* `gaia_allsky_hammer_16k.png`
* `gaia_allsky_hammer_16k_layout.npz`

### Step 3: Generate Animation and Videos
After completing Steps 1 and 2, run the microlensing simulation:

```bash
python microlensing_animation.py
```

*Processing workflow:*
1. Reads microlensing events from `gaia_microlensing.fits`.
2. Creates a static overview image `microlensing_events.png`.
3. Renders animation frames in parallel into the `frames_micro/` directory.
4. Automatically encodes frames into the 16K video `microlensing_animation.mp4`.
5. Converts to Apple/QuickTime compatible H.265 video `microlensing_animation_git.mp4` with `hvc1` tag.
6. (Optional) Upon user confirmation, generates an optimized 8K Lanczos downscaled video `microlensing_animation_8k.mp4` for mobile devices.

---

## ⚙️ Configuration and Parameters

### In `download_gaia_catalog.py`:
* `OUTPUT = "gaia_microlensing.fits"`: Target FITS output filename.
* `query`: ADQL query defining event filtering criteria (e.g., parallax range `parallax > 0.03 AND parallax < 21`).

### In `gaia_allsky_map.py`:
* `BINS_L = 16384`, `BINS_B = 8192`: Resolution of the 2D spherical histogram.
* `FIG_W_IN = 53.333`, `FIG_H_IN = 30.0` (at DPI=300): Canvas drawing dimensions.
* `_COLORS`: Custom RGB palette defining Milky Way tonal transitions.

### In `microlensing_animation.py`:
* `MAX_RENDER_WORKERS = 4`: Number of parallel rendering processes.
* `FRAMES = 625`, `FPS = 25`: Sequence length (625 frames = 25 seconds at 25 FPS).
* `OMP_NUM_THREADS = "1"`: Limits C/Fortran threading to prevent CPU contention and context switching overhead.

---

## ⚡ Performance Optimization

The code implements several engineering solutions for efficient processing of massive datasets:

1. **Asynchronous TAP data retrieval:** Using `Gaia.launch_job_async` in `download_gaia_catalog.py` enables complex SQL queries without HTTP timeout issues.
2. **Headless backend enforcement:** Calling `matplotlib.use("Agg")` before importing `pyplot` prevents automatic GUI selection on macOS, avoiding excessive DPI scaling and RAM exhaustion on Retina/5K displays.
3. **Shared memory and worker initializer:** The `_init_worker(bg_path)` function in `multiprocessing.Pool` loads the large background PNG (~hundreds of MB) into each child process’s RAM once at startup, eliminating huge Pickle data transfer overhead.
4. **Direct float32 canvas:** Avoids slow Matplotlib `Axes` objects during frame rendering by direct manipulation of a 3-channel `numpy.ndarray` float32 array.
5. **Spatial bounding box clipping:** Gaussian profile computations are restricted to a $3.5\sigma$ radius window around each event, minimizing unnecessary floating-point operations over empty map areas.

---

## 📜 License and Data Sources

* **Astronomical data source:** Data originates from the ESA Gaia space mission ([https://www.cosmos.esa.int/gaia](https://www.cosmos.esa.int/gaia)) and the `gaiadr3.vari_microlensing` table.
* **Software license:** The project is released under the open-source MIT license.
