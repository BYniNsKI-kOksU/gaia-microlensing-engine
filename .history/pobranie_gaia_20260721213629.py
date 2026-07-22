from astroquery.gaia import Gaia
from astropy.table import vstack, Table
import os
import threading
import time

OUTPUT        = "gaia_150m_allsky.fits"
N_CHUNKS      = 50
PER_CHUNK     = 1_000_000
QUERY_TIMEOUT = 20 * 60  # s – jeśli zapytanie nie skończy się w tym czasie, uznajemy je za zawieszone

# Gaia DR3: łączna liczba źródeł → random_index ∈ [0, N_TOTAL)
# random_index jest losowo przypisany do każdej gwiazdy bez
# żadnej zależności przestrzennej, więc podział wartości na
# N równych przedziałów daje N przestrzennie jednorodnych prób.
N_TOTAL = 1_811_709_771


def _run_query_with_timeout(query, timeout):
    """
    Gaia.launch_job_async() samo w sobie BLOKUJE i czeka aż zadanie
    skończy się po stronie serwera ESA (odpytuje status w pętli bez
    żadnego limitu czasu). Jeśli kolejka na serwerze jest zapchana,
    ta pojedyncza linijka potrafi wisieć godzinami – i wtedy istniejący
    blok except nigdy się nie uruchamia, bo nic nie rzuca wyjątku.
    Dlatego uruchamiamy zapytanie w osobnym wątku (daemon, więc nie
    zablokuje zamknięcia programu) i sami narzucamy twardy timeout
    przez t.join(timeout).
    """
    box = {}

    def worker():
        try:
            job = Gaia.launch_job_async(query)
            box["result"] = job.get_results()
        except Exception as e:
            box["error"] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout)

    if t.is_alive():
        raise TimeoutError(f"Zapytanie nie zakończyło się w {timeout}s – prawdopodobnie zawieszone na serwerze.")
    if "error" in box:
        raise box["error"]
    return box["result"]


def download_gaia():

    if os.path.exists(OUTPUT):
        print(f"Plik {OUTPUT} już istnieje – pomijam pobieranie.")
        return

    print(f"Pobieranie Gaia DR3 – {N_CHUNKS} przedziałów random_index z jasnością G")
    print(f"Każdy przedział: TOP {PER_CHUNK:,} gwiazd\n")

    chunk = N_TOTAL // N_CHUNKS

    for i in range(N_CHUNKS):
        ri_lo = i * chunk
        ri_hi = (i + 1) * chunk if i < N_CHUNKS - 1 else N_TOTAL

        chunk_file = f"gaia_chunk_{i:02d}.fits"
        if os.path.exists(chunk_file):
            print(f"{chunk_file} już istnieje – pomijam.")
            continue

        print(f"Przedział {i+1}/{N_CHUNKS}:  random_index ∈ [{ri_lo:,}, {ri_hi:,})")

        # Bez ORDER BY: sortowanie kilku milionów wierszy po stronie serwera
        # jest kosztowne i wcale nie jest tu potrzebne — random_index już
        # jest losowy, a próbka jest jednorodna dzięki samemu filtrowi WHERE.
        # ORDER BY tylko wydłuża czas wykonania zapytania (a to on najpewniej
        # odpowiadał za wielogodzinne oczekiwanie).
        query = f"""
        SELECT TOP {PER_CHUNK}
            l,
            b,
            phot_g_mean_mag
        FROM gaiadr3.gaia_source
        WHERE
            random_index >= {ri_lo}
            AND random_index < {ri_hi}
            AND l IS NOT NULL
            AND b IS NOT NULL
            AND phot_g_mean_mag IS NOT NULL
        """

        start = time.time()
        result = None
        for attempt in range(5):
            try:
                result = _run_query_with_timeout(query, QUERY_TIMEOUT)
                break
            except Exception as e:
                print(f"Błąd: {e}")
                if attempt == 4:
                    raise
                wait = 30 * (attempt + 1)
                print(f"Ponawiam za {wait} s...")
                time.sleep(wait)

        print(f"  → {len(result):,} gwiazd  ({time.time()-start:.1f}s)")
        result.write(chunk_file, format="fits", overwrite=True)
        del result

    print("\nŁączenie przedziałów...")
    tables = []
    for i in range(N_CHUNKS):
        chunk_file = f"gaia_chunk_{i:02d}.fits"
        if not os.path.exists(chunk_file):
            raise FileNotFoundError(f"Brak pliku {chunk_file}")
        tables.append(Table.read(chunk_file))

    data = vstack(tables)
    print(f"Łącznie gwiazd: {len(data):,}")

    print(f"Zapisywanie → {OUTPUT}")
    data.write(OUTPUT, format="fits", overwrite=True)
    print("Gotowe.")


if __name__ == "__main__":
    download_gaia()