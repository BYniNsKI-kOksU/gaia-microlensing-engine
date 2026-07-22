from astroquery.gaia import Gaia
from astropy.table import vstack, Table
import os
import time

OUTPUT    = "gaia_120m_allsky.fits"
N_CHUNKS  = 40
PER_CHUNK = 3_000_000

# Gaia DR3: łączna liczba źródeł → random_index ∈ [0, N_TOTAL)
# random_index jest losowo przypisany do każdej gwiazdy bez
# żadnej zależności przestrzennej, więc podział wartości na
# N równych przedziałów daje N przestrzennie jednorodnych prób.
N_TOTAL = 1_811_709_771

def download_gaia():

    if os.path.exists(OUTPUT):
        print(f"Plik {OUTPUT} już istnieje – pomijam pobieranie.")
        return

    print(f"Pobieranie Gaia DR3 – {N_CHUNKS} przedziałów random_index")
    print(f"Każdy przedział: TOP {PER_CHUNK:,} gwiazd\n")

    chunk  = N_TOTAL // N_CHUNKS

    for i in range(N_CHUNKS):
        ri_lo = i * chunk
        ri_hi = (i + 1) * chunk if i < N_CHUNKS - 1 else N_TOTAL

        chunk_file = f"gaia_chunk_{i:02d}.fits"
        if os.path.exists(chunk_file):
            print(f"{chunk_file} już istnieje – pomijam.")
            continue

        print(f"Przedział {i+1}/{N_CHUNKS}:  random_index ∈ [{ri_lo:,}, {ri_hi:,})")

        query = f"""
        SELECT TOP {PER_CHUNK}
            l,
            b
        FROM gaiadr3.gaia_source
        WHERE
            random_index >= {ri_lo}
            AND random_index < {ri_hi}
            AND l IS NOT NULL
            AND b IS NOT NULL
        ORDER BY random_index
        """

        start = time.time()
        for attempt in range(5):
            try:
                job = Gaia.launch_job_async(query)
                result = job.get_results()
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