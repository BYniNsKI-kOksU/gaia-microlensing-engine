from astroquery.gaia import Gaia
from astropy.table import vstack
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

    tables = []
    chunk  = N_TOTAL // N_CHUNKS

    for i in range(N_CHUNKS):
        ri_lo = i * chunk
        ri_hi = (i + 1) * chunk if i < N_CHUNKS - 1 else N_TOTAL

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
        job   = Gaia.launch_job_async(query)
        result = job.get_results()

        print(f"  → {len(result):,} gwiazd  ({time.time()-start:.1f}s)")
        tables.append(result)

    print("\nŁączenie przedziałów...")
    data = vstack(tables)
    print(f"Łącznie gwiazd: {len(data):,}")

    print(f"Zapisywanie → {OUTPUT}")
    data.write(OUTPUT, format="fits", overwrite=True)
    print("Gotowe.")

if __name__ == "__main__":
    download_gaia()