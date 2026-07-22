from astroquery.gaia import Gaia
from astropy.table import vstack, Table
import os
import time

OUTPUT = "gaia_30m_allsky.fits"
N_SECTORS = 10
MAX_PER_QUERY = 3000000

def download_gaia():

    if os.path.exists(OUTPUT):
        print(f"Plik {OUTPUT} już istnieje.")
        return

    print("Pobieranie Gaia DR3 - 10 sektorów całego nieba")
    
    tables = []

    # podział po długości galaktycznej
    step = 360 / N_SECTORS

    for i in range(N_SECTORS):

        l_min = i * step
        l_max = (i + 1) * step

        print(
            f"\nSektor {i+1}/{N_SECTORS}: "
            f"l={l_min:.1f} - {l_max:.1f}"
        )

        query = f"""
        SELECT TOP {MAX_PER_QUERY}
            l,
            b
        FROM gaiadr3.gaia_source
        WHERE
            l >= {l_min}
            AND l < {l_max}
            AND l IS NOT NULL
            AND b IS NOT NULL
        ORDER BY random_index
        """

        start = time.time()

        job = Gaia.launch_job_async(query)
        result = job.get_results()

        print(
            f"Pobrano: {len(result):,} gwiazd "
            f"({time.time()-start:.1f}s)"
        )

        tables.append(result)


    print("\nŁączenie sektorów...")
    data = vstack(tables)

    # sortowanie według długości galaktycznej dla spójnego katalogu
    data.sort("l")
    print(
        f"Łącznie gwiazd w katalogu: {len(data):,}"
    )
    print("Zapisywanie pełnego katalogu Gaia całego nieba...")

    data.write(
        OUTPUT,
        format="fits",
        overwrite=True
    )

    print(
        f"Gotowy katalog: {OUTPUT}"
    )

if __name__ == "__main__":
    download_gaia()