"""
Downloads a large Gaia DR3 stellar catalog for all-sky map generation.

The program retrieves Gaia star positions and G-band magnitudes in chunks
using random_index ranges, saves intermediate FITS files, and combines them
into a final catalog ready for further processing.
"""
from astroquery.gaia import Gaia
from astropy.table import vstack, Table
import os
import threading
import time

OUTPUT        = "gaia_150m_allsky.fits"
N_CHUNKS      = 50
PER_CHUNK     = 3_000_000
QUERY_TIMEOUT = 20 * 60  # Maximum waiting time for a single Gaia archive query.

# Total number of Gaia DR3 sources used for dividing the download into chunks.
# random_index provides a uniform distribution of stars across the catalog.
N_TOTAL = 1_811_709_771


def _run_query_with_timeout(query, timeout):
    """Runs a Gaia archive query with a maximum execution time limit."""
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
        raise TimeoutError(f"Query did not complete in {timeout}s – probably hung on the server.")
    if "error" in box:
        raise box["error"]
    return box["result"]


def download_gaia():

    if os.path.exists(OUTPUT):
        print(f"File {OUTPUT} already exists - skipping download.")
        return

    print(f"Downloading Gaia DR3 - {N_CHUNKS} random_index ranges with G-band magnitudes")
    print(f"Each range: TOP {PER_CHUNK:,} stars\n")

    chunk = N_TOTAL // N_CHUNKS

    for i in range(N_CHUNKS):
        ri_lo = i * chunk
        ri_hi = (i + 1) * chunk if i < N_CHUNKS - 1 else N_TOTAL

        chunk_file = f"gaia_chunk_{i:02d}.fits"
        if os.path.exists(chunk_file):
            print(f"{chunk_file} already exists - skipping.")
            continue

        print(f"Range {i+1}/{N_CHUNKS}: random_index ∈ [{ri_lo:,}, {ri_hi:,})")

        # random_index ranges provide evenly distributed samples without additional sorting.
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
        ORDER BY random_index
        """

        start = time.time()
        result = None
        for attempt in range(5):
            try:
                result = _run_query_with_timeout(query, QUERY_TIMEOUT)
                break
            except Exception as e:
                print(f"Error: {e}")
                if attempt == 4:
                    raise
                wait = 30 * (attempt + 1)
                print(f"Retrying in {wait} s...")
                time.sleep(wait)

        print(f"  → {len(result):,} gwiazd  ({time.time()-start:.1f}s)")
        result.write(chunk_file, format="fits", overwrite=True)
        del result

    print("\nCombining catalog chunks...")
    tables = []
    for i in range(N_CHUNKS):
        chunk_file = f"gaia_chunk_{i:02d}.fits"
        if not os.path.exists(chunk_file):
            raise FileNotFoundError(f"Brak pliku {chunk_file}")
        tables.append(Table.read(chunk_file))

    data = vstack(tables)
    print(f"Total stars: {len(data):,}")

    print(f"Saving catalog → {OUTPUT}")
    data.write(OUTPUT, format="fits", overwrite=True)
    print("Finished.")


if __name__ == "__main__":
    download_gaia()