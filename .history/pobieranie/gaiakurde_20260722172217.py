from astroquery.gaia import Gaia
from astropy.table import vstack, Table
import os
import threading
import time

OUTPUT        = "gaia_150m_allsky.fits"
N_CHUNKS      = 50
PER_CHUNK     = 3_000_000
QUERY_TIMEOUT = 20 * 60  # s – if the query does not finish within this time, we consider it hung

# Gaia DR3: total number of sources → random_index ∈ [0, N_TOTAL)
# random_index is randomly assigned to each star without
# any spatial dependence, so dividing the values into
# N equal intervals gives N spatially homogeneous samples.
N_TOTAL = 1_811_709_771


def _run_query_with_timeout(query, timeout):
    """
    Gaia.launch_job_async() itself BLOCKS and waits until the job
    finishes on the ESA server side (it polls the status in a loop without
    any timeout). If the server queue is full,
    this single line can hang for hours – and then the existing
    except block never triggers, because no exception is thrown.
    Therefore, we run the query in a separate thread (daemon, so it won't
    block program exit) and enforce a hard timeout
    via t.join(timeout).
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
        raise TimeoutError(f"Query did not finish within {timeout}s – probably hung on the server.")
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

    for i in range(38, N_CHUNKS):
        ri_lo = i * chunk
        ri_hi = (i + 1) * chunk if i < N_CHUNKS - 1 else N_TOTAL

        chunk_file = f"gaia_chunk_{i:02d}.fits"
        if os.path.exists(chunk_file):
            print(f"{chunk_file} już istnieje – pomijam.")
            continue

        print(f"Przedział {i+1}/{N_CHUNKS}:  random_index ∈ [{ri_lo:,}, {ri_hi:,})")

        # Without ORDER BY: sorting several million rows on the server side
        # is costly and not needed here — random_index is already
        # random, and the sample is homogeneous thanks to the WHERE filter itself.
        # ORDER BY only prolongs the query execution time (and this most likely
        # caused the multi-hour waiting).
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