from astroquery.gaia import Gaia
from astropy.table import Table
import os
import time

OUTPUT = "gaia_microlensing.fits"


def download_microlensing():

    if os.path.exists(OUTPUT):
        print(f"Plik {OUTPUT} już istnieje – pomijam pobieranie.")
        return

    print("Pobieranie zdarzeń mikrosoczewkowania Gaia DR3...")

    query = """
    SELECT
        v.source_id,
        g.ra,
        g.dec,
        g.l,
        g.b,
        v.paczynski0_tmax,
        v.paczynski0_te,
        v.paczynski0_u0,
        g.parallax,
        g.phot_g_mean_mag
    FROM gaiadr3.vari_microlensing AS v
    JOIN gaiadr3.gaia_source AS g
        ON v.source_id = g.source_id
    WHERE
        g.ra IS NOT NULL
        AND g.dec IS NOT NULL
        AND v.paczynski0_te IS NOT NULL
        AND g.parallax IS NOT NULL
        AND g.parallax > 0.1
        AND g.parallax < 20
    """

    start = time.time()

    job = Gaia.launch_job_async(query)
    data = job.get_results()

    print(f"Pobrano zdarzeń: {len(data):,}")
    print(f"Czas: {time.time()-start:.1f}s")

    print(f"Zapisywanie → {OUTPUT}")

    data.write(
        OUTPUT,
        format="fits",
        overwrite=True
    )

    print("Gotowe.")


if __name__ == "__main__":
    download_microlensing()