"""
Downloads Gaia DR3 gravitational microlensing events.

The program queries the Gaia DR3 vari_microlensing table, combines microlensing
parameters with stellar information from gaia_source, filters invalid sources,
and saves the resulting catalog as a FITS file.

The output catalog is used by the animation pipeline to visualize microlensing
events on the Gaia all-sky Milky Way map.
"""

from astroquery.gaia import Gaia
from astropy.table import Table
import os
import time

OUTPUT = "gaia_microlensing.fits"


def download_microlensing():

    if os.path.exists(OUTPUT):
        print(f"File {OUTPUT} already exists - skipping download.")
        return

    print("Downloading Gaia DR3 microlensing events...")

    query = """
    SELECT
        v.source_id,
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
        g.l IS NOT NULL
        AND g.b IS NOT NULL
        AND v.paczynski0_te IS NOT NULL
        AND g.parallax IS NOT NULL
        AND g.parallax > 0.03
        AND g.parallax < 21
    """

    start = time.time()

    job = Gaia.launch_job_async(query)
    data = job.get_results()

    print(f"Downloaded microlensing events: {len(data):,}")
    print(f"Elapsed time: {time.time()-start:.1f}s")

    print(f"Saving catalog -> {OUTPUT}")

    data.write(
        OUTPUT,
        format="fits",
        overwrite=True
    )

    print("Finished.")


if __name__ == "__main__":
    download_microlensing()