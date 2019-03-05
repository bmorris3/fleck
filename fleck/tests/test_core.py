import numpy as np
import os
import astropy.units as u

from ..core import Stars


def test_stsp():
    """
    Compare fleck results to STSP results
    """
    stsp_lc = np.loadtxt(os.path.join(os.path.dirname(__file__), os.pardir,
                                      'data', 'stsp.txt'))

    n_phases = 1000
    spot_contrast = 0.7
    u_ld = [0.5079, 0.2239]
    inc_stellar = 90

    lat1, lon1, rad1 = 10, 0, 0.1
    lat2, lon2, rad2 = 75, 180, 0.1

    lats = np.array([lat1, lat2])[:, np.newaxis]
    lons = np.array([lon1, lon2])[:, np.newaxis]
    rads = np.array([rad1, rad2])[:, np.newaxis]

    stars = Stars(spot_contrast, n_phases, u_ld)
    fleck_lc = stars.light_curves(lons * u.deg, lats * u.deg, rads,
                                  inc_stellar * u.deg).value

    # Assert matches STSP results to within 100 ppm:
    np.testing.assert_allclose(fleck_lc[:, 0], stsp_lc, atol=100e-6)
