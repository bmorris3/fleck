import numpy as np
import os
import astropy.units as u
from batman import TransitParams

from ..core import Star


def test_stsp_rotational_modulation():
    """
    Compare fleck results to STSP results
    """
    stsp_lc = np.loadtxt(os.path.join(os.path.dirname(__file__), os.pardir,
                                      'data', 'stsp_rotation.txt'))

    n_phases = 1000
    spot_contrast = 0.7
    u_ld = [0.5079, 0.2239]
    inc_stellar = 90

    lat1, lon1, rad1 = 10, 0, 0.1
    lat2, lon2, rad2 = 75, 180, 0.1

    lats = np.array([lat1, lat2])[:, np.newaxis]
    lons = np.array([lon1, lon2])[:, np.newaxis]
    rads = np.array([rad1, rad2])[:, np.newaxis]

    star = Star(spot_contrast, u_ld, n_phases=n_phases)
    fleck_lc = star.light_curve(lons * u.deg, lats * u.deg, rads,
                                inc_stellar * u.deg)

    # Assert matches STSP results to within 100 ppm:
    np.testing.assert_allclose(fleck_lc[:, 0], stsp_lc, atol=100e-6)


def test_stsp_transit():

    planet = TransitParams()
    planet.per = 88
    planet.a = float(0.387*u.AU / u.R_sun)
    planet.rp = 0.1
    planet.w = 90
    planet.ecc = 0
    planet.inc = 90
    planet.t0 = 0
    planet.limb_dark = 'quadratic'
    planet.u = [0.5079, 0.2239]

    stsp_lc = np.loadtxt(os.path.join(os.path.dirname(__file__), os.pardir,
                                      'data', 'stsp_transit.txt'))

    inc_stellar = 90 * u.deg
    spot_radii = np.array([[0.1], [0.1]])
    spot_lats = np.array([[0], [0]]) * u.deg
    spot_lons = np.array([[360-30], [30]]) * u.deg

    times = np.linspace(-0.5, 0.5, 500)

    star = Star(spot_contrast=0.7, u_ld=planet.u, rotation_period=100)

    fleck_lc = star.light_curve(spot_lons, spot_lats, spot_radii,
                                inc_stellar, planet=planet, times=times)

    # Assert matches STSP results to within 100 ppm:
    np.testing.assert_allclose(fleck_lc[:, 0], stsp_lc, atol=350e-6)
