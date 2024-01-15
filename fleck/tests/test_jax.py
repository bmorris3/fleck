import pytest

jax = pytest.importorskip("jax")

import numpy as np
import os
import astropy.units as u
from jax import numpy as jnp

from ..jax import ActiveStar


def test_rotation():
    stsp_lc = np.loadtxt(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir, 'data', 'stsp_rotation.txt'
        )
    )

    n_phases = 1000
    # spot_contrast = 0.7
    # inc_stellar = 90
    u_ld = [0.5079, 0.2239]

    lat1, lon1, rad1 = 10, 0, 0.1
    lat2, lon2, rad2 = 75, 180, 0.1

    times = jnp.linspace(0, 1, n_phases)
    active_star = ActiveStar(
        times=times,
        wavelength=jnp.array([1]),
        inclination=np.pi / 2,
        phot=jnp.array([1]),
        P_rot=1,

    )

    active_star.lon = jnp.radians(jnp.array([lon1, lon2]))
    active_star.lat = jnp.radians(jnp.array([lat1, lat2])) - np.pi / 2
    active_star.rad = jnp.array([rad1, rad2])
    active_star.spectrum = jnp.array([0.7, 0.7])

    fleck_lc = active_star.rotation_model(
        f0=1, t0_rot=0.5, u1=u_ld[0], u2=u_ld[1]
    )[0][:, 0, 0]

    # Assert matches STSP results to within 50 ppm:
    np.testing.assert_allclose(fleck_lc, stsp_lc, atol=50e-6)


def test_jax_transit():
    from batman import TransitParams

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

    stsp_lc = np.loadtxt(
        os.path.join(
            os.path.dirname(__file__), os.pardir,
            'data', 'stsp_single_transit.txt'
        )
    )

    inc_stellar = 90 * u.deg
    spot_radii = np.array([[0.1], [0.1]])
    spot_lats = np.array([[0], [0]]) * u.deg
    spot_lons = np.array([[360 - 30], [30]]) * u.deg

    times = np.linspace(-0.5, 0.5, 500)

    active_star = ActiveStar(
        times=times,
        lon=np.ravel(spot_lons.to_value(u.rad)),
        lat=np.ravel(np.pi/2 + spot_lats.to_value(u.rad)),
        rad=np.ravel(spot_radii),
        inclination=inc_stellar.to_value(u.rad),
        phot=jnp.ones(10),
        wavelength=jnp.linspace(0.1, 1, 10),
        spectrum=jnp.ones(10) * 0.7,
        P_rot=100,
        T_eff=2600,
        temperature=jnp.array([2400])
    )

    jax_lc = active_star.transit_model(
        t0=0, period=planet.per, rp=planet.rp,
        a=planet.a, inclination=np.radians(planet.inc),
        u1=jnp.array([planet.u[0]]), u2=jnp.array([planet.u[1]])
    )[0]

    assert np.max(np.abs(stsp_lc - jax_lc[:, 0]) / jax_lc[:, 0]) < 5.0e-4
    assert np.std(stsp_lc / jax_lc[:, 0] - 1) < 1.5e-4
