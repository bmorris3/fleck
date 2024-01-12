import numpy as np
import os
import astropy.units as u
from jax import numpy as jnp

from ..jax import ActiveStar


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
        u_ld=jnp.array(planet.u),
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
