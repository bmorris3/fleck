import os
import pytest
from collections import namedtuple

from batman import TransitParams
import numpy as np
import astropy.units as u

from fleck.core import Star
from fleck.tests.stsp import STSP


p = TransitParams()
p.per = 1.5
p.a = 15
p.t0 = 0
p.rp = 0.05
p.u = [0.1, 0.05]
p.limb_dark = 'quadratic'
p.w = 90
p.ecc = 0

# stellar params:
p.per_rot = 5
contrast = 0.2

test_times = np.linspace(-0.05, 0.05, 500)

# spot_params order: [r, theta, phi]
system_configurations = [

    # columns: i_s, i_orb, lambda, OOT max, and
    # spot [r, theta, phi] int STSP coord convention:

    # with stellar inclination 85 deg, b=0:
    # vary projected spin-orbit angle and spot positions
    (85, 90, 0, 1, [0.05, np.pi/2, 0]),
    (85, 90, 30, 1, [0.05, 1.15 * np.pi/2, 2 * np.pi - 0.5]),
    (85, 90, 30, 1, [0.05, 0.8 * np.pi/2, 0.5 + 0.03]),
    (85, 90, 90, 1, [0.05, np.pi/4, 0]),
    (85, 90, 270, 1, [0.05, 3 * np.pi/4, 0]),

    # with stellar inclination 120 deg, b=0.4:
    # vary projected spin-orbit angle and spot positions
    (120, 88.5, -30, 1, [0.03, 0.55 * np.pi, 2*np.pi - 1.1]),
    (200, 88.5, 200, 1, [0.05, 0.55 * np.pi, 2*np.pi - 0.45]),

    # with stellar inclination {0, 180} deg, b=0.1, check that STSP's
    # "real maximum" parameter brings the STSP model into
    # agreement with fleck's default output.
    (0, 89.6, 0, 0.9992497284217082, [0.03, 0.05, 0.05]),
    (180, 89.6, 45, 0.9992497284217082, [0.03, np.pi - 0.05, 0.9 * np.pi]),

    # with stellar inclination {0, 180} deg, b={0.1, 0.4}:
    # vary projected spin-orbit angle and spot positions
    (0, 89.6, 0, 0.9992497284217082, [0.03, 0.05, 0.05]),
    (180, 89.6, 45, 0.9939219210476027,
     [[0.04, 0.7 * np.pi, 0.3 * np.pi], [0.08, np.pi - 0.06, 2.5]]),
    (180, 89.6, 90, 0.9939219210476027,
     [[0.04, 0.7 * np.pi, 3.03], [0.08, np.pi - 0.06, 2.5]]),
    (0, 89.6, 90, 0.9939219210476027,
     [[0.04, 0.3 * np.pi, 3.03], [0.08, 0.06, 2.5]]),
    (0, 88.5, 100, 0.998567636449246,
     [[0.04, 0.3 * np.pi, 0.2 * np.pi], [0.03, 0.4, 0.4 * np.pi]]),
]


def generate_stsp_light_curves():
    # run this manually to recreate the test light curve files
    LightCurve = namedtuple('LightCurve', 'times')
    JD = namedtuple('JD', 'jd')

    for i, config in enumerate(system_configurations):
        inc_stellar, inc, lam, real_max, spot_params = config
        p.lam = lam
        p.inc_stellar = inc_stellar
        p.inc = inc
        lc = LightCurve(times=JD(test_times))
        sim = STSP(lc, transit_params=p, spot_params=spot_params)
        _, flux_stsp = sim.stsp_lc(
            contrast=contrast,
            n_ld_rings=100,
            real_max=real_max,
            verbose=False
        )
        path = os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            'data',
            f'stsp_{i:02d}.txt'
        )
        np.savetxt(path, flux_stsp)


@pytest.mark.parametrize(
    "i, config,",
    enumerate(system_configurations)
)
def test_fleck_against_stsp(i, config):
    inc_stellar, inc, lam, _, spot_params = config
    p.lam = lam
    p.inc_stellar = inc_stellar
    p.inc = inc
    star = Star(contrast, p.u, rotation_period=p.per_rot)
    spot_params = np.atleast_2d(spot_params)
    fleck_lon, fleck_lat, fleck_rad = [
        spot_params[:, 2] * u.rad,
        # convert colatitude to latitude:
        (np.pi/2 - spot_params[:, 1]) * u.rad,
        spot_params[:, 0]
    ]
    flux_fleck = star.light_curve(
        fleck_lon, fleck_lat, fleck_rad,
        inc_stellar=p.inc_stellar * u.deg,
        times=test_times,
        planet=p,
        time_ref=0.0
    ).ravel()

    path = os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        'data',
        f'stsp_{i:02d}.txt'
    )
    flux_stsp = np.loadtxt(path)

    # absolute agreement of 80 ppm
    np.testing.assert_allclose(flux_fleck, flux_stsp, atol=80e-6)


def test_flux_decrement():
    n_phases = 1000
    spot_contrast = 0.7
    u_ld = [0, 0]
    inc_stellar = 90

    lats = np.array([0])[:, np.newaxis]
    lons = np.array([180])[:, np.newaxis]
    rads = np.array([0.1])[:, np.newaxis]

    star = Star(spot_contrast, u_ld, n_phases=n_phases)
    fleck_lc = star.light_curve(lons * u.deg, lats * u.deg, rads,
                                inc_stellar * u.deg)

    analytic_depth = rads ** 2 * (1 - spot_contrast)

    # Ensure that flux minimum occurs when star is rotated half-way:
    assert fleck_lc.argmin() == fleck_lc.shape[0] // 2

    # Ensure that flux minimum is the correct depth:
    assert abs(fleck_lc.min() - (1 - analytic_depth)) < 1e-6

    # Ensure that the maximum flux is unity:
    assert fleck_lc.max() == 1.0


if __name__ == '__main__':
    """
    To re-generate STSP light curves for the tests, clone
    https://github.com/lesliehebb/STSP and compile the executable.
    Then add an env var $STSP_PATH to your .bashrc which points to the
    executable, and run this python script by calling:
         python fleck/tests/test_core.py
    """
    generate_stsp_light_curves()
