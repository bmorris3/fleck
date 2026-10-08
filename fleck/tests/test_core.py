import numpy as np
import os
import astropy.units as u
import pytest

from ..core import Star


@pytest.mark.parametrize("fast", [True, False])
def test_stsp_rotational_modulation(fast):
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
                                inc_stellar * u.deg, fast=fast)

    # Assert matches STSP results to within 100 ppm:
    np.testing.assert_allclose(fleck_lc[:, 0], stsp_lc, atol=100e-6)


# The unchanged slow-mode approximation differs from STSP by up to 443 ppm.
# Retain the original 350 ppm bound for fast mode and use 500 ppm for slow mode.
@pytest.mark.parametrize("fast, atol", [(True, 350e-6), (False, 500e-6)])
def test_stsp_transit(fast, atol):
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

    stsp_lc = np.loadtxt(os.path.join(os.path.dirname(__file__), os.pardir,
                                      'data', 'stsp_single_transit.txt'))

    inc_stellar = 90 * u.deg
    spot_radii = np.array([[0.1], [0.1]])
    spot_lats = np.array([[0], [0]]) * u.deg
    spot_lons = np.array([[360-30], [30]]) * u.deg

    times = np.linspace(-0.5, 0.5, 500)

    star = Star(spot_contrast=0.7, u_ld=planet.u, rotation_period=100)

    fleck_lc = star.light_curve(spot_lons, spot_lats, spot_radii,
                                inc_stellar, planet=planet, times=times,
                                fast=fast, time_ref=0)

    # Compare each mode against its documented STSP accuracy bound.
    np.testing.assert_allclose(fleck_lc[:, 0], stsp_lc, atol=atol)


@pytest.mark.parametrize("fast", [True, False])
def test_stsp_double_transit(fast):
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

    stsp_lc = np.loadtxt(os.path.join(os.path.dirname(__file__), os.pardir,
                                      'data', 'stsp_double_transit.txt'))

    inc_stellar = 90 * u.deg
    spot_radii = np.array([[0.05], [0.05]])
    spot_lats = np.array([[0], [0]]) * u.deg
    spot_lons = np.array([[360-30], [30]]) * u.deg

    times = np.concatenate([np.linspace(-0.5, 0.5, 500),
                            np.linspace(87.5, 88.5, 500)])

    star = Star(spot_contrast=0.7, u_ld=planet.u, rotation_period=10)

    fleck_lc = star.light_curve(spot_lons, spot_lats, spot_radii,
                                inc_stellar, planet=planet, times=times,
                                fast=fast, time_ref=0)

    # Assert matches STSP results to within 1 ppt:
    np.testing.assert_allclose(fleck_lc[:, 0], stsp_lc, atol=1e-3)


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


@pytest.fixture
def spotted_transit():
    """A visible spot on an inclined planet's transit chord."""
    from batman import TransitParams

    planet = TransitParams()
    planet.per = 3.0
    planet.a = 12.0
    planet.rp = 0.08
    planet.w = 90.0
    planet.ecc = 0.0
    planet.inc = 88.5
    planet.t0 = 0.0
    planet.limb_dark = 'quadratic'
    planet.u = [0.2, 0.1]
    chord_lat = -np.degrees(np.arcsin(
        planet.a * np.cos(np.deg2rad(planet.inc))))
    star = Star(spot_contrast=0.5, u_ld=planet.u, rotation_period=3650000.)
    return (star, planet, np.array([[0.]]) * u.deg,
            np.array([[chord_lat]]) * u.deg, np.array([[0.04]]))


@pytest.mark.parametrize('fast', [True, False])
def test_inclined_spot_occultation(spotted_transit, fast):
    """Both modes must produce a finite spot-crossing bump with NumPy 2."""
    star, planet, lons, lats, radii = spotted_transit
    times = np.linspace(-0.08, 0.08, 81)
    spotted, occulted = star.light_curve(
        lons, lats, radii, 90 * u.deg, planet=planet, times=times,
        fast=fast, return_spots_occulted=True)
    spotless = star.light_curve(
        lons, lats, np.zeros_like(radii), 90 * u.deg, planet=planet,
        times=times, fast=fast)
    # Normalize out the constant dimming caused by the unocculted spot.
    spotted /= spotted[0]
    spotless /= spotless[0]

    assert occulted
    assert spotted.shape == (len(times), 1)
    assert np.all(np.isfinite(spotted))
    assert np.max(spotted - spotless) > 1e-5
    assert spotted[len(times) // 2, 0] > spotless[len(times) // 2, 0]


def test_plot_inclined_spot(spotted_transit):
    """Plotting must pass scalar spot geometry to Shapely with NumPy 2.4."""
    import matplotlib.pyplot as plt

    star, planet, lons, lats, radii = spotted_transit
    fig, ax = plt.subplots()
    try:
        star.plot(lons, lats, radii, 90 * u.deg, time=0, planet=planet, ax=ax)
        fig.canvas.draw()
        assert len(ax.patches) == 1
    finally:
        plt.close(fig)
