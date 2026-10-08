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
    active_star.lat = jnp.radians(jnp.array([lat1, lat2]))
    active_star.rad = jnp.array([rad1, rad2])
    active_star.spectrum = jnp.array([0.7, 0.7])

    fleck_lc = active_star.rotation_model(
        f0=1, t0_rot=0, u1=u_ld[0], u2=u_ld[1]
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
        lat=np.ravel(spot_lats.to_value(u.rad)),
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


@pytest.fixture
def star_inputs():
    return dict(times=jnp.linspace(-0.05, 0.05, 11), lon=[0.], lat=[0.],
                rad=[0.1], inclination=np.pi/2, wavelength=[1e-6],
                phot=[1.], P_rot=10.)


@pytest.mark.parametrize('contrast', [0.5, [0.5], np.array([0.5]),
                                      jnp.array(0.5), jnp.array([0.5])])
def test_contrast_inputs_produce_half_dark_spot(star_inputs, contrast):
    star = ActiveStar(**star_inputs, contrast=contrast)
    np.testing.assert_allclose(star.contrast, [[0.5]])
    flux = np.asarray(star.rotation_model(f0=1)[0])
    # At disk center a spot removes its area times its missing flux fraction.
    assert flux[len(flux)//2, 0, 0] == pytest.approx(1 - 0.1**2*0.5,
                                                   abs=1e-6)


@pytest.mark.parametrize('contrast, expected', [
    (0.5, [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),
    ([0.4, 0.7], [[0.4, 0.4, 0.4], [0.7, 0.7, 0.7]]),
    ([0.4, 0.5, 0.7], [[0.4, 0.5, 0.7], [0.4, 0.5, 0.7]]),
])
def test_contrast_broadcasts_over_spots_and_wavelengths(contrast, expected):
    star = ActiveStar(times=[0.], lon=[0., 0.2], lat=[0., 0.1],
                      rad=[0.1, 0.05], inclination=np.pi/2,
                      wavelength=[1e-6, 2e-6, 3e-6], phot=[2., 3., 4.],
                      contrast=contrast)
    np.testing.assert_allclose(star.contrast, expected)
    np.testing.assert_allclose(star.spectrum, np.asarray(expected)*[2., 3., 4.])
    assert star.rotation_model(f0=1)[0].shape == (1, 3, 1)


def test_add_first_spot_and_mix_temperature_and_contrast():
    star = ActiveStar(times=jnp.linspace(-0.05, 0.05, 11),
                      inclination=np.pi/2, wavelength=[1e-6, 2e-6],
                      T_eff=5000.)
    original_phot = np.asarray(star.phot).copy()
    star.add_spot(lon=0., lat=0., rad=0.1, contrast=0.5)
    star.add_spot(lon=0.2, lat=-0.1, rad=0.05, temperature=4000.)
    star.add_spot(lon=-0.2, lat=0.1, rad=0.05, spectrum=0.8*original_phot)

    assert star.n_spots == 3
    np.testing.assert_allclose(star.lon, [0., 0.2, -0.2])
    np.testing.assert_allclose(star.lat, [0., -0.1, 0.1])
    np.testing.assert_allclose(star.rad, [0.1, 0.05, 0.05])
    np.testing.assert_allclose(star.contrast[[0, 2], :], [[0.5, 0.5], [0.8, 0.8]])
    np.testing.assert_allclose(star.phot, original_phot)
    assert star.spectrum.shape == (3, 2)
    assert star.contrast.shape == (3, 2)
    assert np.isnan(np.asarray(star.temperature)[[0, 2]]).all()
    assert star.temperature[1] == 4000.
    assert np.isfinite(star.rotation_model(f0=1)[0]).all()


@pytest.mark.parametrize('construct_spot', [True, False])
def test_temperature_spot_uses_blackbody_photosphere(construct_spot):
    wavelength = np.array([1e-6, 2e-6])
    kwargs = dict(times=[0.], inclination=np.pi/2, wavelength=wavelength,
                  T_eff=5000.)
    if construct_spot:
        star = ActiveStar(**kwargs, lon=[0.], lat=[0.], rad=[0.1],
                          temperature=[4000.])
    else:
        star = ActiveStar(**kwargs)
        star.add_spot(lon=0., lat=0., rad=0.1, temperature=4000.)
    hc_over_k = 6.62607015e-34*299792458./1.380649e-23
    expected = (np.expm1(hc_over_k/(wavelength*5000.)) /
                np.expm1(hc_over_k/(wavelength*4000.)))
    np.testing.assert_allclose(star.contrast[0], expected, rtol=1e-5)
    assert np.all((star.contrast > 0) & (star.contrast < 1))


def test_explicit_photosphere_is_preserved(star_inputs):
    star = ActiveStar(**star_inputs, T_eff=5000., contrast=0.5)
    np.testing.assert_array_equal(star.phot, [1.])
    np.testing.assert_allclose(star.spectrum, [[0.5]])


@pytest.mark.parametrize('temperature', [None, 5000.])
@pytest.mark.parametrize('contrast', [0.5, 1.5])
def test_plot_contrast_only_spot_without_temperature(star_inputs, temperature,
                                                    contrast):
    import matplotlib.pyplot as plt

    star = ActiveStar(**star_inputs, contrast=contrast, T_eff=temperature)
    fig, ax = plt.subplots()
    try:
        star.plot_star(0., 0.1, 10., np.deg2rad(89), ax=ax, annotate=True)
        fig.canvas.draw()
        assert len(ax.patches) == 2
        assert ax.patches[1].get_alpha() == pytest.approx(0.5)
        assert ax.texts[0].get_text() == f'1: {contrast:.2f}'
    finally:
        plt.close(fig)


@pytest.mark.parametrize('temperatures', [[4000., 5000.], [5000., 5000.],
                                         [4000., 6000.]])
def test_plot_temperature_colors_follow_temperature(temperatures):
    import matplotlib.pyplot as plt

    star = ActiveStar(times=[0.], lon=[-0.2, 0.2], lat=[0., 0.],
                      rad=[0.1, 0.1], inclination=np.pi/2,
                      wavelength=[1e-6], phot=[1.], contrast=0.5,
                      T_eff=5000., temperature=temperatures)
    fig, ax = plt.subplots()
    try:
        star.plot_star(0., 0.1, 10., np.deg2rad(89), ax=ax, annotate=True)
        fig.canvas.draw()
        colors = np.array([patch.get_facecolor() for patch in ax.patches])
        assert np.isfinite(colors).all()
        if temperatures[0] == temperatures[1]:
            np.testing.assert_array_equal(colors[0], colors[1])
        else:
            assert not np.array_equal(colors[0], colors[1])
            assert not np.array_equal(colors[1], colors[2])
        assert len(ax.texts) == 2
    finally:
        plt.close(fig)


def test_contrast_is_differentiable(star_inputs):
    def center_flux(contrast):
        star = ActiveStar(**star_inputs, contrast=contrast)
        return star.rotation_model(f0=1)[0][5, 0, 0]

    assert float(jax.grad(center_flux)(0.5)) == pytest.approx(0.1**2, abs=1e-6)
