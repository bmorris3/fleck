import numpy as np
from scipy.stats import binned_statistic
from specutils import Spectrum1D
import astropy.units as u

from jax import jit, numpy as jnp


@jit
def limb_darkening(mu, u_ld):
    return (
        1 / np.pi *
        (1 - u_ld[0] * (1 - mu) - u_ld[1] * (1 - mu) ** 2) /
        (1 - u_ld[0] / 3 - u_ld[1] / 6)
    )


@jit
def rotation_model(phase, lon, lat, rad, contrast, inclination, f0=0, u_ld=[0, 0]):
    """
    Limits:
    lat: (0, pi)
    lon: (0, 2pi)
    rad: (0, None)
    contrast: (0, inf)
    inclination: (0, pi/2)

    broadcasting dimensions:
    0. phase
    1. spot location (lat, lon, rad)
    2. contrast/wavelength
    3. inclination
    """

    if len(contrast.shape) == 1:
        contrast = contrast[None, :]

    phase = jnp.expand_dims(phase, [1, 2, 3])
    lon = jnp.expand_dims(lon, [0, 2, 3])
    lat = jnp.expand_dims(lat, [0, 2, 3])
    rad = jnp.expand_dims(rad, [0, 2, 3])
    contrast = jnp.expand_dims(contrast, [0, 3])
    inclination = jnp.expand_dims(inclination, [0, 1, 2])

    comp_inclination = np.pi / 2 - inclination
    phi = phase - lon

    sin_lat = jnp.sin(lat)
    cos_lat = jnp.cos(lat)
    sin_c_inc = jnp.sin(comp_inclination)
    cos_c_inc = jnp.cos(comp_inclination)

    spot_position_x = (
        jnp.cos(phi - np.pi / 2) * sin_c_inc * sin_lat +
        cos_c_inc * cos_lat
    )
    spot_position_y = -jnp.sin(phi - np.pi / 2) * sin_lat
    spot_position_z = (
        cos_lat * sin_c_inc -
        jnp.sin(phi) * cos_c_inc * sin_lat
    )

    rsq = jnp.hypot(spot_position_x, spot_position_y)
    mu = jnp.sqrt(1 - rsq)
    mask_behind_star = jnp.where(
        spot_position_z < 0, mu, 0
    )

    # Morris 2020 Eqn 6-7
    spot_model = f0 - jnp.sum(
        rad ** 2 *
        (1 - contrast) *
        limb_darkening(mu, u_ld) / limb_darkening(1, u_ld) *
        mask_behind_star,
        axis=1
    )

    return spot_model


def spectral_binning(y, all_x, all_y):
    """
    Spectral binning via trapezoidal approximation.
    """
    min_ind = np.argwhere(all_y == y[0])[0, 0]
    max_ind = np.argwhere(all_y == y[-1])[0, 0]
    if max_ind > min_ind and y.shape == all_x[min_ind:max_ind + 1].shape:
        return np.trapz(y, all_x[min_ind:max_ind + 1]) / (all_x[max_ind] - all_x[min_ind])
    return np.nan


def bin_spectrum(spectrum, bins=None, log=True, min=None, max=None, **kwargs):
    """

    Bin a spectrum, with log-spaced frequency bins.

    Parameters
    ----------
    spectrum :
    log : bool
        If true, compute bin edges based on the log base 10 of
        the frequency.
    bins : int or ~numpy.ndarray
        Number of bins, or the bin edges

    Returns
    -------
    new_spectrum :
    """
    nirspec_wl_range = (spectrum.wavelength > min) & (spectrum.wavelength < max)

    wavelength = spectrum.wavelength[nirspec_wl_range]
    flux = spectrum.flux[nirspec_wl_range]

    if log:
        wl_axis = np.log10(wavelength.to(u.um).value)
    else:
        wl_axis = wavelength.to(u.um).value

    # Bin the power spectrum:
    bs = binned_statistic(
        wl_axis, flux.value,
        statistic=lambda y: spectral_binning(
            y, all_x=wl_axis, all_y=flux.value
        ),
        bins=bins
    )
    if log:
        wl_bins = 10 ** (
            0.5 * (bs.bin_edges[1:] + bs.bin_edges[:-1])
        ) * u.um
    else:
        wl_bins = (
            0.5 * (bs.bin_edges[1:] + bs.bin_edges[:-1])
        ) * u.um
    nans = np.isnan(bs.statistic)
    interp_fluxes = bs.statistic.copy()
    interp_fluxes[nans] = np.interp(wl_bins[nans], wl_bins[~nans], bs.statistic[~nans])
    return Spectrum1D(
        flux=interp_fluxes * flux.unit, spectral_axis=wl_bins, meta=spectrum.meta
    )