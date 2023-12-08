from jax import jit, numpy as jnp, random, lax
from jax.tree_util import register_pytree_node_class

import numpy as np

import astropy.units as u
from jaxoplanet.core import kepler
from jaxoplanet.core.limb_dark import light_curve

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import to_hex

from scipy.stats import binned_statistic
from specutils import Spectrum1D

key = random.PRNGKey(0)

empty = jnp.array([])


@register_pytree_node_class
class ActiveStar:
    def __init__(
        self,
        times=empty,
        lon=empty,
        lat=empty,
        rad=empty,
        spectrum=empty,
        T_eff=None,
        temperature=empty,
        inclination=empty,
        wavelength=None,
        phot=None,
        u_ld=[0, 0],
        P_rot=3.3,
    ):
        self.times = jnp.array(times)
        self.lon = jnp.array(lon)
        self.lat = jnp.array(lat)
        self.rad = jnp.array(rad)
        self.spectrum = jnp.array(spectrum)
        self.T_eff = T_eff
        self.temperature = jnp.array(temperature)
        self.inclination = inclination
        self.wavelength = wavelength
        self.phot = phot
        self.u_ld = u_ld
        self.P_rot = P_rot

    def tree_flatten(self):
        children = (
            self.times,
            self.lon,
            self.lat,
            self.rad,
            self.spectrum,
            self.T_eff,
            self.temperature,
            self.inclination,
            self.wavelength,
            self.phot,
            self.u_ld,
            self.P_rot

        )
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @jit
    def rotation_model(self, f0=0, t0=0):
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

        contrast = self.spectrum / self.phot

        if contrast.ndim == 1:
            contrast = contrast[None, :]

        phase = jnp.expand_dims(2 * np.pi * (self.times - t0) / self.P_rot, [1, 2, 3])
        lon = jnp.expand_dims(self.lon, [0, 2, 3])
        lat = jnp.expand_dims(self.lat, [0, 2, 3])
        rad = jnp.expand_dims(self.rad, [0, 2, 3])
        contrast = jnp.expand_dims(contrast, [0, 3])
        inclination = jnp.expand_dims(jnp.asarray(self.inclination), [0, 1, 2])

        comp_inclination = np.pi / 2 - inclination
        phi = phase + lon + np.pi / 2

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
            self.limb_darkening(mu) / self.limb_darkening(1.0) *
            mask_behind_star,
            axis=1
        )

        f_S = (np.pi * rad ** 2 * jnp.sqrt(1 - rsq)) * (spot_position_z < 0).astype(int)

        return spot_model, f_S

    @jit
    def spot_coords(self, times=None, t0=0):
        contrast = self.spectrum / self.phot

        if contrast.ndim == 1:
            contrast = contrast[None, :]

        if times is None:
            times = self.times

        phase = jnp.expand_dims(2 * np.pi * (times - t0) / self.P_rot, [1, 2, 3])
        lon = jnp.expand_dims(self.lon, [0, 2, 3])
        lat = jnp.expand_dims(self.lat, [0, 2, 3])
        rad = jnp.expand_dims(self.rad, [0, 2, 3])
        contrast = jnp.expand_dims(contrast, [0, 3])
        inclination = jnp.expand_dims(jnp.asarray(self.inclination), [0, 1, 2])

        comp_inclination = np.pi / 2 - inclination
        phi = phase + lon + np.pi / 2

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

        rsq = spot_position_x ** 2 + spot_position_y ** 2

        major_axis = rad
        minor_axis = rad * jnp.sqrt(1 - rsq)
        angle = -jnp.degrees(jnp.arctan2(spot_position_y, spot_position_x))

        return spot_position_x, spot_position_y, spot_position_z, major_axis, minor_axis, angle

    def add_spot(self, lon, lat, rad, contrast=None, temperature=None, spectrum=None):

        if contrast is None and spectrum is None and temperature is not None:
            self.phot = self._blackbody(self.wavelength, self.T_eff)
            spectrum = self._blackbody(self.wavelength, temperature)

        for attr, new_value in zip("lon, lat, rad, spectrum, temperature".split(', '),
                                   [lon, lat, rad, spectrum, temperature]):

            prop = getattr(self, attr)

            if not hasattr(new_value, 'ndim'):
                new_value = jnp.array([new_value])

            if prop is not None:
                if prop.ndim > 1 or (len(prop) > 1 and len(prop) == len(new_value)):
                    new_value = jnp.vstack([prop, new_value])
                else:
                    new_value = jnp.concatenate([prop, new_value])

                setattr(self, attr, new_value)

    @jit
    def _blackbody(self, wavelength_meters, temperature):
        h = 6.62607015e-34  # J s
        c = 299792458.0  # m/s
        k_B = 1.380649e-23  # J/K

        return (
            2 * h * c ** 2 / jnp.power(wavelength_meters, 5) /
            jnp.expm1(h * c / (wavelength_meters * k_B * temperature))
        )

    @jit
    def limb_darkening(self, mu):
        return (
            1 / np.pi *
            (1 - self.u_ld[0] * (1 - mu) - self.u_ld[1] * (1 - mu) ** 2) /
            (1 - self.u_ld[0] / 3 - self.u_ld[1] / 6)
        )

    @jit
    def transit_model(self, t0, period, rp, a, inclination, f_S, omega=np.pi / 2, ecc=0):
        mean_anomaly = 2 * np.pi * (self.times - t0) / period
        true_anomaly = jnp.arctan2(*kepler(M=mean_anomaly, ecc=ecc))

        # winn 2011 eqn 1
        r = a * (1 - ecc ** 2) / (1 + ecc * jnp.cos(true_anomaly))

        # winn 2011 eqn 3-4
        X = -r * jnp.cos(omega + true_anomaly)
        Y = -r * jnp.sin(omega + true_anomaly) * jnp.cos(inclination)

        photosphere = (1 - f_S.sum(1, keepdims=True)) * self.phot[None, None, :, None]

        time_series_spectrum = (
            # photospheric component:
            photosphere[..., 0] +
            # sum of the active region components:
            jnp.sum(f_S[..., 0] * self.spectrum[None, ...], axis=1, keepdims=True)
        )
        contamination = 1 + (time_series_spectrum - self.phot[None, :]) / time_series_spectrum
        transit = light_curve(u=self.u_ld, r=rp, b=jnp.hypot(X, Y), order=4)

        # find where spots are occulted
        spot_position_x, spot_position_y, spot_position_z, major_axis, minor_axis, angle = self.spot_coords(t0=t0)
        planet_spot_distance = jnp.hypot(spot_position_y - X[:, None, None, None],
                                         spot_position_x - Y[:, None, None, None])
        occultation_possible = np.squeeze(
            (planet_spot_distance < (major_axis + rp)) &
            (spot_position_z < 0)
        )

        def time_step(
            carry, j, X=X, Y=Y, spot_position_y=spot_position_y,
            spot_position_x=spot_position_x, major_axis=major_axis,
            minor_axis=minor_axis, rp=rp, angle=angle, key=key,
            occultation_possible=occultation_possible
        ):
            def spot_step(
                carry, k, X=X[j], Y=Y[j], spot_position_y=spot_position_y[j, :, 0, 0],
                spot_position_x=spot_position_x[j, :, 0, 0],
                major_axis=major_axis[j, :, 0, 0],
                minor_axis=minor_axis[j, :, 0, 0], rp=rp, angle=angle[j, :, 0, 0], key=key,
                occultation_possible=occultation_possible[j]
            ):
                return carry, lax.cond(
                    occultation_possible[k],
                    lambda x: self.area_union_per_time(
                        x0_ellipse=jnp.squeeze(spot_position_y[k]),
                        y0_ellipse=jnp.squeeze(spot_position_x[k]),
                        x0_circle=X,
                        y0_circle=Y,
                        alpha=major_axis[k],
                        beta=minor_axis[k],
                        angle=angle[k],
                        radius=rp,
                        key=key
                    ), lambda x: 0.0, 0.0
                )

            return carry, lax.scan(
                spot_step, 0.0, jnp.arange(spot_position_x.shape[1])
            )[1]

        occultation = lax.scan(
            time_step, 0.0, jnp.arange(self.times.shape[0])
        )[1]

        contrast = self.spectrum / self.phot
        contaminated_transit = jnp.expand_dims(transit[:, None, None] / contamination, axis=3)

        occultation = (
                (1 - contrast[None, ..., None]) *
                occultation[..., None, None] * rp ** 2
        )

        transit_depth_norm = contaminated_transit / contaminated_transit.min(axis=0, keepdims=True)
        scaled_occultation = transit_depth_norm * occultation

        return scaled_occultation.sum(axis=1, keepdims=True) + contaminated_transit, rp ** 2 * contamination, X, Y

    @jit
    def area_union_per_time(
            self, x0_ellipse, y0_ellipse, x0_circle, y0_circle,
            alpha, beta, angle, radius, key=key, n=1_000
    ):
        # Monte Carlo sampling for points inside the planet's disk:
        key, subkey = random.split(key)
        theta_p = random.uniform(key, minval=0, maxval=2 * np.pi, shape=(n,))
        key, subkey = random.split(key)
        rad_p = random.uniform(key, minval=0, maxval=radius, shape=(n,))
        xp = rad_p * jnp.cos(theta_p) + x0_circle
        yp = rad_p * jnp.sin(theta_p) + y0_circle

        # find overlap between the planet and the elliptical region (projected circular spot)
        in_ellipse = jnp.hypot(
            ((xp - x0_ellipse) * jnp.cos(jnp.radians(angle)) +
             (yp - y0_ellipse) * jnp.sin(jnp.radians(angle))) / alpha,
            ((xp - x0_ellipse) * jnp.sin(jnp.radians(angle)) -
             (yp - y0_ellipse) * jnp.cos(jnp.radians(angle))) / beta
        ) < 1

        # ensure overlap only occurs on the stellar surface
        on_star = jnp.hypot(xp, yp) < 1

        return jnp.count_nonzero(in_ellipse & on_star) / n

    def plot_star(self, rp, a, ecc, inclination, t0=0, multiply_radii=1, ax=None):

        if ax is None:
            ax = plt.gca()

        log_temps = np.log10(self.temperature)

        temp_cmap = lambda x: to_hex(
            plt.cm.YlOrRd_r((np.log10(x) - min(log_temps)) / (max(log_temps) - min(log_temps)) * 0.6 + 0.4)
        )

        star = plt.Circle((0, 0), 1, color=to_hex(temp_cmap(self.T_eff)))
        ax.add_patch(star)
        ax.set(xlim=[-1.05, 1.05], ylim=[-1.05, 1.05])

        squeezed_coords = list(map(
            jnp.squeeze, self.spot_coords(times=jnp.array([t0]), t0=t0)
        ))
        for i, (x, y, z, _, _, angle) in enumerate(zip(*squeezed_coords)):
            if z < 0:
                rsq = x ** 2 + y ** 2

                short = np.sqrt(1 - rsq)
                angle = -np.degrees(np.arctan2(y, x))
                ell = Ellipse(
                    (y, x), width=multiply_radii * self.rad[i],
                    height=multiply_radii * self.rad[i] * short, angle=angle,
                    facecolor=temp_cmap(self.temperature[i]), edgecolor='k'
                )
                ax.add_patch(ell)

                # ax.annotate(f"{i+1}: {int(temp)} K", (y, x), va='center', ha='center', fontsize=6)

        ax.set_aspect('equal')

        b = (a * np.cos(inclination) * (1 - ecc ** 2) /
             (1 + ecc * np.sin(np.pi / 2)))

        planet_lower_extent = -b - rp
        planet_upper_extent = -b + rp
        ax.axhline(planet_lower_extent, color='gray', ls='--')
        ax.axhline(planet_upper_extent, color='gray', ls='--')
        ax.axis('off')

        return ax


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


def spectral_binning(y, all_x, all_y):
    """
    Spectral binning via trapezoidal approximation.
    """
    min_ind = np.argwhere(all_y == y[0])[0, 0]
    max_ind = np.argwhere(all_y == y[-1])[0, 0]
    if max_ind > min_ind and y.shape == all_x[min_ind:max_ind + 1].shape:
        return np.trapz(y, all_x[min_ind:max_ind + 1]) / (all_x[max_ind] - all_x[min_ind])
    return np.nan
