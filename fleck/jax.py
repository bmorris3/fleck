from jax import jit, numpy as jnp, random, lax, vmap
from jax.tree_util import register_pytree_node_class
from jax.scipy.integrate import trapezoid

import numpy as np

import astropy.units as u
import jaxoplanet.core

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import to_hex

from scipy.stats import binned_statistic
from specutils import Spectrum1D

key = random.PRNGKey(0)

empty = jnp.array([])

__all__ = [
    'ActiveStar', 'bin_spectrum'
]


@register_pytree_node_class
class ActiveStar:
    """
    Model for a star with active regions and rotation, with
    optional planetary transit models and spot occultations.
    """

    # Number of Monte Carlo samples to use when computing planet+spot overlap
    n_mc = 10_000
    key = random.PRNGKey(0)  # random key seed

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
        P_rot=3.3,
        contrast=empty,
    ):
        """
        Parameters
        ----------
        times : array
            Times at which to compute the flux
        lon : array
            Active region longitudes in radians on (0, 2pi)
        lat : array
            Active region latitudes in radians on (-pi/2, pi/2)
        rad : array
            Active region radii in units of stellar radii
        spectrum : array
            One spectrum for each active region
        T_eff : array
            Effective temperature of the photosphere
        temperature : array
            Effective temperature of the active regions
        inclination : array
            Stellar inclination [radians]
        wavelength : array
            Wavelength for each flux observation in ``phot`` [meters]
        phot : array
            Photospheric flux at each ``wavelength``.
        P_rot : float
            Stellar rotation period
        contrast : float or array
            Ratio of the active region flux to the photospheric
            flux at each ``wavelength``. If provided, this will be
            used directly; otherwise it is computed from ``spectrum``
            and ``phot`` when possible.
        """
        self.times = jnp.array(times)
        self.lon = jnp.array(lon)
        self.lat = jnp.array(lat)
        self.rad = jnp.array(rad)
        self.spectrum = jnp.array(spectrum)
        self.T_eff = T_eff
        self.temperature = jnp.array(temperature)
        self.inclination = inclination
        # Allow wavelength/phot to remain optional, but store them as
        # JAX arrays when provided.
        self.wavelength = None if wavelength is None else jnp.array(wavelength)
        self.phot = None if phot is None else jnp.array(phot)
        self.P_rot = P_rot

        # Raw contrast as passed by the user (may be scalar, 1D, 2D, or
        # empty). We normalise shapes after dimension inference.
        if contrast is None or getattr(contrast, "size", 0) == 0:
            self.contrast = jnp.array([])
        else:
            self.contrast = jnp.array(contrast)

        # ------------------------------------------------------------------
        # Dimension bookkeeping: infer n_times, n_spots, n_lambda
        # ------------------------------------------------------------------
        self.n_times = int(self.times.size)
        self.n_spots = int(self.rad.size)
        self.n_lambda = None

        if (
            self.wavelength is not None and
            getattr(self.wavelength, "size", 0) != 0
        ):
            # 1) Prefer the explicit wavelength grid, if provided
            self.n_lambda = int(self.wavelength.size)

        elif getattr(self.spectrum, "size", 0) != 0:
            # 2) Otherwise, use the spectrum shape

            if self.spectrum.ndim == 1:
                # Treat a 1D spectrum as (n_lambda,)
                self.n_lambda = int(self.spectrum.shape[0])
            else:
                # Assume the last axis is wavelength
                self.n_lambda = int(self.spectrum.shape[-1])

        elif getattr(self.phot, "size", 0) != 0:
            # 3) Otherwise, use the photosphere

            self.n_lambda = self.phot.size

        elif getattr(self.contrast, "size", 0) != 0:
            # 4) Finally, fall back to the contrast shape, using n_spots to
            # break degeneracies between (n_spots,) and (n_lambda,)

            c = self.contrast
            if c.ndim == 0:
                self.n_lambda = 1
            elif c.ndim == 1:
                if self.n_spots > 0 and c.shape[0] == self.n_spots:
                    # One contrast per spot, same for all wavelengths
                    self.n_lambda = 1
                else:
                    # One contrast per wavelength bin, same for all spots
                    self.n_lambda = int(c.shape[0])
            elif c.ndim == 2:
                # Last axis is wavelength
                self.n_lambda = int(c.shape[-1])

        else:
            # Absolute last resort: a single "white-light" wavelength bin
            self.n_lambda = 1

        # Normalise the supplied contrast argument into something at least
        # 2D where possible.
        if self.contrast.size != 0:
            if self.contrast.ndim == 0:
                # Single scalar contrast: broadcast to all spots in a single
                # wavelength bin.
                self.contrast = self.contrast * jnp.ones((self.n_spots, 1))
            elif self.contrast.ndim == 1:
                if self.n_spots > 0 and self.contrast.shape[0] == self.n_spots:
                    # Per-spot contrast for a single wavelength bin.
                    self.contrast = self.contrast[:, None]
                else:
                    # Per-wavelength contrast for a single spot.
                    self.contrast = self.contrast[None, :]
            # If contrast is already 2D+ we assume the user passed
            # (n_spots, n_lambda) and leave it as-is.

        # If we have no phot, set it to unity
        self.phot = (jnp.asarray(self.phot) if self.phot is not None
                     else jnp.ones((self.n_lambda,)))

        # If an explicit contrast was not provided but we do have a
        # spectrum and photosphere, derive the contrast now.
        if self.contrast.size == 0 and self.spectrum.size != 0:
            derived_contrast = self.spectrum / self.phot[None, :]
            if derived_contrast.ndim == 1:
                derived_contrast = derived_contrast[None, :]
            self.contrast = derived_contrast

        # If we have a contrast and a photosphere but no spectrum,
        # construct a spectrum that is consistent with both.
        if self.spectrum.size == 0 and self.contrast.size != 0:
            self.spectrum = self.contrast * self.phot[None, :]

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
            self.P_rot,
            self.contrast,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @jit
    def rotation_model(self, f0=0, t0_rot=0, u1=0, u2=0):
        """
        Spectrophotometry of stellar rotation.

        Parameters
        ----------
        f0 : float
            Baseline flux of an unspotted star (usually zero or one)
        t0_rot : float
            Zero-point reference time for stellar rotation

        Returns
        -------
        spot_model : array
            Relative flux as a function of time and wavelength
        """
        (
            spot_position_x, spot_position_y, spot_position_z,
            major_axis, minor_axis, angle, rad, contrast
        ) = self.spot_coords(t0_rot=t0_rot)

        rsq = spot_position_x ** 2 + spot_position_y ** 2
        mu = jnp.sqrt(1 - rsq)
        mask_behind_star = jnp.where(
            spot_position_z < 0, mu, 0
        )

        radial_coord = 1 - jnp.geomspace(1e-5, 1, 100)[::-1]
        unspotted_total_flux = trapezoid(
            y=(
                2 * jnp.pi * radial_coord *
                self.limb_darkening(radial_coord, u1, u2)
            ),
            x=radial_coord
        )

        # Morris 2020 Eqn 6-7
        spot_model = f0 - jnp.sum(
            jnp.pi * rad ** 2 *
            (1 - contrast) *
            self.limb_darkening(mu, u1, u2) *
            mask_behind_star,
            axis=1
        ) / unspotted_total_flux
        f_S = rad ** 2 * mu * (spot_position_z < 0).astype(int)

        return spot_model, f_S

    @jit
    def spot_coords(self, times=None, t0_rot=0):
        """
        Compute the spatial coordinates and projected dimensions of
        active regions.

        Parameters
        ----------
        times : array
            Times on which to compute spectrophotometry
        t0_rot : float
            Zero-point reference time for stellar rotation

        Returns
        -------
        spot_position_x : array
            x-position of the active region in the observer oriented
            coordinate system [1]_.
        spot_position_y : array
            y-position of the active region in the observer oriented
            coordinate system [1]_.
        spot_position_z : array
            y-position of the active region in the observer oriented
            coordinate system [1]_.
        major_axis : array
            Apparent semimajor axis of the circular active region, which is
            elliptical when projected active onto the sky plane (in general)
        minor_axis : array
            Apparent semiminor axis of the circular active region, which is
            elliptical when projected active onto the sky plane (in general)
        angle : array
            Angle between the +x-axis and the projected active region's
            semimajor axis
        rad : array
            Active region radius [stellar radii]
        contrast: array
            Ratio of the active region spectrum and the photosphere spectrum

        References
        ----------
        .. [1]  Fabrycky & Winn (2009) https://arxiv.org/abs/0902.0737
        """
        if times is None:
            times = self.times

        # Use the dimensions inferred in __init__
        n_spots = int(self.n_spots)
        n_lambda = int(self.n_lambda)

        # ------------------------------------------------------------------
        # Build a per-spot, per-wavelength contrast array of shape
        # (n_spots, n_lambda), then broadcast to (1, n_spots, n_lambda, 1)
        # for use with the f_S tensor in transit_model.
        # ------------------------------------------------------------------
        if self.contrast.size != 0:
            base_contrast = jnp.asarray(self.contrast)
        elif self.spectrum.size != 0:
            base_contrast = jnp.asarray(self.spectrum) / self.phot[None, :]
        else:
            # No contrast information available.
            # Default to maximum contrast: spot == 0
            base_contrast = jnp.zeros((n_spots, n_lambda))

        # Normalize to exactly (n_spots, n_lambda)
        if n_spots == 0:
            # No spots: keep a well-defined empty (0, n_lambda) array
            # to avoid ambiguous reshapes.
            base_contrast = jnp.zeros((0, n_lambda))
        elif base_contrast.ndim == 0:
            base_contrast = jnp.full((n_spots, n_lambda), base_contrast)
        elif base_contrast.ndim == 1:
            if base_contrast.shape[0] == n_spots:
                # One contrast per spot, broadcast over wavelength.
                base_contrast = base_contrast[:, None]*jnp.ones((1, n_lambda))
            else:
                # One contrast per wavelength, broadcast over spots.
                base_contrast = base_contrast[None, :]*jnp.ones((n_spots, 1))
        elif base_contrast.shape[0] == 1 and n_spots > 1:
            # Single-spot input, broadcast along spot axis first.
            base_contrast = jnp.broadcast_to(
                base_contrast, (n_spots,)+base_contrast.shape[1:])

        # Final contrast tensor used downstream:
        # shape: (1, n_spots, n_lambda, 1)
        contrast = base_contrast[None, :, :, None]

        """
        Limits:
        lat: (-pi/2, pi/2)
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

        phase = jnp.expand_dims(2 * jnp.pi * (times - t0_rot) / self.P_rot,
                                [1, 2, 3])
        lon = jnp.expand_dims(self.lon, [0, 2, 3])
        lat = jnp.expand_dims(self.lat, [0, 2, 3])
        rad = jnp.expand_dims(self.rad, [0, 2, 3])
        inclination = jnp.expand_dims(jnp.asarray(self.inclination), [0, 1, 2])

        comp_inclination = jnp.pi / 2 - inclination
        phi = jnp.pi / 2 - phase - lon

        sin_lat = jnp.sin(lat)
        cos_lat = jnp.cos(lat)
        sin_c_inc = jnp.sin(comp_inclination)
        cos_c_inc = jnp.cos(comp_inclination)

        spot_position_x = (
            jnp.cos(phi - jnp.pi / 2) * sin_c_inc * cos_lat +
            cos_c_inc * sin_lat
        )
        spot_position_y = -jnp.sin(phi - jnp.pi / 2) * cos_lat
        spot_position_z = (
            sin_lat * sin_c_inc -
            jnp.sin(phi) * cos_c_inc * cos_lat
        )

        rsq = spot_position_x ** 2 + spot_position_y ** 2

        major_axis = rad
        minor_axis = rad * jnp.sqrt(1 - rsq)
        angle = -jnp.degrees(jnp.arctan2(spot_position_y, spot_position_x))

        return (
            spot_position_x, spot_position_y, spot_position_z,
            major_axis, minor_axis, angle, rad, contrast
        )

    def add_spot(self, lon, lat, rad, contrast=None, temperature=None,
                 spectrum=None):
        """
        Add an active region to the stellar model.

        Parameters
        ----------
        lon : float
            Active region longitudes in radians on (0, 2pi)
        lat : float
            Active region latitudes in radians on (-pi/2, pi/2)
        rad : float
            Active region radii in units of stellar radii
        contrast : float or array
            Ratio of the active region's flux to the photospheric
            flux at each ``ActiveStar.wavelength``.
        temperature : float
            Effective temperature of the active region. If provided
            (and neither ``contrast`` nor ``spectrum`` are given),
            a blackbody spectrum will be computed.
        spectrum : float or array
            The spectrum of the active region on the same wavelength
            grid as ``ActiveStar.phot``.
        """
        if contrast is not None:
            # User-supplied contrast: convert to spectrum
            contrast = jnp.array(contrast)
            if contrast.ndim == 0:
                # Scalar contrast: broadcast to all wavelengths
                contrast = contrast[None] * jnp.ones(self.n_lambda)
            elif contrast.ndim == 1:
                # 1D contrast are defined on the wavelength grid
                contrast = contrast[None, :]
            # Derive the spectrum from the contrast
            spectrum = contrast * self.phot[None, :]
        elif spectrum is not None:
            # User-supplied spectrum: ensure correct shape
            spectrum = jnp.array(spectrum)
            if spectrum.ndim == 0:
                # Scalar spectrum: broadcast to all wavelengths
                spectrum = spectrum[None] * jnp.ones(self.n_lambda)
            elif spectrum.ndim == 1:
                # 1D spectra are defined on the wavelength grid
                spectrum = spectrum[None, :]
            # Derive the contrast from the spectrum
            contrast = spectrum / self.phot[None, :]
        elif spectrum is None and temperature is not None:
            # No contrast or spectrum, but a temperature: compute a blackbody
            spectrum = self._blackbody(self.wavelength, temperature)
            contrast = spectrum / self.phot[None, :]
            if contrast.ndim == 1:
                contrast = contrast[None, :]

        for attr, new_value in zip(
            "lon, lat, rad, spectrum, temperature, contrast".split(', '),
            [lon, lat, rad, spectrum, temperature, contrast],
        ):

            prop = getattr(self, attr)

            if not hasattr(new_value, 'ndim'):
                new_value = jnp.array([new_value])

            if prop is not None and getattr(prop, "size", 0) != 0:
                if prop.ndim > 1 or (
                    len(prop) > 1 and len(prop) == len(new_value)
                ):
                    new_value = jnp.vstack([prop, new_value])
                else:
                    new_value = jnp.concatenate([prop, new_value])

                setattr(self, attr, new_value)

        # Update n_spots
        self.n_spots += 1

    @jit
    def _blackbody(self, wavelength_meters, temperature):
        """
        Compute a blackbody spectrum.
        """
        h = 6.62607015e-34  # J s
        c = 299792458.0  # m/s
        k_B = 1.380649e-23  # J/K

        return (
            2 * h * c ** 2 / jnp.power(wavelength_meters, 5) /
            jnp.expm1(h * c / (wavelength_meters * k_B * temperature))
        )

    @jit
    def limb_darkening(self, mu, u1, u2):
        """
        Compute quadratic limb darkening as a function of :math:`\\mu`.
        """
        return (
            1 / jnp.pi *
            (1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2) /
            (1 - u1 / 3 - u2 / 6)
        )

    @jit
    def transit_model(self, t0, period, rp, a, inclination,
                      omega=jnp.pi/2, ecc=0., f0=1., t0_rot=0.,
                      u1=0., u2=0.):
        """
        Compute spectrophotometry with rotation and a planetary transit.

        The transit is computed with ``jaxoplanet`` for a star with
        quadratic limb darkening.

        Parameters
        ----------
        t0 : float
            Mid-transit time
        period : float
            Orbital period of the transiting planet
        rp : float or array
            Exoplanet radius in units of stellar radii for each wavelength
        a : float
            Planetary semi-major axis in units of stellar radii
        inclination : float
            Planetary orbital inclination [radians]
        omega : float
            Argument of periapse [radians], default is :math:`\\pi/2`.
        ecc : float
            Orbital eccentricity, default is zero.
        f0 : float
            Out-of-transit flux for an unspotted star, default is one.
        t0_rot : float
            Zero-point in time for stellar rotation, default is zero
        u1 : float or array
            Limb-darkening parameter :math:`u_1` for each wavelength
        u2 : float or array
            Limb-darkening parameter :math:`u_2` for each wavelength

        Returns
        -------
        lc : array
            Flux as a function of time and wavelength
        apparent_rprs2 : array
            The apparent squared ratio of planet-to-star radius with stellar
            spectral contamination by active regions
         X : array
            x-position of the planet in the observer oriented coordinate
            system [1]_.
         Y : array
            y-position of the planet in the observer oriented coordinate
            system [1]_.

        References
        ----------
        .. [1] Fabrycky & Winn (2009) https://arxiv.org/abs/0902.0737
        """
        # Photosphere: ensure a 1D array of length n_lambda
        n_lambda = int(self.n_lambda)
        phot = (
            jnp.asarray(self.phot)
            if self.phot is not None
            else jnp.ones((n_lambda,))
        )

        # Limb-darkening coefficients
        u1 = jnp.atleast_1d(u1)
        u2 = jnp.atleast_1d(u2)
        u_ld = jnp.column_stack([u1, u2])

        # handle the out-of-transit spectroscopic rotational modulation:
        (
            spot_position_x, spot_position_y, spot_position_z,
            major_axis, minor_axis, angle, rad, contrast
        ) = self.spot_coords(t0_rot=t0_rot)

        rsq = spot_position_x ** 2 + spot_position_y ** 2
        mu = jnp.sqrt(1. - rsq)
        mask_behind_star = jnp.where(
            spot_position_z < 0., mu, 0.
        )

        radial_coord = 1. - jnp.geomspace(1e-5, 1., 100)[::-1]

        unspotted_total_flux = trapezoid(
            y=(
                2. * jnp.pi * radial_coord[:, None] *
                self.limb_darkening(
                    radial_coord[:, None], *u_ld.T
                )
            ).T,
            x=radial_coord
        )

        limb_dark = self.limb_darkening(
            mu,
            u1=u1[None, None, :, None],
            u2=u2[None, None, :, None]
        )

        # Morris 2020 Eqn 6-7
        out_of_transit = f0 - jnp.sum(
            jnp.pi * rad ** 2 *
            (1. - contrast) *
            limb_dark *
            mask_behind_star /
            unspotted_total_flux[None, None, :, None],
            axis=1
        )

        f_S = rad ** 2 * mu * (spot_position_z < 0.).astype(int)

        # ------------------------------------------------------------------
        # Orbital geometry (independent of spots)
        # ------------------------------------------------------------------
        ecc = jnp.asarray(ecc)
        omega = jnp.asarray(omega)

        # Index of time sample closest to mid-transit
        t_ind = jnp.argmin(jnp.abs(self.times - t0))

        # True anomaly at (inferior) conjunction in this geometry
        f_tr = 0.5 * jnp.pi - omega

        # Convert f_tr -> E_tr -> M_tr:
        #   tan(f/2) = sqrt((1+e)/(1-e)) * tan(E/2)
        #   M = E - e sin(E)
        sqrt_factor = jnp.sqrt((1.0 - ecc) / (1.0 + ecc + 1e-15))
        E_tr = 2.0 * jnp.arctan(jnp.tan(0.5 * f_tr) * sqrt_factor)
        M_tr = E_tr - ecc * jnp.sin(E_tr)

        # Mean anomaly as a function of time with t0 at mid-transit
        mean_anomaly = (
            2.0 * jnp.pi * (self.times - t0) / period + M_tr
        )
        true_anomaly = jnp.arctan2(
            *jaxoplanet.core.kepler(M=mean_anomaly, ecc=ecc)
        )

        # Winn 2011 Eqn 1: sky-projected star-planet distance
        r = a * (1. - ecc ** 2) / (1. + ecc * jnp.cos(true_anomaly))

        # Winn 2011 Eqn 3-4: sky-projected coordinates
        X = -r * jnp.cos(omega + true_anomaly)
        Y = -r * jnp.sin(omega + true_anomaly) * jnp.cos(inclination)

        # --------------------------------------------------------------
        # Enforce that the transit chord lies below the stellar equator
        # in the (X, Y) plane used for spot occultations:
        #
        # Choose the branch with Y(t0) <= 0 by reflecting across the
        # equator if necessary. This does NOT change b(t) or the
        # light curve because b = hypot(X, Y) is sign-invariant.
        # --------------------------------------------------------------
        Y_mid = Y[t_ind]
        sign_Y = jnp.where(Y_mid > 0., -1., 1.)
        Y = sign_Y * Y

        # Total fractional coverage by *any* spot
        coverage_sum = f_S[..., 0].sum(axis=1)
        # (n_times, 1)

        # Spot-weighted contrast
        spot_weighted_contrast = jnp.sum(
            f_S * contrast, axis=1
        )[..., 0]
        # (n_times, n_lambda)

        flux_factor = (1.0 - coverage_sum) + spot_weighted_contrast
        time_series_spectrum = phot[None, :] * flux_factor
        # (n_times, n_lambda)

        rp = jnp.broadcast_to(rp, (n_lambda,))
        # (n_lambda,)

        # Broadcast limb-darkening if needed
        u_ld = u_ld * jnp.ones((rp.shape[0], 1))
        # (n_lambda, 2)

        # Uncontaminated transit model
        transit = vmap(
            lambda rp, u: jaxoplanet.core.light_curve(
                u=u, b=jnp.hypot(X, Y), r=rp
            ), in_axes=0, out_axes=1
        )(rp, u_ld)

        contaminated_transit = (
            time_series_spectrum - jnp.abs(transit) * phot[None, :]
        ) / time_series_spectrum

        uncontaminated_max_depth = - transit[t_ind]
        contaminated_max_depth = (
            contaminated_transit.max(0) - contaminated_transit[t_ind]
        ) / contaminated_transit.max(0)

        depth_ratio = contaminated_max_depth / uncontaminated_max_depth
        apparent_rprs2 = rp ** 2 * depth_ratio

        # Planet–spot distance in the (X, Y) plane
        planet_spot_distance = jnp.hypot(
            spot_position_y - X[:, None, None, None],
            spot_position_x - Y[:, None, None, None],
        )
        occultation_possible = jnp.squeeze(
            (planet_spot_distance < (major_axis + rp.mean())) &
            (spot_position_z < 0.)
        )

        @jit
        def time_step(
            carry, j, X=X, Y=Y, spot_position_y=spot_position_y,
            spot_position_x=spot_position_x, major_axis=major_axis,
            minor_axis=minor_axis, rp=rp.mean(), angle=angle,
            occultation_possible=occultation_possible
        ):
            return carry, lax.cond(
                jnp.any(occultation_possible[j]),
                lambda *args: self._area_union_per_time(
                    x0_ellipse=spot_position_y[j],
                    y0_ellipse=spot_position_x[j],
                    x0_circle=X[j],
                    y0_circle=Y[j],
                    alpha=jnp.squeeze(major_axis[j]),
                    beta=jnp.squeeze(minor_axis[j]),
                    angle=jnp.squeeze(angle[j]),
                    radius=rp,
                    occultation_possible=occultation_possible[j],
                ),
                lambda *args: jnp.zeros((spot_position_x.shape[1], self.n_mc),
                                        dtype=bool),
            )

        occultation_per_time_per_spot_per_mc_sample = lax.scan(
            time_step, 0.0, jnp.arange(self.times.shape[0])
        )[1]
        # (n_times, n_spots, n_mc_samples)

        frac_occulted_per_time_per_spot = jnp.count_nonzero(
            occultation_per_time_per_spot_per_mc_sample, axis=2
        ) / self.n_mc

        occultation = (
            (1. - contrast) *
            jnp.expand_dims(frac_occulted_per_time_per_spot, axis=(2, 3))
        )
        scaled_occultation = ((1. - contaminated_transit)
                              * jnp.sum(occultation, axis=1)[..., 0])

        spectrum_at_transit = time_series_spectrum[t_ind]

        return (
            out_of_transit[..., 0]*(contaminated_transit+scaled_occultation),
            apparent_rprs2, X, Y,
            spectrum_at_transit
        )

    @jit
    def _area_union_per_time(
        self, x0_ellipse, y0_ellipse, x0_circle, y0_circle,
        alpha, beta, angle, radius, occultation_possible,
    ):
        # Monte Carlo sampling for points inside the planet's disk:
        key, subkey = random.split(self.key)
        theta_p = random.uniform(key, minval=0, maxval=2 * jnp.pi,
                                 shape=(self.n_mc,))
        key, subkey = random.split(key)
        rad_p = radius * random.uniform(subkey, minval=0, maxval=1,
                                        shape=(self.n_mc,)) ** 0.5
        xp = rad_p * jnp.cos(theta_p) + x0_circle
        yp = rad_p * jnp.sin(theta_p) + y0_circle

        # ensure overlap only occurs on the stellar surface
        on_star = jnp.hypot(xp, yp) < 1

        # ensure the one-spot case is indexed correctly below:
        x0_ellipse = jnp.atleast_1d(x0_ellipse)
        y0_ellipse = jnp.atleast_1d(y0_ellipse)
        alpha = jnp.atleast_1d(alpha)
        beta = jnp.atleast_1d(beta)
        angle = jnp.atleast_1d(angle)

        # If there are no active regions (no spots), return an empty
        # (0, n_mc) array and skip the per-spot scan entirely. This
        # prevents index errors when `occultation_possible` is empty.
        if (
            x0_ellipse.size == 0 or
            getattr(occultation_possible, "size", 0) == 0
        ):
            return jnp.zeros((0, self.n_mc), dtype=bool)

        @jit
        def find_overlap(k):
            # find overlap between the planet and the elliptical region
            # (projected circular spot)
            in_ellipse = jnp.hypot(
                ((xp - x0_ellipse[k]) * jnp.cos(jnp.radians(angle[k])) +
                 (yp - y0_ellipse[k]) * jnp.sin(jnp.radians(angle[k]))
                 ) / alpha[k],
                ((xp - x0_ellipse[k]) * jnp.sin(jnp.radians(angle[k])) -
                 (yp - y0_ellipse[k]) * jnp.cos(jnp.radians(angle[k]))
                 ) / beta[k]
            ) < 1

            return in_ellipse & on_star

        @jit
        def spot_step(carry, k, occultation_possible=occultation_possible):
            # where occultations are possible, compute the overlap
            occultation_possible = jnp.atleast_1d(occultation_possible)

            return carry, lax.cond(
                occultation_possible[k],
                lambda *args: jnp.squeeze(find_overlap(k)),
                lambda *args: jnp.zeros(self.n_mc, dtype=bool)
            )

        monte_carlo_occulted_area = lax.scan(
            spot_step, 0, jnp.arange(x0_ellipse.shape[0]))[1]

        return monte_carlo_occulted_area

    def plot_star(self, t0, rp, a, inclination,
                  ecc=0, omega=jnp.pi/2, t0_rot=0, multiply_radii=1,
                  ax=None, annotate=False):
        """
        Plot a 2D representation of the star and transit chord.

        Parameters
        ----------
        t0 : float
            Mid-transit time
        rp : float
            Exoplanet radius in units of stellar radii
        a : float
            Planetary semi-major axis in units of stellar radii
        inclination : float
            Planetary orbital inclination [radians]
        ecc : float
            Orbital eccentricity, default is zero.
        omega : float
            Argument of periapse [radians], default is :math:`\\pi/2`.
        t0_rot : float
            Zero-point in time for stellar rotation, default is zero
        multiply_radii : float
            Visually represent scaled-up active regions where the radii are
            increased by factor ``multiply_radii``, default is one.
        ax : matplotlib.axes.Axes
            Add the visualization to this matplotlib axis
        annotate : bool
            Add a text label with active region indices and temperatures
            to the visualization
        """

        if ax is None:
            ax = plt.gca()

        if self.temperature is not None:
            log_temps = np.log10(self.temperature)

        def temp_cmap(x):
            return to_hex(
                plt.cm.YlOrRd_r(
                    (np.log10(x) - min(log_temps)) /
                    (max(log_temps) - min(log_temps)) * 0.6 + 0.4
                )
            )

        if self.T_eff is None:
            color = 'white'
        else:
            color = to_hex(temp_cmap(self.T_eff))

        star = plt.Circle((0, 0), 1, facecolor=color, edgecolor='k')
        ax.add_patch(star)
        ax.set(xlim=[-1.05, 1.05], ylim=[-1.05, 1.05])

        (
            spot_position_x,
            spot_position_y,
            spot_position_z,
            _,
            _,
            angle,
            _,
            contrast,
        ) = self.spot_coords(times=jnp.array([t0, ]), t0_rot=t0_rot)

        # Slice out that time and drop the singleton trailing dims.
        # After this, each has shape (n_spots,).
        spot_x = spot_position_x[0, :, 0, 0]
        spot_y = spot_position_y[0, :, 0, 0]
        spot_z = spot_position_z[0, :, 0, 0]
        # contrast is usually (1, n_spots, n_lambda, 1)
        # for plotting, pick λ = 0
        contr = contrast[0, :, 0, 0]

        # Convert to NumPy for plotting (avoids JAX tracing issues)
        spot_x_np = np.asarray(spot_x)
        spot_y_np = np.asarray(spot_y)
        spot_z_np = np.asarray(spot_z)
        contr_np = np.asarray(contr)

        for i in range(self.n_spots):
            x = spot_x_np[i]
            y = spot_y_np[i]
            z = spot_z_np[i]
            c = contr_np[i]

            if len(self.temperature) == 0:
                color = 'k'
                alpha = 1-c
            else:
                color = temp_cmap(self.temperature[i])
                alpha = 1

            if z < 0:
                rsq = x ** 2 + y ** 2

                short = np.sqrt(1 - rsq)
                angle = -np.degrees(np.arctan2(y, x))
                ell = Ellipse(
                    (y, x), width=multiply_radii * 2 * self.rad[i],
                    height=multiply_radii * 2 * self.rad[i] * short,
                    angle=angle, facecolor=color, alpha=alpha, edgecolor='k'
                )
                ax.add_patch(ell)

                if annotate and len(self.temperature) == 0:
                    ax.annotate(
                        f"{i+1}: {c:.2f}", (y, x),
                        va='center', ha='center', fontsize=6
                    )
                elif annotate:
                    ax.annotate(
                        f"{i+1}: {int(self.temperature[i])} K", (y, x),
                        va='center', ha='center', fontsize=6
                    )

        ax.set_aspect('equal')

        b = (a * np.cos(inclination) * (1 - ecc ** 2) /
             (1 + ecc * np.sin(omega)))

        if hasattr(rp, '__len__'):
            rp = rp.mean()

        planet_lower_extent = -b - rp
        planet_upper_extent = -b + rp
        ax.axhline(planet_lower_extent, color='gray', ls='--')
        ax.axhline(planet_upper_extent, color='gray', ls='--')
        ax.axis('off')

        return ax

    @jit
    def rotation_spectrum(self, t0_rot=0):
        """
        Compute spectrophotometry during a rotation.

        Parameters
        ----------
        t0_rot : float
            Zero-point in time for stellar rotation, default is zero

        Returns
        -------
        time_series_spectrum : array
            Flux as a function of time and wavelength. Flux units are
            the same as the units for the input spectra.
        """
        # handle the out-of-transit spectroscopic rotational modulation:
        (
            spot_position_x, spot_position_y, spot_position_z,
            major_axis, minor_axis, angle, rad, contrast
        ) = self.spot_coords(t0_rot=t0_rot)

        rsq = spot_position_x ** 2 + spot_position_y ** 2
        mu = jnp.sqrt(1 - rsq)
        f_S = rad ** 2 * mu * (spot_position_z < 0).astype(int)
        photosphere = (1 - f_S[..., 0].sum(axis=1)) * self.phot[None, :]

        spot_coverages, spot_spectra = jnp.broadcast_arrays(
            f_S[..., 0], self.spectrum[None, ...]
        )

        time_series_spectrum = jnp.squeeze(
            # photospheric component:
            photosphere +

            # sum of the active region components:
            jnp.sum(spot_coverages * spot_spectra, axis=1)
        )

        return time_series_spectrum


def bin_spectrum(spectrum, bins=None, log=True, min=None, max=None, **kwargs):
    """
    Bin a spectrum, with log-spaced frequency bins.

    Parameters
    ----------
    spectrum : `specutils.Spectrum1D`
        Spectrum to be binned
    bins : int or ~numpy.ndarray
        Number of bins, or the bin edges
    log : bool
        If true, compute bin edges based on the log base 10 of
        the frequency.
    min : ~astropy.units.Quantity
        Minimum wavelength to include in the binned spectrum.
    max : ~astropy.units.Quantity
        Maximum wavelength to include in the binned spectrum.
    **kwargs : dict
        Additional keyword arguments passed to `scipy.stats.binned_statistic`

    Returns
    -------
    new_spectrum : `specutils.Spectrum1D`
        Binned spectrum
    """
    in_range = (spectrum.wavelength > min) & (spectrum.wavelength < max)
    wavelength = spectrum.wavelength[in_range]
    flux = spectrum.flux[in_range]

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
    if np.any(nans) and all(
        map(lambda x: len(x) > 0, [wl_bins[nans], wl_bins[~nans],
                                   bs.statistic[~nans]])
    ):
        interp_fluxes[nans] = np.interp(wl_bins[nans], wl_bins[~nans],
                                        bs.statistic[~nans])
    return Spectrum1D(
        flux=interp_fluxes * flux.unit, spectral_axis=wl_bins,
        meta=spectrum.meta
    )


def spectral_binning(y, all_x, all_y):
    """
    Spectral binning via trapezoidal approximation.

    Parameters
    ----------
    y : array
        Values to be binned
    all_x : array
        Full x-axis of the spectrum
    all_y : array
        Full y-axis of the spectrum
    """
    min_ind = np.argwhere(all_y == y[0])[0, 0]
    max_ind = np.argwhere(all_y == y[-1])[0, 0]
    if max_ind > min_ind and y.shape == all_x[min_ind:max_ind + 1].shape:
        return (np.trapz(y, all_x[min_ind:max_ind + 1])
                / (all_x[max_ind] - all_x[min_ind]))
    return np.nan
