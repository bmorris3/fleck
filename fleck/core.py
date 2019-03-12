import numpy as np
import astropy.units as u
from astropy.coordinates import (CartesianRepresentation,
                                 UnitSphericalRepresentation)
from astropy.coordinates.matrix_utilities import rotation_matrix
from scipy.integrate import quad
from shapely.geometry.point import Point
from shapely import affinity
from batman import TransitModel

__all__ = ['Star', 'generate_spots']


def limb_darkening(u_ld, r):
    """
    Quadratic limb darkening function.

    Parameters
    ----------
    u_ld : list
        Quadratic limb-darkening parameters
    r : float or `~numpy.ndarray`
        Radius in units of stellar radii.

    Returns
    -------
    f : float or `~numpy.ndarray`
        Flux at ``r``.
    """
    u1, u2 = u_ld
    mu = np.sqrt(1 - r**2)
    return (1 - u1 * (1 - mu) - u2 * (1 - mu)**2) / (1 - u1/3 - u2/6) / np.pi


def limb_darkening_normed(u_ld, r):
    """
    Limb-darkened flux, normalized by the central flux.

    Parameters
    ----------
    u_ld : list
        Quadratic limb-darkening parameters
    r : float or `~numpy.ndarray`
        Radius in units of stellar radii.

    Returns
    -------
    f : float or `~numpy.ndarray`
        Normalized flux at ``r``
    """
    return limb_darkening(u_ld, r)/limb_darkening(u_ld, 0)


def total_flux(u_ld):
    """
    Compute the total flux of the limb-darkened star.

    Parameters
    ----------
    u_ld : list
        Quadratic limb-darkening parameters

    Returns
    -------
    f : float
        Total flux
    """
    return 2 * np.pi * quad(lambda r: r * limb_darkening_normed(u_ld, r),
                            0, 1)[0]


def create_ellipse(center, lengths, angle=0):
    """
    Create a shapely ellipse.

    Parameters
    ----------
    center : list
        [x, y] centroid of the ellipse
    lengths : list
        [a, b] semimajor and semiminor axes
    angle : float
        Angle in degrees to rotate the semimajor axis

    Returns
    -------
    ellipse : `~shapely.geometry.polygon.Polygon`
        Elliptical shapely object
    """
    ell = affinity.scale(Point(center).buffer(1),
                         xfact=lengths[0], yfact=lengths[1])
    ell_rotated = affinity.rotate(ell, angle=angle)
    return ell_rotated


def create_circle(center, radius):
    """
    Create a shapely ellipse.

    Parameters
    ----------
    center : list
        [x, y] centroid of the ellipse
    radius : float
        Radius of the circle

    Returns
    -------
    circle : `~shapely.geometry.polygon.Polygon`
        Circular shapely object
    """
    circle = affinity.scale(Point(center).buffer(1),
                            xfact=radius, yfact=radius)
    return circle


def consecutive(data, stepsize=1):
    """
    Identify groups of consecutive integers.

    Parameters
    ----------
    data
    stepsize

    Returns
    -------

    """
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


class Star(object):
    """
    Object describing properties of a population of stars
    """
    def __init__(self, spot_contrast, u_ld, phases=None, n_phases=None,
                 rotation_period=None):
        """
        Parameters
        ----------
        spot_contrast : float
            Contrast of spots (0=perfectly dark, 1=same as photosphere)
        u_ld : list
            Quadratic limb-darkening parameters
        n_phases : int, optional
            Number of rotation steps to iterate over
        phases : `~numpy.ndarray`, optional
            Rotational phases of the star
        rotation_period : `~astropy.units.Quantity`, optional
            Rotation period of the star
        """
        self.spot_contrast = spot_contrast
        if phases is not None:
            n_phases = len(phases)
        self.n_phases = n_phases
        self.u_ld = u_ld

        if phases is None and self.n_phases is not None:
            phases = np.arange(0, 2 * np.pi, 2 * np.pi / self.n_phases) * u.rad

        self.phases = phases
        self.f0 = total_flux(u_ld)
        self.rotation_period = rotation_period

    def light_curve(self, spot_lons, spot_lats, spot_radii, inc_stellar,
                    planet=None, times=None):
        """
        Generate an ensemble of light curves.

        Light curve output will have shape ``(n_phases, len(inc_stellar))`` or
        ``(len(times), len(inc_stellar))``.

        Parameters
        ----------
        spot_lons : `~numpy.ndarray`
            Spot longitudes
        spot_lats : `~numpy.ndarray`
            Spot latitudes
        spot_radii : `~numpy.ndarray`
            Spot radii
        inc_stellar : `~numpy.ndarray`
            Stellar inclinations
        planet : `~batman-package.TransitParams`
            Transiting planet parameters
        times : `~numpy.ndarray`
            Times at which to compute the light curve

        Returns
        -------
        light_curves : `~numpy.ndarray`
            Stellar light curves of shape ``(n_phases, len(inc_stellar))`` or
            ``(len(times), len(inc_stellar))``
        """
        usr = UnitSphericalRepresentation(spot_lons, spot_lats)
        cartesian = usr.represent_as(CartesianRepresentation)
        if times is None:
            rotate = rotation_matrix(self.phases[:, np.newaxis, np.newaxis],
                                     axis='z')
        else:
            rotational_phase = 2*np.pi*((times - planet.t0) /
                                        self.rotation_period) * u.rad
            rotate = rotation_matrix(rotational_phase[:, np.newaxis, np.newaxis],
                                     axis='z')
        tilt = rotation_matrix(inc_stellar - 90*u.deg, axis='y')
        rotated_spots = cartesian.transform(rotate)
        tilted_spots = rotated_spots.transform(tilt)

        r = np.ma.masked_array(np.hypot(tilted_spots.y.value,
                                        tilted_spots.z.value),
                               mask=tilted_spots.x.value < 0)
        ld = limb_darkening_normed(self.u_ld, r)

        f_spots = (np.pi * spot_radii**2 * (1 - self.spot_contrast) * ld *
                   np.sqrt(1 - r**2))

        if planet is None:
            lambda_e = np.zeros((len(self.phases), 1))
        else:
            if not inc_stellar.isscalar:
                raise ValueError('Transiting exoplanets are implemented for '
                                 'planets transiting single stars only, but '
                                 '``inc_stellar`` has multiple values. ')
            p = planet.rp
            n_spots = len(spot_lons)
            m = TransitModel(planet, times)
            lambda_e = 1 - m.light_curve(planet)[:, np.newaxis]
            f = m.get_true_anomaly()

            # Eqn 53-55 of Murray & Correia (2010)
            I = np.radians(90 - planet.inc)
            Omega = np.radians(planet.w)  # this is 90 deg by default
            omega = np.pi/2
            X = planet.a * (np.cos(Omega) * np.cos(omega + f) -
                            np.sin(Omega) * np.sin(omega + f) * np.cos(I))
            Y = planet.a * (np.sin(Omega) * np.cos(omega + f) +
                            np.cos(Omega) * np.sin(omega + f) * np.cos(I))
            Z = planet.a * np.sin(omega + f) * np.sin(I)

            planet_disk = [create_circle([Y[i], Z[i]], p)
                           if (np.abs(Y[i]) < 1 + p) and (X[i] < 0) else None
                           for i in range(len(f))]

            t0_inds = np.argwhere((np.sign(Y[1:]) != np.sign(Y[:-1])) &
                                  (Y[:-1] > 0))

            transit_ind_groups = consecutive(np.argwhere((X < 0) &
                                                         (np.abs(Y) < 1))[:, 0])

            for k, t0_ind, transit_inds in zip(range(len(t0_inds)), t0_inds,
                                               transit_ind_groups):

                spots = []
                spot_ld_factors = []

                for i in range(n_spots):
                    if tilted_spots.x.value[t0_ind, i] > 0:

                        r_spot = np.hypot(tilted_spots.z.value[t0_ind, i],
                                          tilted_spots.y.value[t0_ind, i])

                        angle = np.arctan2(tilted_spots.z.value[t0_ind, i],
                                           tilted_spots.y.value[t0_ind, i])

                        ellipse_centroid = [tilted_spots.y.value[t0_ind, i],
                                            tilted_spots.z.value[t0_ind, i]]

                        ellipse_axes = [spot_radii[i, 0] *
                                        np.sqrt(1 - r_spot**2),
                                        spot_radii[i, 0]]

                        ellipse = create_ellipse(ellipse_centroid, ellipse_axes,
                                                 np.degrees(angle))
                        spots.append(ellipse)
                        spot_ld_factors.append(limb_darkening_normed(self.u_ld,
                                                                     r_spot))

                if len(spots) > 0:
                    intersections = np.zeros((transit_inds.ptp()+1, len(spots)))
                    for i in range(len(transit_inds)):
                        planet_disk_i = planet_disk[transit_inds[i]]
                        if planet_disk_i is not None:
                            for j in range(len(spots)):
                                intersections[i, j] = ((1 - self.spot_contrast) /
                                                       spot_ld_factors[j] *
                                                       planet_disk_i.intersection(spots[j]).area /
                                                       np.pi)
                    lambda_e[transit_inds] -= intersections.max(axis=1)[:, np.newaxis]

        return 1 - np.sum(f_spots.filled(0)/self.f0, axis=1) - lambda_e


def generate_spots(min_latitude, max_latitude, spot_radius, n_spots,
                   n_inclinations=None, inclinations=None):
    """
    Generate matrices of spot parameters.

    Will generate ``n_spots`` spots on different stars observed at
    ``n_inclinations`` different inclinations.

    Parameters
    ----------
    min_latitude : float
        Minimum spot latitude
    max_latitude : float
        Maximum spot latitude
    spot_radius : float or `~numpy.ndarray`
        Spot radii
    n_spots : int
        Number of spots to generate
    n_inclinations : int, optional
        Number of inclinations to generate
    inclinations : `~numpy.ndarray`, optional
        Inclinations (user defined). Default (`None`): randomly generate.

    Returns
    -------
    lons : `~astropy.units.Quantity`
        Spot longitudes, shape ``(n_spots, n_inclinations)``
    lats : `~astropy.units.Quantity`
        Spot latitudes, shape ``(n_spots, n_inclinations)``
    radii : float or `~numpy.ndarray`
        Spot radii, shape ``(n_spots, n_inclinations)``
    inc_stellar : `~astropy.units.Quantity`
        Stellar inclinations, shape ``(n_inclinations, )``
    """
    delta_latitude = max_latitude - min_latitude
    if n_inclinations is not None and inclinations is None:
        inc_stellar = (180*np.random.rand(n_inclinations) - 90) * u.deg
    else:
        n_inclinations = len(inclinations)
        inc_stellar = inclinations
    radii = spot_radius * np.ones((n_spots, n_inclinations))
    lats = (delta_latitude*np.random.rand(n_spots, n_inclinations) +
            min_latitude) * u.deg
    lons = 360*np.random.rand(n_spots, n_inclinations) * u.deg
    return lons, lats, radii, inc_stellar
