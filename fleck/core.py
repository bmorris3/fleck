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
    u1, u2 = u_ld
    mu = np.sqrt(1 - r**2)
    return (1 - u1 * (1 - mu) - u2 * (1 - mu)**2) / (1 - u1/3 - u2/6) / np.pi


def limb_darkening_normed(u_ld, r):
    return limb_darkening(u_ld, r)/limb_darkening(u_ld, 0)


def total_flux(u_ld):
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
        self.n_phases = n_phases if n_phases is not None else len(phases)
        self.u_ld = u_ld

        if phases is None:
            phases = np.arange(0, 2 * np.pi, 2 * np.pi / self.n_phases) * u.rad

        self.phases = phases
        self.f0 = total_flux(u_ld)
        self.rotation_period = rotation_period

    def light_curve(self, spot_lons, spot_lats, spot_radii, inc_stellar,
                    planet=None, times=None):
        """
        Generate an ensemble of light curves.

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
        planet : `~batman.TransitParams`
            Transiting planet parameters

        Returns
        -------
        light_curves : `~numpy.ndarray`
            Stellar light curves
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
        rotated_spot_positions = cartesian.transform(rotate)
        tilted_spot_positions = rotated_spot_positions.transform(tilt)

        r = np.ma.masked_array(np.sqrt(tilted_spot_positions.y.value**2 +
                                       tilted_spot_positions.z.value**2),
                               mask=tilted_spot_positions.x.value < 0)
        ld = limb_darkening_normed(self.u_ld, r)

        f_spots = (np.pi * spot_radii**2 * (1 - self.spot_contrast) * ld *
                   np.sqrt(1 - r**2))

        if planet is None:
            lambda_e = np.zeros((len(self.phases), 1))
        else:
            if not inc_stellar.isscalar:
                raise ValueError('Currently implemented for planets transiting '
                                 'single stars. ')
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

            planet_disk = [create_ellipse([Y[i], Z[i]], [p, p])
                           if (np.abs(Y[i]) < 1 + p) and (X[i] < 0) else None
                           for i in range(len(f))]

            spots = []
            spot_ld_factors = []

            mid_transit_time = len(times)//2

            for i in range(n_spots):
                if tilted_spot_positions.x.value[mid_transit_time, i] > 0:

                    r_spot = np.hypot(tilted_spot_positions.z.value[mid_transit_time, i],
                                      tilted_spot_positions.y.value[mid_transit_time, i])

                    angle = np.arctan2(tilted_spot_positions.z.value[mid_transit_time, i],
                                       tilted_spot_positions.y.value[mid_transit_time, i])

                    ellipse = create_ellipse([tilted_spot_positions.y.value[mid_transit_time, i],
                                              tilted_spot_positions.z.value[mid_transit_time, i]],
                                             [spot_radii[i, 0]*np.sqrt(1 - r_spot**2),
                                              spot_radii[i, 0]],
                                             np.degrees(angle))
                    spots.append(ellipse)
                    spot_ld_factors.append(limb_darkening_normed(self.u_ld,
                                                                 r_spot))

            print(spot_ld_factors)
            if len(spots) > 0:
                intersections = np.zeros((len(f), len(spots)))
                for i in range(len(f)):
                    if planet_disk[i] is not None:
                        for j in range(len(spots)):
                            intersections[i, j] = ((1 - self.spot_contrast) /
                                                   spot_ld_factors[j] *
                                                   planet_disk[i].intersection(spots[j]).area /
                                                   np.pi)

                lambda_e -= intersections.max(axis=1)[:, np.newaxis]

        return 1 - np.sum(f_spots.filled(0)/self.f0, axis=1) - lambda_e


def generate_spots(min_latitude, max_latitude, spot_radius, n_spots,
                   n_inclinations=None, inclinations=None):
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