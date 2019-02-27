import numpy as np
import astropy.units as u
from astropy.coordinates import (CartesianRepresentation,
                                 UnitSphericalRepresentation)
from astropy.coordinates.matrix_utilities import rotation_matrix
from scipy.integrate import quad

__all__ = ['Stars']


def limb_darkening(u_ld, r):
    u1, u2 = u_ld
    mu = np.sqrt(1 - r**2)
    return (1 - u1 * (1 - mu) - u2 * (1 - mu)**2) / (1 - u1/3 - u2/6) / np.pi


def limb_darkening_normed(u_ld, r):
    return limb_darkening(u_ld, r)/limb_darkening(u_ld, 0)


def total_flux(u_ld):
    return 2 * np.pi * quad(lambda r: r * limb_darkening_normed(u_ld, r),
                            0, 1)[0]


class Stars(object):
    """
    Object describing properties of a population of stars
    """
    def __init__(self, spot_contrast, n_phases, u_ld):
        """
        Parameters
        ----------
        spot_contrast : float
            Contrast of spots (0=perfectly dark, 1=same as photosphere)
        n_phases : int
            Number of rotation steps to iterate over
        u_ld : list
            Quadratic limb-darkening parameters
        """
        self.spot_contrast = spot_contrast
        self.n_phases = n_phases
        self.u_ld = u_ld
        self.phases = np.arange(0, 2 * np.pi,
                                2 * np.pi / self.n_phases) * u.rad
        self.f0 = total_flux(u_ld)

    def light_curve(self, spot_lons, spot_lats, spot_radii, inc_stellar):
        """
        Generate an ensemble of stellar rotational light curves.

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

        Returns
        -------
        light_curves : `~numpy.ndarray`
            Stellar light curves
        """
        usr = UnitSphericalRepresentation(spot_lons, spot_lats)
        cartesian = usr.represent_as(CartesianRepresentation)
        rotate = rotation_matrix(self.phases[:, np.newaxis, np.newaxis],
                                 axis='z')
        tilt = rotation_matrix(inc_stellar - 90*u.deg, axis='y')
        rotated_spot_positions = cartesian.transform(rotate)
        tilted_spot_positions = rotated_spot_positions.transform(tilt)

        r = np.ma.masked_array(np.sqrt(tilted_spot_positions.y**2 +
                                       tilted_spot_positions.z**2),
                               mask=tilted_spot_positions.x < 0)
        ld = limb_darkening_normed(self.u_ld, r)

        f_spots = (np.pi * spot_radii**2 * (1 - self.spot_contrast) * ld *
                   np.sqrt(1 - r**2))

        delta_f = (1 - np.sum(f_spots/self.f0, axis=1)).data
        return delta_f/delta_f.max(axis=0)

