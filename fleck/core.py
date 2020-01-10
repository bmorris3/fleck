import numpy as np
import astropy.units as u
from astropy.coordinates import (CartesianRepresentation,
                                 UnitSphericalRepresentation)
from astropy.coordinates.matrix_utilities import rotation_matrix
from scipy.spatial.distance import pdist, squareform
from scipy.integrate import quad
from shapely.geometry.point import Point
from shapely import affinity
from batman import TransitModel
import matplotlib.pyplot as plt


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


def ellipse(center, lengths, angle=0):
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


def circle(center, radius):
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
    return affinity.scale(Point(center).buffer(1),
                          xfact=radius, yfact=radius)


def consecutive(data, step_size=1):
    """
    Identify groups of consecutive integers, split them into separate arrays.
    """
    return np.split(data, np.where(np.diff(data) != step_size)[0]+1)


def sort_plot_points(xy_coord, k0=0):
    """
    Iteratively identify a continuous path from the given points xy_coord,
    starting by the point indexed by k0
    """
    n = len(xy_coord)
    distance_matrix = squareform(pdist(xy_coord, metric='euclidean'))
    mask = np.ones(n, dtype='bool')
    sorted_order = np.zeros(n, dtype=np.int)
    indices = np.arange(n)

    i = 0
    k = k0
    while True:
        sorted_order[i] = k
        mask[k] = False

        dist_k = distance_matrix[k][mask]
        indices_k = indices[mask]

        if not len(indices_k):
            break

        # find next unused closest point
        k = indices_k[np.argmin(dist_k)]
        i += 1
    return sorted_order


class Star(object):
    """
    Object describing properties of a (population of) star(s)
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
                    planet=None, times=None, fast=False, time_ref=None,
                    return_spots_occulted=False):
        """
        Generate a(n ensemble of) light curve(s).

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
        planet : `~batman.TransitParams`, optional
            Transiting planet parameters
        times : `~numpy.ndarray`, optional
            Times at which to compute the light curve
        fast : bool, optional
            When `True`, use approximation that spots are fixed on the star
            during a transit event. When `False`, account for motion of
            starspots on stellar surface due to rotation during transit event.
            Default is `False`.
        time_ref : float, optional
            Reference time used as the initial rotational phase of the star,
            such that the sub-observer point is at zero longitude at
            ``time_ref``.
        return_spots_occulted : bool, optional
            Return whether or not spots have been occulted.

        Returns
        -------
        light_curves : `~numpy.ndarray`
            Stellar light curves of shape ``(n_phases, len(inc_stellar))`` or
            ``(len(times), len(inc_stellar))``
        """
        if time_ref is None:
            if times is not None:
                time_ref = 0
            else:
                time_ref = self.phases[0]

        # Compute the spot positions in cartesian coordinates:
        tilted_spots = self.spherical_to_cartesian(spot_lons, spot_lats,
                                                   inc_stellar, times=times,
                                                   planet=planet,
                                                   time_ref=time_ref)

        # Compute the distance of each spot from the stellar centroid, mask
        # any spots that are "behind" the star, in other words, x < 0
        r = np.ma.masked_array(np.hypot(tilted_spots.y.value,
                                        tilted_spots.z.value),
                               mask=tilted_spots.x.value < 0)
        ld = limb_darkening_normed(self.u_ld, r)

        # Compute the out-of-transit flux missing due to each spot
        f_spots = (np.pi * spot_radii**2 * (1 - self.spot_contrast) * ld *
                   np.sqrt(1 - r**2))

        if planet is None:
            # If there is no transiting planet, skip the transit routine:
            lambda_e = np.zeros((len(self.phases), 1))
        else:
            if not inc_stellar.isscalar:
                raise ValueError('Transiting exoplanets are implemented for '
                                 'planets transiting single stars only, but '
                                 '``inc_stellar`` has multiple values. ')
            # Compute a transit model
            n_spots = len(spot_lons)
            m = TransitModel(planet, times)
            lambda_e = 1 - m.light_curve(planet)[:, np.newaxis]
            # Compute the true anomaly of the planet at each time, f:
            f = m.get_true_anomaly()

            # Compute the position of the planet in cartesian coordinates using
            # Equations 53-55 of Murray & Correia (2010). Note that these
            # coordinates are different from the cartesian coordinates used for
            # the spot positions. In this system, the observer is at X-> -inf.
            I = np.radians(90 - planet.inc)
            Omega = np.radians(planet.w)  # this is 90 deg by default
            omega = np.pi / 2
            X = planet.a * (np.cos(Omega) * np.cos(omega + f) -
                            np.sin(Omega) * np.sin(omega + f) * np.cos(I))
            Y = planet.a * (np.sin(Omega) * np.cos(omega + f) +
                            np.cos(Omega) * np.sin(omega + f) * np.cos(I))
            Z = planet.a * np.sin(omega + f) * np.sin(I)

            # Create a shapely circle object for the planet's silhouette only
            # when the planet is in front of the star, otherwise append `None`
            planet_disk = [circle([Y[i], -Z[i]], planet.rp)
                           if (np.abs(Y[i]) < 1 + planet.rp) and
                              (X[i] < 0) else None
                           for i in range(len(f))]

            if fast:
                spots_occulted = self._planet_spot_overlap_fast(planet,
                                                                planet_disk,
                                                                tilted_spots,
                                                                spot_radii,
                                                                n_spots, X, Y,
                                                                lambda_e)
            else:
                spots_occulted = self._planet_spot_overlap_slow(planet,
                                                                planet_disk,
                                                                tilted_spots,
                                                                spot_radii,
                                                                n_spots, X, Y,
                                                                lambda_e)

        # Return the flux missing from the star at each time due to spots
        # (f_spots/self.f0) and due to the transit (lambda_e):
        if return_spots_occulted:
            return (1 - np.sum(f_spots.filled(0)/self.f0, axis=1) - lambda_e,
                    spots_occulted)

        else:
            return 1 - np.sum(f_spots.filled(0)/self.f0, axis=1) - lambda_e

    def spherical_to_cartesian(self, spot_lons, spot_lats, inc_stellar,
                               times=None, planet=None, time_ref=None):
        """
        Convert spot parameter matrices in the original stellar coordinates to
        rotated and tilted cartesian coordinates.

        Parameters
        ----------
        spot_lons : `~astropy.units.Quantity`
            Spot longitudes
        spot_lats : `~astropy.units.Quantity`
            Spot latitudes
        inc_stellar : `~astropy.units.Quantity`
            Stellar inclination
        times : `~numpy.ndarray`
            Times at which evaluate the stellar rotation
        planet : `~batman.TransitParams`
            Planet parameters
        time_ref : float
            Reference time used as the initial rotational phase of the star,
            such that the sub-observer point is at zero longitude at
            ``time_ref``.

        Returns
        -------
        tilted_spots : `~numpy.ndarray`
            Rotated and tilted spot positions in cartesian coordinates
        """
        # Spots by default are given in unit spherical representation (lat, lon)
        usr = UnitSphericalRepresentation(spot_lons, spot_lats)

        # Represent those spots with cartesian coordinates (x, y, z)
        # In this coordinate system, the observer is at positive x->inf,
        # the star is at the origin, and (y, z) is the sky plane.
        cartesian = usr.represent_as(CartesianRepresentation)

        # Generate array of rotation matrices to rotate the spots about the
        # stellar rotation axis
        if times is None:
            rotate = rotation_matrix(self.phases[:, np.newaxis, np.newaxis],
                                     axis='z')
        else:
            if time_ref is None:
                time_ref = 0
            rotational_phase = 2 * np.pi * ((times - time_ref) /
                                            self.rotation_period) * u.rad
            rotate = rotation_matrix(rotational_phase[:, np.newaxis, np.newaxis],
                                     axis='z')

        rotated_spots = cartesian.transform(rotate)

        if planet is not None and hasattr(planet, 'lam'):
            lam = planet.lam * u.deg
        else:
            lam = 0 * u.deg

        # Generate array of rotation matrices to rotate the spots so that the
        # star is observed from the correct stellar inclination
        stellar_inclination = rotation_matrix(inc_stellar - 90*u.deg, axis='y')
        inclined_spots = rotated_spots.transform(stellar_inclination)

        # Generate array of rotation matrices to rotate the spots so that the
        # planet's orbit normal is tilted with respect to stellar spin
        tilt = rotation_matrix(-lam, axis='x')
        tilted_spots = inclined_spots.transform(tilt)

        return tilted_spots

    def _planet_spot_overlap_fast(self, planet, planet_disk, tilted_spots,
                                  spot_radii, n_spots, X, Y, lambda_e):
        """
        Compute the overlap between the planet and starspots using the fast,
        approximate method.

        The approximation used by the fast method is to assume that: (1) the
        mid-transit time is observed for each transit in the observations, and
        (2) starspots are fixed during the transit event. This approximation is
        suitable to use when the stellar rotation period is much longer than the
        transit duration, and only complete transits are observed.

        Parameters
        ----------
        planet : `~batman.TransitParams`
            Planet parameters from the batman API
        planet_disk : list
            List of positions of the planet when the planet is in front of the
            star.
        tilted_spots : `~astropy.units.Quantity`
            Cartesian positions of the starspots in the planet's "observer
            oriented" coordinate frame.
        spot_radii : `~numpy.ndarray`
            Radii of the starspots
        n_spots : int
            Number of spots
        X : `~numpy.ndarray`
            Cartesian `X` position of the planet at all times
        Y : `~numpy.ndarray`
            Cartesian `Y` position of the planet at all times
        lambda_e : `~numpy.ndarray`
            Occulted flux fraction (Mandel & Agol 2002)
        """
        spots_occulted = False
        # Find the approximate mid-transit time indices in the observations
        # by looking for the sign flip in Y (planet crosses the sub-observer
        # point) when also X < 0 (planet in front of star):
        t0_inds = np.argwhere((np.sign(Y[1:]) < np.sign(Y[:-1])) &
                              (X[1:] < 0))

        # Compute the indices where the planet is in front of the star
        # (X < 0) and the planet is near the star |Y| < 1 + p:
        transit_inds_all = np.argwhere((X < 0) &
                                       (np.abs(Y) < 1 + planet.rp))[:, 0]

        # Split these indices up into separate numpy arrays for each
        # contiguous group - this will generate a list of numpy arrays each
        # containing the indices during individual transit events.
        transit_inds_groups = consecutive(transit_inds_all)

        # For each transit in the observations:
        for k, t0_ind, transit_inds in zip(range(len(t0_inds)), t0_inds,
                                           transit_inds_groups):

            spots = []
            spot_ld_factors = []

            for i in range(n_spots):
                # If the spot is visible (x > 0):
                if tilted_spots.x.value[t0_ind, i] > 0:
                    spot_y = tilted_spots.y.value[t0_ind, i]
                    spot_z = tilted_spots.z.value[t0_ind, i]

                    # Compute the spot position and ellipsoidal shape
                    r_spot = np.hypot(spot_z, spot_y)
                    angle = np.arctan2(spot_z, spot_y)
                    ellipse_centroid = [spot_y, spot_z]

                    ellipse_axes = [spot_radii[i, 0] *
                                    np.sqrt(1 - r_spot**2),
                                    spot_radii[i, 0]]

                    spot = ellipse(ellipse_centroid, ellipse_axes,
                                   np.degrees(angle))

                    # Add the spot to our spot list
                    spots.append(spot)
                    spot_ld_factors.append(limb_darkening_normed(self.u_ld,
                                                                 r_spot))

            # If any spots are visible:
            if len(spots) > 0:
                intersections = np.zeros((transit_inds.ptp()+1, len(spots)))

                # For each time when the planet is nearly transiting:
                for i in range(len(transit_inds)):
                    planet_disk_i = planet_disk[transit_inds[i]]
                    if planet_disk_i is not None:
                        for j in range(len(spots)):

                            # Compute the overlap between each spot and the
                            # planet using shapely's `intersection` method
                            spot_planet_overlap = planet_disk_i.intersection(spots[j]).area

                            intersections[i, j] = ((1 - self.spot_contrast) /
                                                   spot_ld_factors[j] *
                                                   spot_planet_overlap /
                                                   np.pi)

                # Subtract the spot occultation amplitudes from the spotless
                # transit model that we computed earlier
                lambda_e[transit_inds] -= intersections.max(axis=1)[:, np.newaxis]
                if not np.all(intersections == 0):
                    spots_occulted = True

        return spots_occulted

    def _planet_spot_overlap_slow(self, planet, planet_disk, tilted_spots,
                                  spot_radii, n_spots, X, Y, lambda_e):
        """
        Compute the overlap between the planet and starspots using the slow,
        precise method.

        This method accounts for the motion of the starspots during the transit
        event.

        Parameters
        ----------
        planet : `~batman.TransitParams`
            Planet parameters from the batman API
        planet_disk : list
            List of positions of the planet when the planet is in front of the
            star.
        tilted_spots : `~astropy.units.Quantity`
            Cartesian positions of the starspots in the planet's "observer
            oriented" coordinate frame.
        spot_radii : `~numpy.ndarray`
            Radii of the starspots
        n_spots : int
            Number of spots
        X : `~numpy.ndarray`
            Cartesian `X` position of the planet at all times
        Y : `~numpy.ndarray`
            Cartesian `Y` position of the planet at all times
        lambda_e : `~numpy.ndarray`
            Occulted flux fraction (Mandel & Agol 2002)
        """
        spots_occulted = False

        # For each time in the observations:
        for k, planet_disk_i in enumerate(planet_disk):

            if planet_disk_i is not None:
                spots = []
                spot_ld_factors = []

                for i in range(n_spots):
                    # If the spot is visible (x > 0):
                    if tilted_spots.x.value[k, i] > 0:
                        spot_y = tilted_spots.y.value[k, i]
                        spot_z = tilted_spots.z.value[k, i]

                        # Compute the spot position and ellipsoidal shape
                        r_spot = np.hypot(spot_z, spot_y)
                        angle = np.arctan2(spot_z, spot_y)
                        ellipse_centroid = [spot_y, spot_z]

                        ellipse_axes = [spot_radii[i, 0] *
                                        np.sqrt(1 - r_spot**2),
                                        spot_radii[i, 0]]

                        spot = ellipse(ellipse_centroid, ellipse_axes,
                                       np.degrees(angle))

                        # Add the spot to our spot list
                        spots.append(spot)
                        spot_ld_factors.append(limb_darkening_normed(self.u_ld,
                                                                     r_spot))

                # If any spots are visible:
                if len(spots) > 0:
                    intersections = np.zeros(len(spots))
                    for j in range(len(spots)):

                        # Compute the overlap between each spot and the
                        # planet using shapely's `intersection` method
                        spot_planet_overlap = planet_disk_i.intersection(spots[j]).area

                        intersections[j] = ((1 - self.spot_contrast) /
                                            spot_ld_factors[j] *
                                            spot_planet_overlap /
                                            np.pi)

                    # Subtract the spot occultation amplitudes from the spotless
                    # transit model that we computed earlier
                    lambda_e[k] -= intersections.max()
                    if not np.all(intersections == 0):
                        spots_occulted = True
        return spots_occulted

    def plot(self, spot_lons, spot_lats, spot_radii, inc_stellar, time=None,
             planet=None, ax=None, time_ref=None):
        """
        Generate a plot of the stellar surface at ``time``.

        Takes the same arguments as `~fleck.Star.light_curve` with the exception
        of the singular ``time`` rather than ``times``, plus ``ax`` for
        pre-defined matplotlib axes.

        Coordinate frame is the "observer oriented" view defined in Fabrycky &
        Winn (2009) Figure 1a. The planet transits from left to right across the
        image. The dashed gray lines represent the upper and lower bounds of the
        planet's transit chord.

        Parameters
        ----------
        spot_lons : `~astropy.units.Quantity`
            Spot longitudes
        spot_lats : `~astropy.units.Quantity`
            Spot latitudes
        spot_radii : `~numpy.ndarray`
            Spot radii
        inc_stellar : `~astropy.units.Quantity`
            Stellar inclination
        time : float
            Time at which to evaluate the spot parameters
        planet : `~batman.TransitParams` or list
            Planet parameters, or list of planet parameters.
        ax : `~matplotlib.pyplot.Axes`, optional
            Predefined matplotlib axes
        time_ref : float
            Reference time used as the initial rotational phase of the star,
            such that the sub-observer point is at zero longitude at
            ``time_ref``.

        Returns
        -------
        ax : `~matplotlib.pyplot.Axes`
            Axis object.
        """
        tilted_spots = self.spherical_to_cartesian(spot_lons, spot_lats,
                                                   inc_stellar,
                                                   times=np.array([time]),
                                                   planet=planet,
                                                   time_ref=time_ref)
        spots = []

        for i in range(len(spot_lons)):
            # If the spot is visible (x > 0):
            if tilted_spots.x.value[0, i] > 0:
                spot_y = tilted_spots.y.value[0, i]
                spot_z = tilted_spots.z.value[0, i]

                # Compute the spot position and ellipsoidal shape
                r_spot = np.hypot(spot_z, spot_y)
                angle = np.arctan2(spot_z, spot_y)
                ellipse_centroid = [spot_y, spot_z]

                ellipse_axes = [spot_radii[i, 0] *
                                np.sqrt(1 - r_spot**2),
                                spot_radii[i, 0]]

                spot = ellipse(ellipse_centroid, ellipse_axes,
                               np.degrees(angle))

                # Add the spot to our spot list
                spots.append(spot)

        if ax is None:
            ax = plt.gca()

        if hasattr(planet, "__len__"):
            # If there are multiple planets, plot their transit chord boundaries
            for p, color in zip(planet, ["C{0:d}".format(i)
                                         for i in range(len(planet))]):
                # Calculate impact parameter
                b = (p.a * np.cos(np.radians(p.inc)) * (1 - p.ecc**2) /
                     (1 + p.ecc * np.sin(np.radians(p.w))))

                # Compute the upper and lower envelopes of the transit chord in
                # the "observer oriented" reference frame (Fabrycky & Winn 2009)
                planet_lower_extent = -b-p.rp
                planet_upper_extent = -b+p.rp

                ax.axhline(planet_lower_extent, color=color, ls='--')
                ax.axhline(planet_upper_extent, color=color, ls='--')
        else:
            # Calculate impact parameter
            b = (planet.a * np.cos(np.radians(planet.inc)) * (1 - planet.ecc**2) /
                 (1 + planet.ecc * np.sin(np.radians(planet.w))))

            # Compute the upper and lower envelopes of the transit chord in the
            # "observer oriented" reference frame (Fabrycky & Winn 2009)
            planet_lower_extent = -b-planet.rp
            planet_upper_extent = -b+planet.rp

            ax.axhline(planet_lower_extent, color='gray', ls='--')
            ax.axhline(planet_upper_extent, color='gray', ls='--')

        # Compute the position of the rotational pole of the star
        pole_lat, pole_lon = np.array([90])*u.deg, np.array([0])*u.deg
        polar_spot = self.spherical_to_cartesian(pole_lon, pole_lat,
                                                 inc_stellar,
                                                 times=np.array([0]),
                                                 planet=planet)

        equator_lon = np.linspace(0, 2*np.pi, 50) * u.rad
        equator_lat = np.zeros(len(equator_lon)) * u.rad
        equatorial_line = self.spherical_to_cartesian(equator_lon, equator_lat,
                                                      inc_stellar,
                                                      times=np.array([time]),
                                                      planet=planet)

        # Draw the outline of the star:
        x = np.linspace(-1, 1, 1000)
        ax.plot(x, np.sqrt(1-x**2), color='k')
        ax.plot(x, -np.sqrt(1-x**2), color='k')

        # If pole is visible, mark it:
        if polar_spot.x > 0:
            ax.scatter(-polar_spot.y, polar_spot.z, color='k', marker='x')

        # Where equator is visible, mark it:
        equator_visible = equatorial_line.x > 0
        equator_xy = np.vstack([-equatorial_line.y[equator_visible],
                                equatorial_line.z[equator_visible]]).T
        sort_equator = sort_plot_points(equator_xy,
                                        k0=np.argmax(equator_xy[:, 1]))
        ax.plot(-equatorial_line.y[equator_visible][sort_equator],
                equatorial_line.z[equator_visible][sort_equator],
                ls=':', color='gray')

        ax.set(ylim=[-1.01, 1.01], xlim=[-1.01, 1.01], aspect=1)

        # Draw each starspot:
        for i in range(len(spots)):
            spot_x, spot_y = [np.array(j.tolist())
                              for j in spots[i].exterior.xy]
            ax.fill(-spot_x, spot_y, alpha=1-self.spot_contrast,
                    color='k')
        return ax


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
        n_inclinations = len(inclinations) if not inclinations.isscalar else 1
        inc_stellar = inclinations
    radii = spot_radius * np.ones((n_spots, n_inclinations))
    lats = (delta_latitude*np.random.rand(n_spots, n_inclinations) +
            min_latitude) * u.deg
    lons = 360*np.random.rand(n_spots, n_inclinations) * u.deg
    return lons, lats, radii, inc_stellar
