***************
Getting started
***************

Rotational modulation
---------------------

Suppose you want to observe 100 stars with randomly drawn stellar inclinations,
each with spot contrast ``c=0.7`` (where ``c=0`` means perfectly dark spots),
with quadratic limb-darkening. Let's distribute three spots on each star
randomly above 70 degrees latitude up to the pole. We can use the
`~fleck.generate_spots` method to quickly create spot property matrices in the
correct shape::

    from fleck import generate_spots

    spot_contrast = 0.7
    u_ld = [0.5079, 0.2239]

    n_phases = 30
    n_inclinations = 100
    n_spots = 3

    spot_radius = 0.1   # Rspot/Rstar
    min_latitude = 70   # deg
    max_latitude = 90   # deg

    lons, lats, radii, inc_stellar = generate_spots(min_latitude, max_latitude,
                                                    spot_radius, n_spots,
                                                    n_inclinations=n_inclinations)

``lons, lats, radii`` will each have shape ``(n_spots, n_inclinations)`` and
``inc_stellar`` will have shape ``(n_inclinations, )``. Now let's initialize
a `~fleck.Star` object::

    from fleck import Star

    star = Star(spot_contrast=spot_contrast, n_phases=n_phases, u_ld=u_ld)

If we initialize the `~fleck.Star` object with a number of phases ``n_phases``,
it will evenly sample all phases on :math:`(0, 2\pi)`.

Now we can compute light curves for stars with the spots we generated like so::

    lcs = star.light_curve(lons, lats, radii, inc_stellar)

where ``lcs`` will have shape ``(n_phases, n_inclinations)``. Let's plot each of
the light curves::

    import matplotlib.pyplot as plt
    plt.plot(star.phases, lcs)
    plt.show()

.. plot::

    import matplotlib.pyplot as plt
    from fleck import generate_spots, Star

    spot_contrast = 0.7
    u_ld = [0.5079, 0.2239]

    n_phases = 30
    n_inclinations = 100
    n_spots = 3

    spot_radius = 0.1   # Rspot/Rstar
    min_latitude = 70   # deg
    max_latitude = 90   # deg

    lons, lats, radii, inc_stellar = generate_spots(min_latitude, max_latitude,
                                                    spot_radius, n_spots,
                                                    n_inclinations=n_inclinations)

    star = Star(spot_contrast=spot_contrast, n_phases=n_phases, u_ld=u_ld)

    lcs = star.light_curve(lons, lats, radii, inc_stellar)
    plt.plot(star.phases, lcs, alpha=0.1, color='k')
    plt.xlabel('Phases')
    plt.ylabel('Flux')
    plt.show()

Spot Occultations
-----------------

Now let's make a transiting exoplanet, and observe spot occultations. We can
specify the parameters of the transiting exoplanet using the same specification
used by `batman <https://github.com/lkreidberg/batman>`_::

    from batman import TransitParams
    import astropy.units as u

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

Let's now specify some spots on the stellar surface::

    import numpy as np

    inc_stellar = 90 * u.deg
    spot_radii = np.array([[0.1], [0.1]])
    spot_lats = np.array([[0], [0]]) * u.deg
    spot_lons = np.array([[360-30], [30]]) * u.deg

and some times at which to observe the system::

    times = np.linspace(-0.5, 0.5, 500)

let's initialize our `~fleck.Star` object, specifying a stellar rotation
period::

    star = Star(spot_contrast=0.7, u_ld=planet.u, rotation_period=10)

We generate a light curve using the same `~fleck.Star.light_curve` method that
we used earlier, but this time we will supply it with the planet's parameters
and the times at which to evaluate the model::

    lc = star.light_curve(spot_lons, spot_lats, spot_radii,
                          inc_stellar, planet=planet, times=times)

Finally we can plot the transit light curve::

    import matplotlib.pyplot as plt
    plt.plot(times, lc, color='k')
    plt.show()

.. plot::

    from batman import TransitParams
    import matplotlib.pyplot as plt
    import numpy as np
    import astropy.units as u
    from fleck import Star

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

    inc_stellar = 90 * u.deg
    spot_radii = np.array([[0.1], [0.1]])
    spot_lats = np.array([[0], [0]]) * u.deg
    spot_lons = np.array([[360-30], [30]]) * u.deg
    times = np.linspace(-0.5, 0.5, 500)

    star = Star(spot_contrast=0.7, u_ld=planet.u, rotation_period=10)
    lc = star.light_curve(spot_lons, spot_lats, spot_radii,
                          inc_stellar, planet=planet, times=times)

    plt.plot(times, lc, color='k')
    plt.xlabel('Time [d]')
    plt.ylabel('Flux')
    plt.show()
