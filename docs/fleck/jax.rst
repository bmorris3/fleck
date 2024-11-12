*************
JAX Interface
*************

.. warning::

    If you install ``fleck`` via pip, you may not have the ``jax``
    dependencies necessary for this module. To ensure that you do,
    follow the instructions for ``jax`` compatibility in :doc:`installation`.


Intro
-----

We're going to compute the transmission spectrum for a transit
of TRAPPIST-1 c, assuming the planet has no atmosphere and
the transmission spectrum changes with wavelength due only to
stellar surface features. We will place one warm and one cool
active region on the photosphere,

.. note::

    This tutorial depends on the package ``expecto``
    (`code <https://github.com/bmorris3/expecto>`_,
    `docs <https://expecto.readthedocs.io/>`_). You can
    install ``expecto`` via pip with:

    .. code-block:: bash

        python -m pip install expecto

First, we need to import some things,

.. code-block:: python

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    import numpy as np
    import astropy.units as u
    from expecto import get_spectrum

    from jax import numpy as jnp
    from fleck.jax import ActiveStar, bin_spectrum

and we choose the times and wavelengths at which to compute the
spectrophotometry for our active star+planet model,

.. code-block:: python

    times = np.linspace(-0.04, 0.04, 250)
    wavelength = np.geomspace(0.5, 5, 101) * u.um

Choose active region spectra
----------------------------

Active regions emit spectra that are distinct from the photosphere.
In this model, we will use three PHOENIX model spectra, downloaded
via the `expecto <https://expecto.readthedocs.io/en/latest/>`_ package.
The photospheric, cool, and warm regions will have temperatures:
:math:`T_{\rm phot}=2600 {\rm ~K}`,
:math:`T_{\rm cool}=2400 {\rm ~K}`, and
:math:`T_{\rm warm}=2800 {\rm ~K}`, all with the same surface gravity.
We will bin those spectra onto the ``wavelength`` grid that we chose
above using `~fleck.jax.bin_spectrum`:

.. code-block:: python

    # Set model spectrum binning preferences:
    kwargs = dict(
        bins=wavelength,
        min=wavelength.min(),
        max=wavelength.max(),
        log=False
    )

    # Download and bin PHOENIX model spectra:
    phot, cool, hot = [
        bin_spectrum(
            get_spectrum(T_eff=T_eff, log_g=5.0, cache=True), **kwargs
        )
        for T_eff in [2600, 2400, 2800]
    ]


We will visualize the contaminated transmission spectrum, the transit light curves
at several wavelengths, and a 2D orthographic projection of the stellar surface.
A lot of code goes into making such a plot, which we have hidden in this collapsible
element in the documentation below. If you're interested in how it works, click
it to expand.

.. collapse:: Define a (long) plotting function (click to expand)

    .. code-block:: python

        def plot_transit_contamination(
            active_star, planet_params,
            norm_oot_per_wavelength=True,
            norm_stellar_spectrum=True
        ):
            lc, contam, X, Y, spectrum_at_transit = active_star.transit_model(**planet_params)
            fig = plt.figure(figsize=(8, 4), dpi=100)
            gs = GridSpec(2, 2, figure=fig)

            ax = [
                fig.add_subplot(gs[0, 0]),
                fig.add_subplot(gs[1, 0]),
                fig.add_subplot(gs[:, 1:3]),
            ]

            skip = (len(active_star.wavelength) - 1) // 10

            cmap = lambda i: plt.cm.Spectral_r(
                (active_star.wavelength[i] - active_star.wavelength.min()) /
                active_star.wavelength.ptp()
            )

            if norm_stellar_spectrum:
                scale_relative_to_flux_at_wavelength = 1
            else:
                scale_relative_to_flux_at_wavelength = (
                    spectrum_at_transit / spectrum_at_transit.mean()
                )[::skip]

            for i, lc_i in enumerate(
                (lc * scale_relative_to_flux_at_wavelength)[:, ::skip].T
            ):

                if norm_oot_per_wavelength:
                    lc_i /= lc_i.mean()

                ax[0].plot(active_star.times, lc_i, color=cmap(skip * i))


            ax[0].set(
                xlabel='Time [d]',
                ylabel='$\\left(F(t)/\\bar{F}\\right)_{\\lambda}$',
            )

            contaminated_depth = 1e6 * contam

            ax[1].plot(
                active_star.wavelength * 1e6,
                contaminated_depth,
                zorder=-3, lw=2.5, color='silver'
            )
            ax[1].scatter(
                active_star.wavelength[::skip] * 1e6, contaminated_depth[::skip].T,
                c=cmap(skip * np.arange(len(active_star.wavelength) // skip + 1)),
                s=50, edgecolor='k', zorder=4
            )
            ax[1].set(
                xlabel='Wavelength [µm]',
                ylabel='Transit depth [ppm]',
                xscale='log',
                xlim=[
                    1e6 * 0.9 * active_star.wavelength.min(),
                    1e6 * 1.1 * active_star.wavelength.max()
                ],
            )

            active_star.plot_star(
                t0=planet_params['t0'],
                rp=planet_params['rp'],
                a=planet_params['a'],
                ecc=planet_params['ecc'],
                inclination=planet_params['inclination'],
                ax=ax[2]
            )

            for sp in ['right', 'top']:
                for axis in ax:
                    axis.spines[sp].set_visible(False)

            fig.tight_layout()
            plt.show()



.. raw:: html

    <br />

Construct a model for an active star
------------------------------------

Now we define our `~fleck.jax.ActiveStar` model by the specific times and wavelengths
that we will observe, the stellar inclination, and spectrum of the stellar photosphere.

.. code-block:: python

    # stellar parameters:
    active_star = ActiveStar(
        times=times,
        inclination=np.pi/2,  # stellar inc [rad]
        T_eff=phot.meta['PHXTEFF'],
        wavelength=phot.wavelength.to_value(u.m),
        phot=phot.flux.value,
    )


We can add active regions to the star with `~fleck.jax.ActiveStar.add_spot`

.. code-block:: python

    # add a cool spot:
    active_star.add_spot(
        lon=-0.2,  # [rad]
        lat=1.65,  # [rad]
        rad=0.15,  # [R_star]
        spectrum=cool.flux.value,
        temperature=cool.meta['PHXTEFF']
    )

    # add a hot spot:
    active_star.add_spot(
        lon=0.95,
        lat=1.75,
        rad=0.08,
        spectrum=hot.flux.value,
        temperature=hot.meta['PHXTEFF']
    )

Add a transiting planet with spot occultations
----------------------------------------------

In order to model the transit of a planet, we need to define the
planet's parameters like so:

.. code-block:: python

    # planet parameters for TRAPPIST-1 c from Agol 2021:
    t1c = dict(
        inclination = np.radians(89.778),
        a = 28.549,
        rp = 0.08440,
        period = 2.421937,
        t0 = 0,
        ecc = 0,
        u1 = 0.1,
        u2 = 0.3
    )

And finally, we're prepared to make the plot. We will call the function
`plot_transit_contamination` below, which we defined in a collapsible
code cell above, makes a lot of plotting calls to visualize the results of
`~fleck.jax.ActiveStar.transit_model` and
`~fleck.jax.ActiveStar.plot_star`:

.. code-block:: python

    plot_transit_contamination(active_star, t1c)

.. plot::

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    import numpy as np
    import astropy.units as u
    from expecto import get_spectrum

    from jax import numpy as jnp
    from fleck.jax import ActiveStar, bin_spectrum


    times = np.linspace(-0.04, 0.04, 250)
    # times = np.linspace(-0.04, 3.3, 250)
    wavelength = np.geomspace(0.5, 5, 101) * u.um

    # Download and bin PHOENIX model spectra to compute contrast:
    kwargs = dict(
        bins=wavelength,
        min=wavelength.min(),
        max=wavelength.max(),
        log=False
    )

    phot, cool, hot = [
        bin_spectrum(
            get_spectrum(T_eff=T_eff, log_g=5.0, cache=True), **kwargs
        )
        for T_eff in [2600, 2400, 2800]
    ]

    def plot_transit_contamination(
        active_star, planet_params,
        norm_oot_per_wavelength=True,
        norm_stellar_spectrum=True
    ):
        lc, contam, X, Y, spectrum_at_transit = active_star.transit_model(**planet_params)
        fig = plt.figure(figsize=(9.5, 5), dpi=150)
        gs = GridSpec(2, 2, figure=fig)

        ax = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[:, 1:3]),
        ]

        skip = (len(active_star.wavelength) - 1) // 10

        cmap = lambda i: plt.cm.Spectral_r(
            (active_star.wavelength[i] - active_star.wavelength.min()) /
            active_star.wavelength.ptp()
        )

        if norm_stellar_spectrum:
            scale_relative_to_flux_at_wavelength = 1
        else:
            scale_relative_to_flux_at_wavelength = (
                spectrum_at_transit / spectrum_at_transit.mean()
            )[::skip]

        for i, lc_i in enumerate(
            (lc * scale_relative_to_flux_at_wavelength)[:, ::skip].T
        ):

            if norm_oot_per_wavelength:
                lc_i /= lc_i.mean()

            ax[0].plot(active_star.times, lc_i, color=cmap(skip * i))

        ax[0].set(
            xlabel='Time [d]',
            ylabel='$\\left(F(t)/\\bar{F}\\right)_{\\lambda}$',
        )

        contaminated_depth = 1e6 * contam

        ax[1].plot(
            active_star.wavelength * 1e6,
            contaminated_depth,
            zorder=-3, lw=2.5, color='silver'
        )
        ax[1].scatter(
            active_star.wavelength[::skip] * 1e6, contaminated_depth[::skip].T,
            c=cmap(skip * np.arange(len(active_star.wavelength) // skip + 1)),
            s=50, edgecolor='k', zorder=4
        )
        ax[1].set(
            xlabel='Wavelength [µm]',
            ylabel='Transit depth [ppm]',
            xscale='log',
            xlim=[
                1e6 * 0.9 * active_star.wavelength.min(),
                1e6 * 1.1 * active_star.wavelength.max()
            ],
        )

        active_star.plot_star(
            t0=planet_params['t0'],
            rp=planet_params['rp'],
            a=planet_params['a'],
            ecc=planet_params['ecc'],
            inclination=planet_params['inclination'],
            ax=ax[2]
        )

        for sp in ['right', 'top']:
            for axis in ax:
                axis.spines[sp].set_visible(False)

        fig.tight_layout()
        plt.show()

    # stellar parameters:
    active_star = ActiveStar(
        times=times,
        inclination=np.pi/2,
        T_eff=2600,
        wavelength=phot.wavelength.to_value(u.m),
        phot=phot.flux.value,
    )

    # add a cool spot:
    active_star.add_spot(
        lon=-0.2,  # [rad]
        lat=1.65,  # [rad]
        rad=0.15,  # [R_star]
        spectrum=cool.flux.value,
        temperature=cool.meta['PHXTEFF']
    )

    # add a hot spot:
    active_star.add_spot(
        lon=0.95,
        lat=1.75,
        rad=0.08,
        spectrum=hot.flux.value,
        temperature=hot.meta['PHXTEFF']
    )

    # planet parameters for TRAPPIST-1 c from Agol 2021:
    t1c = dict(
        inclination = np.radians(89.778),
        a = 28.549,
        rp = 0.08440,
        period = 2.421937,
        t0 = 0,
        ecc = 0,
        u1 = 0.1,
        u2 = 0.3
    )

    plot_transit_contamination(active_star, t1c)

In the plot above, several time series light curves are shown in the top left,
where each color corresponds to a different wavelength. There is an occultation of
the cool spot by the planet just before mid-transit, and an occultation of the hot
spot by the planet just before egress. Rotational modulation of the star is seen in
the slope in the wavelength dependent out-of-transit flux.

The plot in the bottom left shows the apparent transmission spectrum as the planet
transits the star. For an airless planet with `rp = 0.08440`, the expected transit
depth (`rp**2`) is 7123 ppm, and the deviations from that value in the transmission
spectrum arise from unocculted active regions. The colored points on the spectrum
label the wavelengths of each light curve of the same color on the upper left panel.

The stellar schematic on the right shows the stellar surface with the photosphere in
light orange, the cool region with darker orange, and the warm region in light yellow.
The two dashed lines trace the upper and lower limit of the planet's transit chord.
The transit occurs with ingress on the left of the plot, and egress to the right, and
the stellar rotation occurs in the same direction.
