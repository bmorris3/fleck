*****
fleck
*****

Fast starspot rotational modulation light curves in Python 3.

``fleck`` is a pure Python software package for simulating rotational modulation
of stars due to starspots, which we use to overcome the degeneracies and
determine starspot coverages accurately for a sample of young stars. ``fleck``
simulates starspots as circular dark regions on the surfaces of rotating stars,
accounting for foreshortening towards the limb, and limb darkening.

The user supplies the latitudes, longitudes and radii of spots, and the stellar
inclinations from which each star is viewed, and ``fleck`` takes advantage of
efficient array broadcasting with ``numpy`` to  return approximate light curves.
For example, the present algorithm can compute rotational
modulation curves sampled at ten points throughout the rotation of each star
for one million stars, with two unique spots each, all viewed at unique
inclinations, in about 10 seconds on a 2.5 GHz Intel Core i7 processor.
This rapid computation of light curves *en masse* makes it possible to measure
starspot distributions with Approximate Bayesian Computation, for example.

.. toctree::
  :maxdepth: 2

  fleck/installation.rst
  fleck/gettingstarted.rst
  fleck/details.rst
  fleck/index.rst
