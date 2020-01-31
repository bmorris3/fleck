---
title: 'fleck: Fast approximate light curves for starspot rotational modulation'
tags:
  - Python
  - astronomy
  - stellar astrophysics
  - starspots
authors:
  - name: Brett M. Morris
    orcid: 0000-0003-2528-3409
    affiliation: "1"
affiliations:
 - name: Center for Space and Habitability, University of Bern, 
         Gesellschaftsstrasse 6, CH-3012, Bern, Switzerland
   index: 1
date: 31 January 2020
bibliography: paper.bib

aas-doi: 10.3847/xxxxx
aas-journal: Astrophysical Journal
---

# Summary

Stars are born rapidly rotating, and dappled with dark starspots in their 
photospheres. Starspots are regions of intense magnetic fields which dominate 
over local convective motions to produce dim, cool regions in stellar 
photospheres. Starspot coverage shrinks from stellar youth into middle age.

In the associated paper submitted to AAS Journals, we show that *ensembles* of 
light curves of stars in young clusters can be used to constrain their spot 
distributions. One can imagine that photometric surveys of young clusters are 
essentially observing the same star at many different inclinations, allowing us 
to marginalize over the unknown inclinations of the individual stars if we model 
their light curves as a population. 

``fleck`` is a pure Python software package for simulating rotational modulation 
of stars due to starspots, which we use to overcome the degeneracies and 
determine starspot coverages accurately for a sample of young stars. ``fleck`` 
simulates starspots as circular dark regions on the surfaces of rotating stars, 
accounting for foreshortening towards the limb, and limb darkening. The 
software is an efficient, vectorized iteration of earlier codes used in 
@Morris:2018 and @Morris:2019. 

The user supplies the latitudes, longitudes and radii of spots, and the stellar 
inclinations from which each star is viewed, and ``fleck`` takes advantage of 
efficient array broadcasting with ``numpy`` to  return approximate light curves 
[@Numpy:2011]. For example, the present algorithm can compute rotational 
modulation curves sampled at ten points throughout the rotation of each star 
for one million stars, with two unique spots each, all viewed at unique 
inclinations, in about 10 seconds on a 2.5 GHz Intel Core i7 processor. 
This rapid computation of light curves en masse makes it possible to measure 
starspot distributions with Approximate Bayesian Computation.

The mathematical formalism of the ``fleck`` algorithm is detailed in the 
software's documentation. 

The ``fleck`` package is built on the ``astropy`` package template 
[@Astropy:2018].

# Acknowledgements

We acknowledge valuable vectorization conversations with Erik Tollerud.

# References

