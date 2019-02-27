import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from fleck import Stars

spot_contrast = 0.7
n_phases = 30
u_ld = [0.5079, 0.2239]
n_inclinations = 1000
n_spots = 3
spot_radius = 0.23

inc_stellar = (180*np.random.rand(n_inclinations) - 90) * u.deg
radii = spot_radius * np.ones((n_spots, n_inclinations))
lats = (20*np.random.rand(n_spots, n_inclinations) + 70) * u.deg
lons = 360*np.random.rand(n_spots, n_inclinations) * u.deg

stars = Stars(spot_contrast=spot_contrast, n_phases=n_phases, u_ld=u_ld)
lcs = stars.light_curve(lons, lats, radii, inc_stellar)
smoothed_amps = 100 * lcs.ptp(axis=0)

fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ax[0].plot(stars.phases, lcs, alpha=0.1, color='k')
ax[0].set(xlabel='Phase', ylabel='Flux')

ax[1].hist(smoothed_amps, density=True)
ax[1].set(xlabel='Smoothed amp', ylabel='Freq')

plt.show()