import matplotlib.pyplot as plt
from fleck import Star, generate_spots

spot_contrast = 0.7
u_ld = [0.5079, 0.2239]

n_phases = 30
n_inclinations = 1000
n_spots = 3

spot_radius = 0.23  # Rspot/Rstar
min_latitude = 70   # deg
max_latitude = 90   # deg

lons, lats, radii, inc_stellar = generate_spots(min_latitude, max_latitude,
                                                spot_radius, n_spots,
                                                n_inclinations=n_inclinations)

stars = Star(spot_contrast=spot_contrast, n_phases=n_phases, u_ld=u_ld)
lcs = stars.light_curve(lons, lats, radii, inc_stellar)

smoothed_amps = 100 * lcs.ptp(axis=0)

fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ax[0].plot(stars.phases, lcs, alpha=0.1, color='k')
ax[0].set(xlabel='Phase', ylabel='Flux')

ax[1].hist(smoothed_amps, density=True)
ax[1].set(xlabel='Smoothed amp', ylabel='Freq')

plt.show()