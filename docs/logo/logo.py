"""
Generate the fleck logo!
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image

# Save the logo here:
logo_dir = os.path.dirname(__file__)
uncropped_svg_path = os.path.join(logo_dir, 'logo_uncropped.svg')
cropped_svg_path = os.path.join(logo_dir, 'logo.svg')
png_path = os.path.join(logo_dir, 'logo.png')
ico_path = os.path.join(logo_dir, 'logo.ico')

circles = [
    Circle((1, 0), radius=1, color='#ff8d00'),
    Circle((0.4, 0), radius=0.65, color='#f9b259'),
    Circle((0, 0), radius=0.4, color='#ffd5a2')
]

fig, ax = plt.subplots(figsize=(6, 4), dpi=200)

for circle in circles:
    ax.add_patch(circle)

ax.set(
    xlim=[-0.5, 2.2],
    ylim=[-1.2, 1.2],
    aspect='equal'
)
ax.axis('off')

savefig_kwargs = dict(
    pad_inches=0, transparent=True
)

fig.savefig(uncropped_svg_path, **savefig_kwargs)

# PNG will be at *high* resolution:
fig.savefig(png_path, dpi=800, **savefig_kwargs)

# This is the default matplotlib SVG configuration which can't be easily tweaked:
default_svg_dims = 'width="144pt" height="144pt" viewBox="0 0 144 144"'

# This is a hand-tuned revision to the SVG file that crops the bounds nicely:
custom_svg_dims = 'width="75pt" height="75pt" viewBox="31 37 75 75"'

# Read the uncropped file, replace the bad configuration with the custom one:
with open(uncropped_svg_path, 'r') as svg:
    cropped_svg_source = svg.read().replace(
        default_svg_dims, custom_svg_dims
    )

# Write out the cropped SVG file:
with open(cropped_svg_path, 'w') as cropped_svg:
    cropped_svg.write(cropped_svg_source)

# Delete the uncropped SVG:
os.remove(uncropped_svg_path)

# Convert the PNG into an ICO file:
img = Image.open(png_path)
img.save(ico_path)