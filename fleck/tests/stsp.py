# Licensed under the MIT License - see LICENSE.rst
import tempfile
import os
import subprocess

import numpy as np

from astropy.table import Table
import astropy.units as u
from astropy.constants import G, M_sun, R_sun


# path to STSP executable:
stsp_executable = os.getenv('STSP_PATH')

infile_template_l = ("""#PLANET PROPERTIES
1							; Number of planets
{t0:2.10f}					; T0, epoch         (middle of first transit) in days.
{period:2.10f}				; Planet Period      (days)
{depth:2.10f}				; (Rp/Rs)^2         (Rplanet / Rstar )^ 2
{duration:2.10f}			; Duration (days)   (physical duration of transit, not used)
{b:2.10f}					; Impact parameter  (0= planet cross over equator)
{inclination:2.10f}			; Inclination angle of orbit (90 deg = transit equator)
{lam:2.10f}					; Lambda of orbit (0 deg = orbital axis along z-axis)
{ecosw:2.10f}			; ecosw
{esinw:2.10f}			; esinw
#STAR PROPERTIES
{rho_s:2.10f} 			; Mean Stellar density (Msun/Rsun^3)
{per_rot:2.10f}			; Stellar Rotation period (days)
4780					; Stellar Temperature
0.31					; Stellar metallicity
{tilt_from_z:2.10f}		; Tilt of stellar rotation axis down from z-axis (deg)
{nonlinear_ld}			; Limb darkening (4 coefficients)
{n_ld_rings:d}			; number of rings for limb darkening appoximation
#SPOT PROPERTIES
{n_spots}						; number of spots
{contrast}				; fractional lightness of spots (0.0=total dark, 1.0=no contrast)
#LIGHT CURVE
{model_path}			; lightcurve input data file
{start_time:2.10f}		; start time to start fitting the light curve
{lc_duration:2.10f}		; duration of light curve to fit (days)
{real_max:2.10f}		; real max of LC, 0 -> use downfrommax
0						; is light curve flattened (to zero) outside of transits?
#ACTION
l						; l= generate light curve from parameters
{spot_params}
1.00
""")

spot_params_template = """{spot_radius:2.10f}		; spot radius
{spot_theta:2.10f}		; theta
{spot_phi:2.10f}		; phi
"""


def quadratic_to_nonlinear(u1, u2):
    a1 = a3 = 0
    a2 = u1 + 2*u2
    a4 = -u2
    return (a1, a2, a3, a4)


def rho_star(transit_params):
    aRs = transit_params.a

    rho_s = 3*np.pi/(G*(transit_params.per*u.day)**2) * aRs**3
    rho_s = rho_s.to(M_sun/(4./3 * np.pi * R_sun**3))
    return rho_s.value


class STSP(object):
    def __init__(self, lc, transit_params, spot_params, outdir=None):
        """
        Parameters
        ----------
        lc : `LightCurve`
            Light curve object
        transit_params : `batman.TransitParams`
            Parameters for planet and star
        spot_params : `numpy.ndarray`
            [r, theta, phi] for each spot to model with STSP
        outdir : str
            Directory to write temporary outputs into
        """
        self.lc = lc
        self.transit_params = transit_params
        self.spot_params = np.atleast_2d(spot_params)

    def stsp_lc(
            self,
            contrast=0.7,
            n_ld_rings=100,
            real_max=1,
            verbose=False,
            stsp_exec=None):

        if stsp_exec is None:
            stsp_exec = stsp_executable

        with tempfile.TemporaryDirectory() as tmp_dir:
            times = self.lc.times.jd
            fluxes = np.ones_like(times)
            model_path = os.path.join(tmp_dir, 'model_lc.dat')
            np.savetxt(model_path,
                       np.vstack([times, fluxes,
                                  0 * fluxes]).T,
                       fmt='%1.10f', delimiter='\t', header='stspinputs')

            # Calculate parameters for STSP:
            eccentricity = self.transit_params.ecc
            omega = self.transit_params.w
            ecosw = eccentricity * np.cos(np.radians(omega))
            esinw = eccentricity * np.sin(np.radians(omega))
            start_time = times[0]
            lc_duration = times[-1] - times[0]
            nonlinear_ld = quadratic_to_nonlinear(*self.transit_params.u)
            nonlinear_ld_string = ' '.join(map("{0:.5f}".format, nonlinear_ld))

            # get spot parameters sorted out
            spot_params_str = spot_params_to_string(self.spot_params)

            # Stick those values into the template file
            b = (
                self.transit_params.a *
                np.cos(np.radians(self.transit_params.inc))
            )
            duration = duration_t14(self.transit_params)
            in_file_text = infile_template_l.format(
                contrast=contrast,
                period=self.transit_params.per,
                ecosw=ecosw,
                esinw=esinw,
                lam=self.transit_params.lam,
                tilt_from_z=90-self.transit_params.inc_stellar,
                start_time=start_time,
                lc_duration=lc_duration,
                real_max=real_max,
                per_rot=self.transit_params.per_rot,
                rho_s=rho_star(self.transit_params),
                depth=self.transit_params.rp**2,
                duration=duration,
                t0=self.transit_params.t0,
                b=b,
                inclination=self.transit_params.inc,
                nonlinear_ld=nonlinear_ld_string,
                n_ld_rings=n_ld_rings,
                spot_params=spot_params_str[:-1],
                n_spots=self.spot_params.shape[0],
                model_path=os.path.basename(model_path)
            )

            # Write out the `.in` file
            with open(os.path.join(tmp_dir, 'test.in'), 'w') as in_file:
                in_file.write(in_file_text)

            # Run STSP
            old_cwd = os.getcwd()
            os.chdir(tmp_dir)
            completed = subprocess.run(
                [stsp_exec, 'test.in'],
                capture_output=True
            )
            if verbose:
                print(completed.stdout)
            os.chdir(old_cwd)

            # Read the outputs
            lc_out = os.path.join(tmp_dir, 'test_lcout.txt')
            tbl = Table.read(lc_out, format='ascii')
            # tbl.write('~/Desktop/tmp.csv', overwrite=True)
            stsp_times = np.array(tbl[tbl.colnames[0]])
            stsp_fluxes = np.array(tbl[tbl.colnames[3]])
        return stsp_times, stsp_fluxes


def spot_params_to_string(spot_params):
    spot_params_str = ""
    spot_params = np.atleast_2d(spot_params)
    for param_set in spot_params:
        spot_params_str += spot_params_template.format(
            spot_radius=param_set[0],
            spot_theta=param_set[1],
            spot_phi=param_set[2]
        )
    return spot_params_str


def duration_t14(transit_params):
    """transit 1-4 contact duration"""
    inc = np.radians(transit_params.inc)
    b = transit_params.a * np.cos(inc)
    return (
        transit_params.per / np.pi *
        np.arcsin(
            np.sqrt((1 - transit_params.rp)**2 - b**2) /
            (transit_params.a * np.sin(inc))
        ) *
        (1 - transit_params.ecc**2) ** 0.5 /
        (1 - transit_params.ecc *
         np.sin(np.radians(transit_params.w)))
    )
