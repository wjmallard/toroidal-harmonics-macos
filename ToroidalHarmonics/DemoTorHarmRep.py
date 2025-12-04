# Written by Vassili Savinov on 25/11/2018
# demonstrate the use of the toroidal harmonic decomposition

# we need a suitable candidate for the decomposition. Why not take
# a configuration of charges, and compute the scalar potential from them?

import numpy as np
import numpy.linalg as npla
from TorHarmRep import TorHarmRep
import pylab as pl
import matplotlib.colors as mpc

# start with one charge, the scalar potential due to
# obs_pos a 3-number positiion of the observer
# ch_eps=charge_epsilon takes care of the units
def single_charge_scalar_pot(obs_pos_x, obs_pos_y, obs_pos_z,
                             ch_pos_x=0.0, ch_pos_y=0.0, ch_pos_z=0.0,
                             ch_eps=1.0):

    return (ch_eps/(4.0*np.pi))/np.sqrt( (obs_pos_x-ch_pos_x)**2 + (obs_pos_y-ch_pos_y)**2 + (obs_pos_z-ch_pos_z)**2 )

# convert toroidal coordinate position to conventioal position
# a_const a necessary numerical pre-factor for the conversion
def tor_coord_to_cart(eta, theta, phi, a_const = 1.0):

    x_coord = a_const * np.sinh(eta) * np.cos(phi) / (np.cosh(eta) - np.cos(theta))
    y_coord = a_const * np.sinh(eta) * np.sin(phi) / (np.cosh(eta) - np.cos(theta))
    z_coord = a_const * np.sin(theta) / (np.cosh(eta) - np.cos(theta))

    return x_coord, y_coord, z_coord

######## get the decomposition for the single charge
# prep the point-mesh
theta_counts = 12
theta_range = np.array(range(theta_counts))*(2*np.pi/theta_counts)-np.pi
phi_counts = 12
phi_range = np.array(range(phi_counts))*(2*np.pi/phi_counts)
eta_range = np.linspace(0.1, 3.0, 5, dtype=np.float64)
#
THETA, PHI, ETA = np.meshgrid(theta_range, phi_range, eta_range, indexing='ij')
X, Y, Z = tor_coord_to_cart(ETA, THETA, PHI)

print('min rad = %.3e, max rad = %.3e' % (np.min(np.sqrt(X**2 + Y**2 + Z**2)), np.max(np.sqrt(X**2 + Y**2 + Z**2))) )

# get the potiential
scal_pot_mat = single_charge_scalar_pot(X, Y, Z, ch_pos_x=1.0, ch_pos_y=0.0, ch_pos_z=0.0)

# get decomposition
thr=TorHarmRep(theta_counts, phi_counts, eta_range, scal_pot_mat)

max_val=10*np.log10(np.max([np.max(np.abs(thr.aPMat)), np.max(np.abs(thr.aQMat))]))
col_abs_norm = mpc.Normalize(vmin=max_val-40,
                             vmax=max_val)
col_pha_norm = mpc.Normalize(vmin=-180, vmax=180)

pl.style.use('dark_background')

pl.figure(1)
pl.subplot(221)
pl.imshow(10*np.log10(np.abs(thr.aPMat)), norm=col_abs_norm, cmap=pl.cm.hot)
pl.colorbar().set_label('$\\left|a^{(1)}\\right|$ (dB)')
pl.xlabel('n - order')
pl.ylabel('m - order')
pl.yticks(np.arange(0, thr.aQMat.shape[0], 2))
pl.xticks(np.arange(0, thr.aQMat.shape[1], 2))
#
pl.subplot(222)
pl.imshow(np.angle(thr.aPMat, deg=True), norm=col_pha_norm, cmap=pl.cm.jet)
pl.colorbar().set_label('$arg\\left(a^{(1)}\\right)$ (deg)')
pl.xlabel('n - order')
pl.ylabel('m - order')
pl.yticks(np.arange(0, thr.aQMat.shape[0], 2))
pl.xticks(np.arange(0, thr.aQMat.shape[1], 2))
#
pl.subplot(223)
pl.imshow(10*np.log10(np.abs(thr.aQMat)), norm=col_abs_norm, cmap=pl.cm.hot)
pl.colorbar().set_label('$\\left|a^{(2)}\\right|$ (dB)')
pl.xlabel('n - order')
pl.ylabel('m - order')
pl.yticks(np.arange(0, thr.aQMat.shape[0], 2))
pl.xticks(np.arange(0, thr.aQMat.shape[1], 2))
#
pl.subplot(224)
pl.imshow(np.angle(thr.aQMat, deg=True), norm=col_pha_norm, cmap=pl.cm.jet)
pl.colorbar().set_label('$arg\\left(a^{(2)}\\right)$ (deg)')
pl.xlabel('n - order')
pl.ylabel('m - order')
pl.yticks(np.arange(0, thr.aQMat.shape[0], 2))
pl.xticks(np.arange(0, thr.aQMat.shape[1], 2))


pl.tight_layout()
pl.show()
