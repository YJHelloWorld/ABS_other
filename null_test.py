import pysm
import numpy as np
import healpy as hp
from pysm.nominal import models
from pysm.common import convert_units
import matplotlib.pyplot as plt
import copy
import camb
from camb import model, initialpower
from numpy import linalg as LA
from functions import *
nside = 256; L = lmax = 600; Q = 30; Nf = 10
nu = np.array([30., 43., 75., 90.,108,129, 155., 223., 268., 321])
beams = np.array([28.3, 22.2, 10.7, 9.5, 7.9, 7.4, 6.2, 3.6, 3.2, 2.6])

beams_30 = np.ones(10)*28.3
Sens_P = np.array([12.4, 7.9, 4.2, 2.8, 2.3, 2.1, 1.8, 4.5,3.1, 4.2]) 
coefficients = convert_units("uK_RJ", "uK_CMB", nu)
# noise_RJ = noise*coefficients

npix = 12*nside**2
pix_amin2 = 4. * np.pi / float(npix) * (180. * 60. / np.pi) ** 2 
# #area of each pixel in unit of amin2.# solid angle per pixel in amin2
# """sigma_pix_I/P is std of noise per pixel. It is an array of length equal to the number of input maps."""
sigma_pix_I = np.sqrt(Sens_P ** 2 / pix_amin2)
sigma_pix_P = np.sqrt(Sens_P ** 2 / pix_amin2)
SamNum = 100
for i in range(100, SamNum + 1):
    noise = np.random.randn(len(Sens_P), 3, npix)
    noise[:, 0, :] *= sigma_pix_I[:, None]
    noise[:, 1, :] *= sigma_pix_P[:, None]
    noise[:, 2, :] *= sigma_pix_P[:, None]
    if i == SamNum:
        break
    noise = deconv(noise)
    noise_B = QU2EB(noise)
    noise_ps = power_spectrum(noise_B, 0, lmax = lmax)
    np.save('./noise/noise_power_spctrum_%s.npy'%(i+1), noise_ps)

nl=[];Nl_std=[]
Nl = np.zeros((lmax,Nf, Nf ))
for n in range(SamNum):
    n_diag=[]
    nl_i = np.load('./noise/noise_power_spctrum_%s.npy'%( n+1))
    ###nl_i = bin_l(nl_i) ##
    for i in range(lmax): ##
       n_diag.append(np.diag(nl_i[i])) ##   
    Nl = Nl + nl_i[0:600]                           #
   
    nl.append(n_diag) ##
for i in range(lmax): ##
    nl_s=[];nl_std=np.empty(Nf) ##
    for n in range(SamNum): ##
        nl_s.append(nl[n][i]) ##
    tran = np.array(nl_s).T   ##
    for f in range(Nf): ## 
       nl_std[f] = np.std(tran[f]) ##
    Nl_std.append(nl_std) ##
    
np.save('./noise/noise_power_spectrum_RMS_600.npy',Nl_std) ##
 
Nl = Nl/SamNum                                 #
np.save('./noise/noise_power_spectrum_real.npy',Nl)  #



# #Beam happens inplace...

s1_config = models("s1", nside)
d1_config = models("d1", nside)
sky_config = {
        'synchrotron' : s1_config,
        'dust' : d1_config,
    }   
sky = pysm.Sky(sky_config)


cl_real = produce_cl(0.05, lmax)
np.random.seed(1234)
cmb_map = hp.synfast(cl_real, nside, new = True, verbose = False)

total_map = convert_unit(sky.signal()(nu)) #+ cmb_map; # cmb_map = convert_unit(sky.cmb(nu))
total_map = smooth(total_map);  ##total_map and cmb_map are smoothed maps.

total_wt_noise = total_map + noise
total_wt_noise = deconv(total_wt_noise)
B_maps_total = QU2EB(total_wt_noise)
total_ps = power_spectrum(B_maps_total, 0, lmax)
np.save('./total_power_spectrum.npy', total_ps)


#cmb_map =  hp.smoothing(cmb_map, fwhm = beams_30[-1]/60/180*np.pi, verbose = False)
#alm_cmb = hp.map2alm(cmb_map)
#CMB_B_maps = hp.alm2map(alm_cmb[2], nside = nside, verbose = False)
#cl = hp.anafast(CMB_B_maps, lmax=lmax)
#np.save('./cmb_power_spectrum.npy', cl)





