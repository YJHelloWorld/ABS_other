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
nu = np.array([30., 43., 75., 90.,108,129, 155., 223., 268., 321])
#nu = np.array([30., 43., 75., 90.,155., 223.,321])
beams_30 = np.ones(len(nu))*28.3
#beams = np.array([28.3, 22.2, 10.7, 9.5, 6.2, 3.6,2.6])
#nu = np.array([30., 43., 75., 90.,108,129, 155., 223., 268., 321])
coefficients = convert_units("uK_RJ", "uK_CMB", nu)
beams = np.array([28.3, 22.2, 10.7, 9.5, 7.9, 7.4, 6.2, 3.6, 3.2, 2.6])
nside = 256
def convert_unit(maps):
    for i in range(0,len(maps)):
        maps[i] = maps[i]*coefficients[i]
    return maps

def smooth(maps):
    for i in range(len(maps)):
        for j in range(0,3):                        ###for Q and U. Exclude I.
            maps[i,j] = hp.smoothing(maps[i,j], fwhm = beams[i]/60/180*np.pi, verbose = False)
    return maps

def deconv(maps):
    for i in range(len(maps)):
        for j in range(0,3):
            maps[i,j] = hp.sphtfunc.decovlving(maps[i,j], fwhm = beams[i]/60/180*np.pi, verbose = False)
            maps[i,j] = hp.smoothing(maps[i,j], fwhm = beams_30[i]/60/180*np.pi, verbose = False)
    return maps

#def normalize(maps):
    #total maps with noise
    

# QU to EB 
def QU2EB(maps):
    B_maps = np.zeros((len(maps), 12*nside**2))
    for i in range(len(maps)):
        alm_total = hp.map2alm(maps[i])
        B_maps[i] = hp.alm2map(alm_total[2], nside = nside, verbose = False)
    return B_maps

def power_spectrum(maps,R, lmax):
    n_f = len(maps)
    cl = []; Cl = np.zeros((lmax+1, n_f, n_f))
    for i in range(n_f):
        for j in range(n_f):
            cross_ps = hp.anafast(maps[i], maps[j], lmax = lmax, gal_cut=R)
            cl.append(cross_ps)
    cl = np.array(cl)
    print cl.shape
    for l in range(lmax+1):
        Cl[l, 0:n_f , 0:n_f] = cl[:,l].reshape(n_f, n_f)
    return Cl

def bin_l(cl, L, Q):
    bin_averages = []
    for l in range(L):
        cl[l] = l*(l+1)/2/np.pi*(cl[l])    
    for q in range(Q):
        bin_averages.append(sum(cl[q*L/Q:((q+1)*L/Q)]/(L/Q)))
    return bin_averages

def produce_cl(r_value, lmax):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06);pars.WantTensors = True
    results = camb.get_transfer_functions(pars)

    inflation_params = initialpower.InitialPowerParams()
    inflation_params.set_params(ns=0.96, r=r_value)
    results.power_spectra_from_transfer(inflation_params)
    cl = results.get_total_cls(lmax, CMB_unit='muK', raw_cl = True)
    Cl = cl.T
    return Cl
