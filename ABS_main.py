import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import copy
from numpy import linalg as LA
from functions import *
nside = 256; L = lmax = 600; Q = 30; Nf = 10 


loc = '.'
n = 1
D = np.load('%s/total_power_spectrum.npy'%(loc))
noise_ps = np.load('%s/noise/noise_power_spectrum_real.npy'%(loc))
CMB = np.load('%s/cmb_power_spectrum.npy'%(loc))  
D = bin_l(D, lmax, Q); CMB = bin_l(CMB, lmax, Q); noise_ps = bin_l(noise_ps, lmax, Q)  #*fr
noise = np.zeros_like(noise_ps);f = []
noise = np.load('%s/noise/noise_power_spectrum_RMS_600.npy'%(loc))#
noise = bin_l(noise, lmax, Q)

D_B = []; Evals = np.ones((Q,Nf)); Evals_all = [];Delta = 1e-3;  E_cut= 0.5;
for i in range(Q):
        f_q = np.ones(Nf)
        for j in range(Nf):
            #noise[i][j,j] = noise_ps[i][j,j]*np.sqrt(2.0/(((2*i+1)*fr)*L/Q)) # *fr
            f_q[j] = f_q[j]/np.sqrt(noise[i][j]) #,j
            #print noise[i][j]
        f.append(f_q)    
for l in range(Q):
    D[l] = D[l] - noise_ps[l]
    for i in range(Nf): 
        for j in range(Nf):
            D[l][i,j] = D[l][i,j]/np.sqrt(noise[l][i]*noise[l][j]) + Delta*f[l][i]*f[l][j] 
for l in range(Q): 
        e_vals,E = LA.eig(D[l])
        #if n == 49:
            #if l ==2:
                #print e_vals
        Evals[l,:] = e_vals        
        
        for i in range(Nf):
            E[:,i]=E[:,i]/LA.norm(E[:,i])**2  
            
        D_B_l = 0
        for i in range(Nf):
            if e_vals[i]>=E_cut:
                G_i = np.dot(f[l],E[:,i])
                D_B_l = D_B_l + (G_i**2/e_vals[i]) 
        D_B_l = 1/ D_B_l - Delta
        D_B.append(D_B_l)
        
ell = np.ones(Q)
for q in range(Q):
    ell[q] = (2*q+1)*L/Q/2
plt.ion()
fig1 = plt.figure(1, figsize=(8,5.3))
#plt.axis('off')
#frame1=fig1.add_axes((.1,.4,.8,.5))
plt.loglog(ell,D_B,'g-^',label = 'recovered CMB')
plt.plot(ell,CMB[0:Q],'r-x',label = 'real CMB')
plt.ylim(1e-10, 1)
plt.xlim(1,2000)
plt.xlabel(r'$\ell$'); plt.ylabel(r'$\mathcal{D}_{\ell}$ [$\mu$k$^2$]')
plt.text(400, 10**(-7.5), '$\Delta \ell = %s$'%(600/Q), fontsize =20)
plt.legend(loc = 'lower right')
#frame1.set_xticklabels([])

#frame2=fig1.add_axes((.1,.1,.8,.3))
#Ell = np.ones(Q); Mean = np.ones(Q)
#for i in range(0,Q):
#    Ell[i] = (2*i+1)*L/Q/2 
#    Mean[i] = abs((D_B[i])-CMB[i])/CMB[i]
#plt.loglog(Ell,100*Mean,'b-o')
#plt.ylabel(r'$\Delta \mathcal{D}_\ell$/$\mathcal{D}^{real}_{\ell} $ [%]');plt.xlabel(r'$\ell$')
#plt.xlim(1,2000)
#plt.show()
##plt.savefig('/home/yao/Desktop/Larissa/10_fre/B_mode_dlbin_%s.pdf'%(600/Q), format = 'pdf')
fig3 = plt.figure(3)
evals = Evals.T; evals_nega = np.ones_like(evals)
for i in range(Nf):
    for j in range(Q):
        if evals[i,j]<= 0:
            evals_nega[i,j] = abs(evals[i,j])
            evals[i,j] = None
        else:
            evals_nega[i,j] = None
for i in range(Nf):  
    x = np.arange(len(evals[i]))
    plt.scatter(ell,(evals[i][:]),color = 'g') #label = 'positive eigenvalues'
    plt.scatter(ell,evals_nega[i][:],color = 'r',marker='^') #,label = 'negative eigenvalues'
    plt.axhline(0.5,color = 'k')
    plt.yscale('log')
plt.xlabel(r'$\ell$');plt.ylabel(r'$\tilde{\lambda}_{\mu}$')
plt.xlim(-100,700)
plt.show()
raw_input("Please press key to exit")


