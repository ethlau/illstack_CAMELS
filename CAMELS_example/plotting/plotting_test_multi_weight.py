import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy             as np
from scipy.interpolate import *
from decimal import Decimal
sys.path.append('/home/jovyan')
sys.path.append('/home/jovyan/illustris_python')

import illustris_python as il

home='/home/jovyan/home/illstack/CAMELS_example/'
 

simulations=['1P_22']
#snap=red_dict_tng.keys()  
snap=['024']
cut_color=0.6 
G=6.67e-11*1.989e30/((3.086e19)**3) #G in units kpc^3/(Msol*s^2)

ptype='pres'
color_cut=False
mh_cut=True

#------------------------------------------------------------
def mhalo_cut(mh_low,mh_high,mstar,mh,rh,sfr,GroupFirstSub,val_pres,val_dens,val_metal,bins):
    idx=np.where((mh > mh_low) & (mh < mh_high))
    mstar,mh,rh,sfr,GroupFirstSub=mstar[idx],mh[idx],rh[idx],sfr[idx],GroupFirstSub[idx]
    nprofs=len(mh)
    val_pres,val_dens,val_metal=val_pres[idx,:],val_dens[idx,:],val_metal[idx,:]
    val_pres,val_dens,val_metal=np.reshape(val_pres,(nprofs,bins)),np.reshape(val_dens,(nprofs,bins)),np.reshape(val_metal,(nprofs,bins))
    return mstar,mh,rh,sfr,GroupFirstSub,val_pres,val_dens,nprofs,val_metal
    
def outer_cut_multi(outer_cut,x,arr):
    idx=np.where(x <= outer_cut)
    idx=np.array(idx[0])
    x,arr=x[idx],arr[:,idx]
    return x,arr

def inner_cut_multi(inner_cut,x,arr):
    idx=np.where(x >= inner_cut)
    idx=np.array(idx[0])
    x,arr=x[idx],arr[:,idx]
    return x,arr

def get_errors(arr):
    arr=np.array(arr)
    percent_84=np.apply_along_axis(lambda v: np.percentile(v[np.nonzero(v)],84),0,arr)
    percent_50=np.apply_along_axis(lambda v: np.percentile(v[np.nonzero(v)],50),0,arr)
    percent_16=np.apply_along_axis(lambda v: np.percentile(v[np.nonzero(v)],16),0,arr)
    errup=percent_84-percent_50
    errlow=percent_50-percent_16

    std_arr=[]
    for i in range(arr.shape[1]): #for every radial bin
        std_arr.append(np.std(np.apply_along_axis(lambda v: np.log10(v[np.nonzero(v)]),0,arr[:,i])))
    std=np.array(std_arr)
    return errup,errlow,std

stacks1=np.load(home+'NPZ_files/profs_IllustrisTNG_1P_22_033_testmulti.npz',allow_pickle=True)
stacks2=np.load(home+'NPZ_files/profs_IllustrisTNG_1P_22_033_test_weight.npz',allow_pickle=True)
z              = 0.0 
val1            = stacks1['val']
val_dens1=val1[0,:]
val_pres1=val1[1,:]
val_metal1=val1[2,:]

print(np.shape(val1))
print(np.shape(val_dens1))
'''
bins1          = stacks1['nbins']
r1              = stacks1['r']
nprofs1         = stacks1['nprofs']
mh1             = stacks1['M_Crit200'] #units 1e10 Msol/h, M200c
rh1             = stacks1['R_Crit200'] #R200c
GroupFirstSub1  = stacks1['GroupFirstSub']
sfr1            = stacks1['sfr'] #Msol/yr
mstar1          = stacks1['mstar'] #1e10 Msol/h

val2           = stacks2['val']
val_dens2=val2[0,:]
val_pres2=val2[1,:]
val_metal2=val2[2,:]
bins2          = stacks2['nbins']
r2              = stacks2['r']
nprofs2         = stacks2['nprofs']
mh2             = stacks2['M_Crit200'] #units 1e10 Msol/h, M200c
rh2             = stacks2['R_Crit200'] #R200c
GroupFirstSub2  = stacks2['GroupFirstSub']
sfr2            = stacks2['sfr'] #Msol/yr
mstar2          = stacks2['mstar'] #1e10 Msol/h


comoving_factor=1.0+z
h=0.6711
mh1       *= 1e10
mstar1    *= 1e10
mh1       /= h
mstar1    /= h
rh1       /= h
rh1      /= comoving_factor
val_dens1 *= 1e10 * h**2
val_pres1 *= 1e10 * h**2
val_pres1 /= (3.086e16*3.086e16)
val_dens1 *= comoving_factor**3
val_pres1 *= comoving_factor**3
#for unscaled
r1 /= h
r1 /= comoving_factor

mh2       *= 1e10
mstar2    *= 1e10
mh2       /= h
mstar2    /= h
rh2       /= h
rh2      /= comoving_factor
val_dens2 *= 1e10 * h**2
val_pres2 *= 1e10 * h**2
val_pres2 /= (3.086e16*3.086e16)
val_dens2 *= comoving_factor**3
val_pres2 *= comoving_factor**3
#for unscaled
r2 /= h
r2 /= comoving_factor

#---------------------------------------------------------------    
mstar1,mh1,rh1,sfr1,GroupFirstSub1,val_pres1,val_dens1,nprofs1,val_metal1=mhalo_cut(10**11,10**12,mstar1,mh1,rh1,sfr1,GroupFirstSub1,val_pres1,val_dens1,val_metal1,bins1)

mstar2,mh2,rh2,sfr2,GroupFirstSub2,val_pres2,val_dens2,nprofs2,val_metal2=mhalo_cut(10**11,10**12,mstar2,mh2,rh2,sfr2,GroupFirstSub2,val_pres2,val_dens2,val_metal2,bins2)
    
r_mpc1=r1/1.e3
r_mpc2=r2/1.e3
        
r_mpc_cut1,val_dens1=outer_cut_multi(5,r_mpc1,val_dens1)
r_mpc_cut12,val_dens1=inner_cut_multi(1.e-2,r_mpc_cut1,val_dens1)

r_mpc_cut2,val_dens2=outer_cut_multi(5,r_mpc2,val_dens2)
r_mpc_cut22,val_dens2=inner_cut_multi(1.e-2,r_mpc_cut2,val_dens2)


    
mean_unnorm_dens1=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_dens1)
mean_unnorm_dens2=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_dens2)

plt.loglog(r_mpc_cut12,mean_unnorm_dens1,color='r',label='old')
plt.loglog(r_mpc_cut22,mean_unnorm_dens2,color='b',label='new')
plt.ylabel(r"$\rho_{gas}(g/cm^3)$",size=14)
plt.xlabel(r'R (Mpc)',size=12)
plt.legend()

plt.savefig(home+'Figures/test_multi_weight.png')
    
'''    