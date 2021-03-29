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
snap=['033']
cut_color=0.6 
G=6.67e-11*1.989e30/((3.086e19)**3) #G in units kpc^3/(Msol*s^2)

ptype='pres'
color_cut=False
mh_cut=True

fig,axes=plt.subplots(3,4,sharey='row',figsize=(20,10)) #20,5
ax1=axes[0,0]
ax2=axes[0,1]
ax3=axes[0,2]
ax4=axes[0,3]
ax5=axes[1,0]
ax6=axes[1,1]
ax7=axes[1,2]
ax8=axes[1,3]
ax9=axes[2,0]
ax10=axes[2,1]
ax11=axes[2,2]
ax12=axes[2,3]
ax_arr=[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12]

mh_low_arr=[10**10.0,10**11.0,10**12.0,10**13.0]
mh_high_arr=[10**11.0,10**12.0,10**13.0,10**14.0]
mh_low_pow=['10','11','12','13']
mh_high_pow=['11','12','13','14']

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

def extract(simulation,snap): #extract the quantities,adjust as necessary
    stacks=np.load(home+'NPZ_files/profs_IllustrisTNG_'+simulation+'_033_testmulti.npz',allow_pickle=True)
    z              = 0.0 
    val            = stacks['val']
    val_dens=val[0,:]
    val_pres=val[1,:]
    val_metal=val[2,:]
    bins           = stacks['nbins']
    r              = stacks['r']
    nprofs         = stacks['nprofs']
    mh             = stacks['M_Crit200'] #units 1e10 Msol/h, M200c
    rh             = stacks['R_Crit200'] #R200c
    GroupFirstSub  = stacks['GroupFirstSub']
    sfr            = stacks['sfr'] #Msol/yr
    mstar          = stacks['mstar'] #1e10 Msol/h
    return z,val_dens,bins,r,val_pres,nprofs,mh,rh,GroupFirstSub,sfr,mstar,val_metal

def correct(z,h,mh,mstar,rh,val_dens,val_pres,r): #correct all h and comoving factors
    comoving_factor=1.0+z
    mh       *= 1e10
    mstar    *= 1e10
    mh       /= h
    mstar    /= h
    rh       /= h
    rh      /= comoving_factor
    val_dens *= 1e10 * h**2
    val_pres *= 1e10 * h**2
    val_pres /= (3.086e16*3.086e16)
    val_dens *= comoving_factor**3
    val_pres *= comoving_factor**3
    #for unscaled
    r /= h
    r /= comoving_factor
    return mh,mstar,rh,val_dens,val_pres,r

def normalize_pressure(nprofs,rh,r,mh,rhocrit_z,omegab,omegam,val_pres):
    x_values=[]
    norm_pres=[]
    for n in np.arange(nprofs):
        #r200c=(3./4./np.pi/rhombar*mh[i]/200)**(1./3.)
        r200c=rh[n]
        x_values.append(r/r200c)
        P200c=200.*G*mh[n]*rhocrit_z*omegab/(omegam*2.*r200c)
        pressure=val_pres[n,:]
        pressure_divnorm=pressure/P200c
        norm_pres.append(pressure_divnorm)
    mean_xvals=np.mean(x_values, axis=0)
    return mean_xvals,np.array(norm_pres)
    
#---------------------------------------------------------------    

for i in np.arange(len(mh_low_arr)):
    mh_low=mh_low_arr[i]
    mh_high=mh_high_arr[i]
    axt=ax_arr[i]
    axm=ax_arr[i+4]
    axb=ax_arr[i+8]
    
    fidu_dens=[]
    fidu_pres=[]
    fidu_metal=[]
    for j in np.arange(len(simulations)):
        for k in snap:
            z,val_dens,bins,r,val_pres,nprofs,mh,rh,GroupFirstSub,sfr,mstar,val_metal=extract(simulations[j],k)
            omegab=0.049
            h=0.6711
            omegam,sigma8=np.loadtxt('/home/jovyan/Simulations/IllustrisTNG/'+simulations[j]+'/CosmoAstro_params.txt',usecols=(1,2),unpack=True)
            omegalam=1.0-omegam
            rhocrit=2.775e2
            rhocrit_z=rhocrit*(omegam*(1+z)**3+omegalam)
            
            mh,mstar,rh,val_dens,val_pres,r=correct(z,h,mh,mstar,rh,val_dens,val_pres,r)
    
            if mh_cut==True:
                    mstar,mh,rh,sfr,GroupFirstSub,val_pres,val_dens,nprofs,val_metal=mhalo_cut(mh_low,mh_high,mstar,mh,rh,sfr,GroupFirstSub,val_pres,val_dens,val_metal,bins)
    
            r_mpc=r/1.e3
        

    
            r_mpc_cut,val_dens=outer_cut_multi(5,r_mpc,val_dens)
            r_mpc_cut2,val_dens=inner_cut_multi(1.e-2,r_mpc_cut,val_dens)
            r_mpc_cut,val_pres=outer_cut_multi(5,r_mpc,val_pres)
            r_mpc_cut2,val_pres=inner_cut_multi(1.e-2,r_mpc_cut,val_pres)
            r_mpc_cut,val_metal=outer_cut_multi(5,r_mpc,val_metal)
            r_mpc_cut2,val_metal=inner_cut_multi(1.e-2,r_mpc_cut,val_metal)

    
            mean_unnorm_dens=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_dens)
            mean_unnorm_pres=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_pres)
            mean_unnorm_metal=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_metal)

            axt.loglog(r_mpc_cut2,mean_unnorm_dens,label='nhalo:%i'%(nprofs))
            axm.loglog(r_mpc_cut2,mean_unnorm_pres)
            axb.loglog(r_mpc_cut2,mean_unnorm_metal)
            axt.set_title(r"Mass $%s \leq M_\odot \leq %s$"%(mh_low_pow[i],mh_high_pow[i]),size=14)

plt.suptitle(r'$\Omega_m = 0.3, \sigma_8 = 0.8$')
ax1.set_ylabel(r"$\rho_{gas}(g/cm^3)$",size=14)
ax5.set_ylabel(r"$P_{th} (g/cm/s^2)$",size=14)
ax9.set_ylabel(r"Metals",size=14)
ax9.set_xlabel(r'R (Mpc)',size=12)
ax10.set_xlabel(r'R (Mpc)',size=12)
ax11.set_xlabel(r'R (Mpc)',size=12)
ax12.set_xlabel(r'R (Mpc)',size=12)

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()

plt.subplots_adjust(wspace=0,hspace=0.1)
#plt.savefig(home+'Figures/'+ptype+'_033_diff_mass_comb.png')
plt.savefig(home+'Figures/test_multi.png')
#plt.show()
    
    