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
 

#simulations=['1P_22','1P_23','1P_24','1P_25','1P_26','1P_27','1P_28','1P_29','1P_30','1P_31','1P_32']
#simulations=['1P_33','1P_34','1P_35','1P_36','1P_37','1P_38','1P_39','1P_40','1P_41','1P_42','1P_43']

simulations=['1P_22']
ASN1=np.array([0.25,0.33,0.44,0.57,0.76,1.0,1.3,1.7,2.3,3.0,4.0])
colors=['r','orchid','orange','gold','lime','green','turquoise','b','navy','blueviolet','k']
#snap=red_dict_tng.keys()  
snap=['024']
cut_color=0.6 
G=6.67e-11*1.989e30/((3.086e19)**3) #G in units kpc^3/(Msol*s^2)

ptype='pres'
color_cut=False
mh_cut=True

fig,axes=plt.subplots(2,2,figsize=(12,10)) #20,5
ax1=axes[0,0]
ax2=axes[0,1]
ax3=axes[1,0]
ax4=axes[1,1]
ax_arr=[ax1,ax2,ax3,ax4]

mh_low_arr=[10**11.0]
mh_high_arr=[10**12.0]
mh_low_pow=['11']
mh_high_pow=['12']

#------------------------------------------------------------
def mhalo_cut(mh_low,mh_high,mstar,mh,rh,sfr,GroupFirstSub,val_pres,val_dens,val_metal_uw,val_metal_gmw,val_temp_uw,val_temp_gmw,val_metal_emm,val_temp_emm,bins):
    idx=np.where((mh > mh_low) & (mh < mh_high))
    mstar,mh,rh,sfr,GroupFirstSub=mstar[idx],mh[idx],rh[idx],sfr[idx],GroupFirstSub[idx]
    nprofs=len(mh)
    val_pres,val_dens,val_metal_uw,val_metal_gmw,val_temp_uw,val_temp_gmw,val_metal_emm,val_temp_emm=val_pres[idx,:],val_dens[idx,:],val_metal_uw[idx,:],val_metal_gmw[idx,:],val_temp_uw[idx,:],val_temp_gmw[idx,:],val_metal_emm[idx,:],val_temp_emm[idx,:]
    val_pres,val_dens,val_metal_uw,val_metal_gmw,val_temp_uw,val_temp_gmw,val_metal_emm,val_temp_emm=np.reshape(val_pres,(nprofs,bins)),np.reshape(val_dens,(nprofs,bins)),np.reshape(val_metal_uw,(nprofs,bins)),np.reshape(val_metal_gmw,(nprofs,bins)),np.reshape(val_temp_uw,(nprofs,bins)),np.reshape(val_temp_gmw,(nprofs,bins)),np.reshape(val_metal_emm,(nprofs,bins)),np.reshape(val_temp_emm,(nprofs,bins))
    return mstar,mh,rh,sfr,GroupFirstSub,val_pres,val_dens,nprofs,val_metal_uw,val_metal_gmw,val_temp_uw,val_temp_gmw,val_metal_emm,val_temp_emm
    
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
    stacks=np.load(home+'NPZ_files/profs_IllustrisTNG_'+simulation+'_024_9profs_SearchRad5.npz',allow_pickle=True)
    z              = 0.0 
    val            = stacks['val']
    val_dens=val[0,:]
    val_pres=val[1,:]
    val_metal_uw=val[2,:]
    val_metal_gmw=val[3,:]
    val_temp_uw=val[5,:]
    val_temp_gmw=val[6,:]
    val_metal_emm=val[7,:]
    val_temp_emm=val[8,:]
    bins           = stacks['nbins']
    r              = stacks['r']
    nprofs         = stacks['nprofs']
    mh             = stacks['M_Crit200'] #units 1e10 Msol/h, M200c
    rh             = stacks['R_Crit200'] #R200c
    GroupFirstSub  = stacks['GroupFirstSub']
    sfr            = stacks['sfr'] #Msol/yr
    mstar          = stacks['mstar'] #1e10 Msol/h
    return z,val_dens,bins,r,val_pres,nprofs,mh,rh,GroupFirstSub,sfr,mstar,val_metal_uw,val_metal_gmw,val_temp_uw,val_temp_gmw,val_metal_emm,val_temp_emm

def correct(z,h,mh,mstar,rh,val_dens,val_pres,r,val_temp_uw,val_temp_gmw,val_temp_emm): #correct all h and comoving factors
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
    val_temp_uw *= 1.e10
    val_temp_gmw*= 1.e10
    val_temp_emm*=1.e10
    #for unscaled
    r /= h
    r /= comoving_factor
    return mh,mstar,rh,val_dens,val_pres,r, val_temp_uw,val_temp_gmw,val_temp_emm

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
    #axt=ax_arr[i]
    #axm=ax_arr[i+4]
    #axb=ax_arr[i+8]
    
    fidu_dens=[]
    fidu_pres=[]
    fidu_metal=[]
    masses=[]
    for j in np.arange(len(simulations)):
        for k in snap:
            z,val_dens,bins,r,val_pres,nprofs,mh,rh,GroupFirstSub,sfr,mstar,val_metal_uw,val_metal_gmw,val_temp_uw,val_temp_gmw,val_metal_emm,val_temp_emm=extract(simulations[j],k)
            omegab=0.049
            h=0.6711
            omegam,sigma8=np.loadtxt('/home/jovyan/Simulations/IllustrisTNG/'+simulations[j]+'/CosmoAstro_params.txt',usecols=(1,2),unpack=True)
            omegalam=1.0-omegam
            rhocrit=2.775e2
            rhocrit_z=rhocrit*(omegam*(1+z)**3+omegalam)
            
            mh,mstar,rh,val_dens,val_pres,r,val_temp_uw,val_temp_gmw,val_temp_emm=correct(z,h,mh,mstar,rh,val_dens,val_pres,r,val_temp_uw,val_temp_gmw,val_temp_emm)
    
            if mh_cut==True:
                    mstar,mh,rh,sfr,GroupFirstSub,val_pres,val_dens,nprofs,val_metal_uw,val_metal_gmw,val_temp_uw,val_temp_gmw,val_metal_emm,val_temp_emm=mhalo_cut(mh_low,mh_high,mstar,mh,rh,sfr,GroupFirstSub,val_pres,val_dens,val_metal_uw,val_metal_gmw,val_temp_uw,val_temp_gmw,val_metal_emm,val_temp_emm,bins)
    
            r_mpc=r/1.e3
        

            masses.append(np.mean(mh))
            #r_mpc_cut,val_dens=outer_cut_multi(5,r_mpc,val_dens)
            #r_mpc_cut2,val_dens=inner_cut_multi(1.e-2,r_mpc_cut,val_dens)
            #r_mpc_cut,val_pres=outer_cut_multi(5,r_mpc,val_pres)
            #r_mpc_cut2,val_pres=inner_cut_multi(1.e-2,r_mpc_cut,val_pres)
            
            print("sim",simulations[j])
            print(r_mpc)
            r_mpc_cut,val_dens=outer_cut_multi(12,r_mpc,val_dens)
            r_mpc_cut2,val_dens=inner_cut_multi(3.e-4,r_mpc_cut,val_dens)           
            r_mpc_cut,val_pres=outer_cut_multi(12,r_mpc,val_pres)
            r_mpc_cut2,val_pres=inner_cut_multi(3.e-4,r_mpc_cut,val_pres)
            r_mpc_cut,val_metal_uw=outer_cut_multi(12,r_mpc,val_metal_uw)
            r_mpc_cut2,val_metal_uw=inner_cut_multi(3.e-4,r_mpc_cut,val_metal_uw)
            r_mpc_cut,val_metal_gmw=outer_cut_multi(12,r_mpc,val_metal_gmw)
            r_mpc_cut2,val_metal_gmw=inner_cut_multi(3.e-4,r_mpc_cut,val_metal_gmw)
            r_mpc_cut,val_temp_uw=outer_cut_multi(12,r_mpc,val_temp_uw)
            r_mpc_cut2,val_temp_uw=inner_cut_multi(3.e-4,r_mpc_cut,val_temp_uw)
            r_mpc_cut,val_temp_gmw=outer_cut_multi(12,r_mpc,val_temp_gmw)
            r_mpc_cut2,val_temp_gmw=inner_cut_multi(3.e-4,r_mpc_cut,val_temp_gmw)  
            
            #different for emm profiles
            r_mpc_cut_emm,val_metal_emm=inner_cut_multi(6.e-4,r_mpc,val_metal_emm)
            r_mpc_cut2_emm,val_metal_emm=outer_cut_multi(12,r_mpc_cut_emm,val_metal_emm)
            r_mpc_cut_emm,val_temp_emm=inner_cut_multi(6.e-4,r_mpc,val_temp_emm)
            r_mpc_cut2_emm,val_temp_emm=outer_cut_multi(12,r_mpc_cut_emm,val_temp_emm)
    
            mean_unnorm_dens=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_dens)
            mean_unnorm_pres=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_pres)
            mean_unnorm_metal_uw=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_metal_uw)
            mean_unnorm_metal_gmw=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_metal_gmw)
            mean_unnorm_temp_uw=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_temp_uw)
            mean_unnorm_temp_gmw=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_temp_gmw)
            mean_unnorm_metal_emm=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_metal_emm)
            mean_unnorm_temp_emm=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_temp_emm)
            
            header='R (Mpc), rho (g/cm^3), pth (g/cm/s^2) \n feedback param %s, nprofs %i'%(str(ASN1[j]),nprofs)
            #np.savetxt(home+'Profiles_for_mopc/IllustrisTNG_'+simulations[j]+'_'+snap[0]+'.txt',np.c_[r_mpc_cut2,mean_unnorm_dens,mean_unnorm_pres],header=header)
            
            ax1.loglog(r_mpc_cut2,mean_unnorm_dens,color=colors[j],label='%s nhalo:%i'%(simulations[j],nprofs))
            ax2.loglog(r_mpc_cut2,mean_unnorm_pres,color=colors[j],label='%s'%str(ASN1[j]))
            ax3.loglog(r_mpc_cut2,mean_unnorm_metal_uw,color=colors[j])#,label='uw')
            ax3.loglog(r_mpc_cut2,mean_unnorm_metal_gmw,color=colors[j],linestyle='dashed')#,label='gmw')
            ax3.loglog(r_mpc_cut2_emm,mean_unnorm_metal_emm,color=colors[j],linestyle='dotted')
            ax4.loglog(r_mpc_cut2,mean_unnorm_temp_uw,color=colors[j])#,label='uw')
            ax4.loglog(r_mpc_cut2,mean_unnorm_temp_gmw,color=colors[j],linestyle='dashed')#,label='gmw')
            ax4.loglog(r_mpc_cut2_emm,mean_unnorm_temp_emm,color=colors[j],linestyle='dotted')
            ax1.set_title(r"Mass $%s \leq M_\odot \leq %s$"%(mh_low_pow[i],mh_high_pow[i]),size=14)
            #ax2.set_title(r"Vary ASN1: Galactic Winds (energy/SFR)")
            #ax2.set_title(r"Vary AAGN1: Kinetic Mode BH feedback")
            

plt.suptitle(r'$\Omega_m = 0.3, \sigma_8 = 0.8$')
ax1.set_ylabel(r"$\rho_{gas}(g/cm^3)$",size=14)
ax2.set_ylabel(r"$P_{th} (g/cm/s^2)$",size=14)
ax3.set_ylabel(r"Metal fraction",size=14)
ax4.set_ylabel(r"Temperature (K)",size=14)
ax3.set_xlabel(r'R (Mpc)',size=12)
ax4.set_xlabel(r'R (Mpc)',size=12)

ax1.legend()
#ax2.legend()
#ax3.legend()
#ax4.legend()

plt.subplots_adjust(wspace=0.2,hspace=0.1)
plt.savefig(home+'Figures/emm.png')

    