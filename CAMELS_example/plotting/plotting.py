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

#nums=np.linspace(0,65,66,dtype='int')
nums=np.linspace(22,28,7,dtype='int')
simulations=[]
for n in nums:
    simulations.append('1P_'+str(n))

ASN1=np.array([0.25,0.33,0.44,0.57,0.76,1.0,1.3])
red_dict_tng={'000':6.0,'001':5.0,'002':4.0,'003':3.5,'004':3.0,'005':2.81329,'006':2.63529,'007':2.46560,'008':2.30383,'009':2.14961,'010':2.00259,'011':1.86243,'012':1.72882,'013':1.60144,'014':1.48001,'015':1.36424,'016':1.25388,'017':1.14868,'018':1.04838,'019':0.95276,'020':0.86161,'021':0.77471,'022':0.69187,'023':0.61290,'024':0.53761,'025':0.46584,'026':0.39741,'027':0.33218,'028':0.27,'029':0.21072,'030':0.15420,'031':0.10033,'032':0.04896,'033':0.0}  

#simulations=['LH_10']
#snap=red_dict_tng.keys()  
snap=['033']
cut_color=0.6 
G=6.67e-11*1.989e30/((3.086e19)**3) #G in units kpc^3/(Msol*s^2)

ptype='pres'
color_cut=False
mh_cut=True

fig,axes=plt.subplots(2,4,sharey='row',figsize=(20,10)) #20,5
ax1=axes[0,0]
ax2=axes[0,1]
ax3=axes[0,2]
ax4=axes[0,3]
ax5=axes[1,0]
ax6=axes[1,1]
ax7=axes[1,2]
ax8=axes[1,3]
ax_arr=[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]

mh_low_arr=[10**10.0,10**11.0,10**12.0,10**13.0]
mh_high_arr=[10**11.0,10**12.0,10**13.0,10**14.0]
mh_low_pow=['10','11','12','13']
mh_high_pow=['11','12','13','14']

#------------------------------------------------------------
def mhalo_cut(mh_low,mh_high,mstar,mh,rh,sfr,GroupFirstSub,val_pres,val_dens,bins):
    idx=np.where((mh > mh_low) & (mh < mh_high))
    mstar,mh,rh,sfr,GroupFirstSub=mstar[idx],mh[idx],rh[idx],sfr[idx],GroupFirstSub[idx]
    nprofs=len(mh)
    val_pres,val_dens=val_pres[idx,:],val_dens[idx,:]
    val_pres,val_dens=np.reshape(val_pres,(nprofs,bins)),np.reshape(val_dens,(nprofs,bins))
    return mstar,mh,rh,sfr,GroupFirstSub,val_pres,val_dens,nprofs
    
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
    stacks_dens=np.load(home+'NPZ_files/gasdens_tng_'+simulation+'_'+snap+'_mh_unscaled.npz')
    stacks_pres=np.load(home+'NPZ_files/gaspth_tng_'+simulation+'_'+snap+'_mh_unscaled.npz')
    z              = red_dict_tng[snap] 
    val_dens       = stacks_dens['val']
    bins           = stacks_dens['nbins']
    r              = stacks_dens['r']
    val_pres       = stacks_pres['val']
    nprofs         = stacks_dens['nprofs']
    mh             = stacks_dens['M_Crit200'] #units 1e10 Msol/h, M200c
    rh             = stacks_dens['R_Crit200'] #R200c
    GroupFirstSub  = stacks_dens['GroupFirstSub']
    sfr            = stacks_dens['sfr'] #Msol/yr
    mstar          = stacks_dens['mstar'] #1e10 Msol/h
    return z,val_dens,bins,r,val_pres,nprofs,mh,rh,GroupFirstSub,sfr,mstar

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
    axb=ax_arr[i+4]
    
    fidu_dens=[]
    fidu_pres=[]
    for j in np.arange(len(simulations)):
        for k in snap:
    
            #basepath='/home/jovyan/Simulations/IllustrisTNG/'+simulations[j]+'/'
            z,val_dens,bins,r,val_pres,nprofs,mh,rh,GroupFirstSub,sfr,mstar=extract(simulations[j],k)
            omegab=0.049
            h=0.6711
            omegam,sigma8=np.loadtxt('/home/jovyan/Simulations/IllustrisTNG/'+simulations[j]+'/CosmoAstro_params.txt',usecols=(1,2),unpack=True)
            omegalam=1.0-omegam
            rhocrit=2.775e2
            rhocrit_z=rhocrit*(omegam*(1+z)**3+omegalam)
            
            mh,mstar,rh,val_dens,val_pres,r=correct(z,h,mh,mstar,rh,val_dens,val_pres,r)
    
            if mh_cut==True:
                    mstar,mh,rh,sfr,GroupFirstSub,val_pres,val_dens,nprofs=mhalo_cut(mh_low,mh_high,mstar,mh,rh,sfr,GroupFirstSub,val_pres,val_dens,bins)
    
            r_mpc=r/1.e3
            
            mean_xvals,norm_pres=normalize_pressure(nprofs,rh,r,mh,rhocrit_z,omegab,omegam,val_pres)
        

    
            #r_mpc_cut,val_dens=outer_cut_multi(5,r_mpc,val_dens)
            #r_mpc_cut2,val_dens=inner_cut_multi(1.e-2,r_mpc_cut,val_dens)
            r_mpc_cut,val_pres=outer_cut_multi(5,r_mpc,val_pres)
            r_mpc_cut2,val_pres=inner_cut_multi(1.e-2,r_mpc_cut,val_pres)
            x_cut,norm_pres=outer_cut_multi(5,mean_xvals,norm_pres)
            x_cut2,norm_pres=inner_cut_multi(1.e-2,x_cut,norm_pres)
    
            #mean_unnorm_dens=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_dens)
            mean_unnorm_pres=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_pres)
            mean_norm_pres=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,norm_pres)
    
    
            #errup_dens_unnorm,errlow_dens_unnorm,std_dens_unnorm=get_errors(val_dens)
            errup_pres_unnorm,errlow_pres_unnorm,std_pres_unnorm=get_errors(val_pres)
        
            if simulations[j]=='1P_22':
                #fidu_dens.append(mean_unnorm_dens)
                #fidu_pres.append(mean_unnorm_pres)
                #fidu_dens=mean_unnorm_dens
                #fidu_pres=mean_unnorm_pres
                axt.fill_between(r_mpc_cut2,mean_unnorm_pres+errup_pres_unnorm, mean_unnorm_pres-errlow_pres_unnorm,facecolor='grey',alpha=0.4)
                #fidu_pres=mean_norm_pres
                errup_pres_norm,errlow_pres_norm,std_pres_norm=get_errors(norm_pres)
                axb.fill_between(x_cut2,mean_norm_pres+errup_pres_norm, mean_norm_pres-errlow_pres_norm,facecolor='grey',alpha=0.4)
            else:
                fidu_dens=fidu_dens
                fidu_pres=fidu_pres
        
       
            if ptype =='pres':
                axt.loglog(r_mpc_cut2,mean_unnorm_pres,label='%s,nhalo:%i'%(simulations[j],nprofs))
                axb.loglog(x_cut2,mean_norm_pres,label='%s'%str(ASN1[j]))
                axt.set_title(r"Mass $%s \leq M_\odot \leq %s$"%(mh_low_pow[i],mh_high_pow[i]),size=14)
                #ax2.semilogx(r_mpc_cut2,mean_unnorm_pres/fidu_pres,label='%s'%(str(ASN1[j])))
                #ax2.set_ylabel(r"$P_{th}/P_{th}^{*}$",size=14)
            else:
                ax.loglog(r_mpc_cut2,mean_unnorm_dens,label='Sim:%s'%(simulations[j]))
                ax.set_ylabel(r"$\rho_{gas}(x) [g/cm^3]$",size=14)
                #ax2.semilogx(r_mpc_cut2,mean_unnorm_dens/fidu_dens,label='%s'%(str(ASN1[j])))
                #ax2.set_ylabel(r"$\rho_{gas}/\rho_{gas}^{*}$",size=14)

plt.suptitle(r'$\Omega_m = 0.3, \sigma_8 = 0.8$')
ax1.set_ylabel(r"$P_{th} (g/cm/s^2)$",size=14)
ax1.set_xlabel(r'R (Mpc)',size=12)
ax2.set_xlabel(r'R (Mpc)',size=12)
ax3.set_xlabel(r'R (Mpc)',size=12)
ax4.set_xlabel(r'R (Mpc)',size=12)

ax5.set_ylabel(r"$P_{th}/P_{200c}$",size=14)
ax5.set_xlabel(r'$r/r_{200c}$',size=14)
ax6.set_xlabel(r'$r/r_{200c}$',size=14)
ax7.set_xlabel(r'$r/r_{200c}$',size=14)
ax8.set_xlabel(r'$r/r_{200c}$',size=14)


#ax2.set_title(r'Varying $A_{SN1}$')
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax5.legend()
#ax2.legend()

plt.subplots_adjust(wspace=0)
plt.savefig(home+'Figures/'+ptype+'_033_diff_mass_comb.png')
#plt.show()
    
    
