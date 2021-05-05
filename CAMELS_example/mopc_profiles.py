import matplotlib.pyplot as plt
import numpy             as np
import profile_functions

home='/home/jovyan/home/illstack/CAMELS_example/'
 

#-----------------------------------input section
#what suite?
suite='SIMBA'
  
nums=np.linspace(22,65,44,dtype='int') #0,65,66 for all
simulations=[]
for n in nums:
    simulations.append('1P_'+str(n))

#what redshifts? either enter snapshot as string, or z as array of floats
snap=['033']
#what masses?
mh_low=10**12.0 #CMASS 12.12-13.98
mh_high=10**12.5
mh_cut=True
#--------------------------------------------------------------- 

def extract(simulation,snap): #extract the quantities,adjust as necessary
    z=profile_functions.red_dict[snap] 
    stacks=np.load(home+'Batch_NPZ_files/'+suite+'/'+suite+'_'+simulation+'_'+str(z)+'.npz',allow_pickle=True)
    val            = stacks['val']
    val_dens       = val[0,:]
    val_pres       = val[1,:]
    val_metal_gmw  = val[2,:]
    val_temp_gmw   = val[3,:]
    bins           = stacks['nbins']
    r              = stacks['r']
    nprofs         = stacks['nprofs']
    mh             = stacks['M_Crit200'] #units 1e10 Msol/h, M200c
    rh             = stacks['R_Crit200'] #R200c
    GroupFirstSub  = stacks['GroupFirstSub']
    sfr            = stacks['sfr'] #Msol/yr
    mstar          = stacks['mstar'] #1e10 Msol/h
    return z,val_dens,bins,r,val_pres,nprofs,mh,rh,GroupFirstSub,sfr,mstar,val_metal_gmw,val_temp_gmw

mean_masses_uw={}
mean_masses_w={}
median_masses={}
for j in np.arange(len(simulations)):
    for k in np.arange(len(snap)):
        z,val_dens,bins,r,val_pres,nprofs,mh,rh,GroupFirstSub,sfr,mstar,val_metal_gmw,val_temp_gmw=extract(simulations[j],snap[k])
        omegab=0.049
        h=0.6711
        omegam,sigma8=np.loadtxt('/home/jovyan/Simulations/'+suite+'/'+simulations[j]+'/CosmoAstro_params.txt',usecols=(1,2),unpack=True)
        omegalam=1.0-omegam
        rhocrit=2.775e2
        rhocrit_z=rhocrit*(omegam*(1+z)**3+omegalam)
            
        mh,mstar,rh,val_dens,val_pres,r,val_temp_gmw=profile_functions.correct(z,h,mh,mstar,rh,val_dens,val_pres,r,val_temp_gmw)
    
        if mh_cut==True:
                mstar,mh,rh,sfr,GroupFirstSub,val_pres,val_dens,nprofs,val_metal_gmw,val_temp_gmw=profile_functions.mhalo_cut(mh_low,mh_high,mstar,mh,rh,sfr,GroupFirstSub,val_pres,val_dens,val_metal_gmw,val_temp_gmw,bins)
    
        r_mpc=r/1.e3
            
        #outer cut 20, inner cut 5e-4. For mopc we need (-3,1)
        print("sim",simulations[j])
        r_mpc_cut,val_dens=profile_functions.outer_cut_multi(20,r_mpc,val_dens)
        r_mpc_cut2,val_dens=profile_functions.inner_cut_multi(5.e-4,r_mpc_cut,val_dens)           
        r_mpc_cut,val_pres=profile_functions.outer_cut_multi(20,r_mpc,val_pres)
        r_mpc_cut2,val_pres=profile_functions.inner_cut_multi(5.e-4,r_mpc_cut,val_pres)
                        
        mean_unnorm_dens=np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]),0,val_dens)
        mean_unnorm_pres=np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]),0,val_pres)
        
        median_unnorm_dens=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_dens)
        median_unnorm_pres=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_pres)
        
        errup_dens_unnorm,errlow_dens_unnorm,std_dens_unnorm=profile_functions.get_errors(val_dens)
        errup_pres_unnorm,errlow_pres_unnorm,std_pres_unnorm=profile_functions.get_errors(val_pres)
        mean_masses_uw[simulations[j]]=np.mean(mh)
        median_masses[simulations[j]]=np.median(mh)
        
        
        header='R (Mpc), mean rho (Msol/kpc^3), errup (Msol/kpc^3), errlow, std, mean pth (Msol/kpc/s^2), errup(Msol/kpc/s^2), errlow, std, median rho (Msol/kpc^3), median pth (Msol/kpc/s^2) \n nprofs %i, mean mh %f, median mh %f \n Mass range 12-12.5'%(nprofs,np.mean(mh),np.median(mh))
        np.savetxt(home+'Mopc_profiles/'+suite+'/'+suite+'_'+simulations[j]+'_'+snap[k]+'_uw_12-12.5.txt',np.c_[r_mpc_cut2,mean_unnorm_dens, errup_dens_unnorm,errlow_dens_unnorm,std_dens_unnorm,mean_unnorm_pres,errup_pres_unnorm,errlow_pres_unnorm,std_pres_unnorm,median_unnorm_dens,median_unnorm_pres],header=header)
            
        #weighting
        mean_mh,mean_unnorm_dens_w,mean_unnorm_pres_w=profile_functions.mass_distribution_weight(mh,val_dens,val_pres)
        header='R (Mpc), rho (Msol/kpc^3), errup, errlow, std, pth (Msol/kpc/s^2), errup, errlow, std \n nprofs %i, average weighted mh %f'%(nprofs,mean_mh)
        mean_masses_w[simulations[j]]=mean_mh
        np.savetxt(home+'Mopc_profiles/'+suite+'/'+suite+'_'+simulations[j]+'_'+snap[k]+'_w.txt',np.c_[r_mpc_cut2,mean_unnorm_dens_w, errup_dens_unnorm,errlow_dens_unnorm,std_dens_unnorm,mean_unnorm_pres_w,errup_pres_unnorm,errlow_pres_unnorm,std_pres_unnorm],header=header)
        
#Adjust this as necessary        
f=open(home+'Mopc_profiles/'+suite+'/'+suite+'_z'+str(z)+'_halo_masses_12-12.5.txt','a')
f.write("median_masses= %s \n"%str(median_masses))
f.write("mean_masses_uw=%s \n"%str(mean_masses_uw))
f.write("mean_masses_w=%s \n"%str(mean_masses_w))
f.close()
