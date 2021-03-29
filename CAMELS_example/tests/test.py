import numpy as np
import h5py
import sys
sys.path.insert(0,'/home/jovyan')
sys.path.append('/home/jovyan/illustris_python')
import illustris_python as il
'''
basepath='/home/jovyan/Simulations/IllustrisTNG/1P_0/'
f=h5py.File(basepath+'snap_033.hdf5','r')

#h5py way
f=h5py.File(basepath+'snap_033.hdf5','r')
gas_mass,metallicity,metals=np.array(f['/PartType0/Masses'],dtype='float'),np.array(f['/PartType0/GFM_Metallicity'],dtype='float'),np.array(f['/PartType0/GFM_Metals'],dtype='float')
#gas_mass*=1.e10
gas_mass_tot=np.sum(gas_mass)
print("h5py:total gas mass from Masses",gas_mass_tot)
print("h5py:total metal mass from Metallicity",np.sum(metallicity))#*gas_mass_tot))
metal_C=np.sum(metals[:,2])#*gas_mass_tot)
metal_N=np.sum(metals[:,3])#*gas_mass_tot)
metal_O=np.sum(metals[:,4])#*gas_mass_tot)
metal_Ne=np.sum(metals[:,5])#*gas_mass_tot)
metal_Mg=np.sum(metals[:,6])#*gas_mass_tot)
metal_Si=np.sum(metals[:,7])#*gas_mass_tot)
metal_Fe=np.sum(metals[:,8])#*gas_mass_tot)
metal_other=np.sum(metals[:,9])#*gas_mass_tot)
metal_total=metal_C+metal_N+metal_O+metal_Ne+metal_Mg+metal_Si+metal_Fe+metal_other
print("h5py:total metal mass from GFM_Metals",metal_total)

#h5py way, without dtype=float
f=h5py.File(basepath+'snap_033.hdf5','r')
gas_mass,metallicity,metals=np.array(f['/PartType0/Masses']),np.array(f['/PartType0/GFM_Metallicity']),np.array(f['/PartType0/GFM_Metals'])
#gas_mass*=1.e10
gas_mass_tot=np.sum(gas_mass)
print("h5py,w/o dtype:total gas mass from Masses",gas_mass_tot)
print("h5py,w/o dtype:total metal mass from Metallicity",np.sum(metallicity))#*gas_mass_tot))
metal_C=np.sum(metals[:,2])#*gas_mass_tot)
metal_N=np.sum(metals[:,3])#*gas_mass_tot)
metal_O=np.sum(metals[:,4])#*gas_mass_tot)
metal_Ne=np.sum(metals[:,5])#*gas_mass_tot)
metal_Mg=np.sum(metals[:,6])#*gas_mass_tot)
metal_Si=np.sum(metals[:,7])#*gas_mass_tot)
metal_Fe=np.sum(metals[:,8])#*gas_mass_tot)
metal_other=np.sum(metals[:,9])#*gas_mass_tot)
metal_total=metal_C+metal_N+metal_O+metal_Ne+metal_Mg+metal_Si+metal_Fe+metal_other
print("h5py,w/o dtype:total metal mass from GFM_Metals",metal_total)

#illustris_python way
fields=['GFM_Metallicity','GFM_Metals','Masses']
particles=il.snapshot.loadSubset(basepath,33,'gas',fields)
gas_mass,metallicity,metals=np.array(particles['Masses'],dtype='float'),np.array(particles['GFM_Metallicity'],dtype='float'),np.array(particles['GFM_Metals'],dtype='float')
gas_mass_tot=np.sum(gas_mass)
print("illpy:total gas mass from Masses",gas_mass_tot)
print("illpy:total metal mass from Metallicity",np.sum(metallicity))#*gas_mass_tot))
metal_C=np.sum(metals[:,2])#*gas_mass_tot)
metal_N=np.sum(metals[:,3])#*gas_mass_tot)
metal_O=np.sum(metals[:,4])#*gas_mass_tot)
metal_Ne=np.sum(metals[:,5])#*gas_mass_tot)
metal_Mg=np.sum(metals[:,6])#*gas_mass_tot)
metal_Si=np.sum(metals[:,7])#*gas_mass_tot)
metal_Fe=np.sum(metals[:,8])#*gas_mass_tot)
metal_other=np.sum(metals[:,9])#*gas_mass_tot)
metal_total=metal_C+metal_N+metal_O+metal_Ne+metal_Mg+metal_Si+metal_Fe+metal_other
print("illpy:total metal mass from GFM_Metals",metal_total)

#illustris_python way,withouy dtype=float
fields=['GFM_Metallicity','GFM_Metals','Masses']
particles=il.snapshot.loadSubset(basepath,33,'gas',fields)
gas_mass,metallicity,metals=np.array(particles['Masses']),np.array(particles['GFM_Metallicity']),np.array(particles['GFM_Metals'])
gas_mass_tot=np.sum(gas_mass)
print("illpy,w/o dtype:total gas mass from Masses",gas_mass_tot)
print("illpy,w/o dtype:total metal mass from Metallicity",np.sum(metallicity))#*gas_mass_tot))
metal_C=np.sum(metals[:,2])#*gas_mass_tot)
metal_N=np.sum(metals[:,3])#*gas_mass_tot)
metal_O=np.sum(metals[:,4])#*gas_mass_tot)
metal_Ne=np.sum(metals[:,5])#*gas_mass_tot)
metal_Mg=np.sum(metals[:,6])#*gas_mass_tot)
metal_Si=np.sum(metals[:,7])#*gas_mass_tot)
metal_Fe=np.sum(metals[:,8])#*gas_mass_tot)
metal_other=np.sum(metals[:,9])#*gas_mass_tot)
metal_total=metal_C+metal_N+metal_O+metal_Ne+metal_Mg+metal_Si+metal_Fe+metal_other
print("illpy,w/o dtype:total metal mass from GFM_Metals",metal_total)

#there's some missing for the GFM_Metals way, I'm guessing in the untracked metals? Also I think weird numerical things happen when the mass is multiplied by 1e10. Everything matches for 2 decimal places and everything before the decimal, so maybe it's fine?
'''

'''
#temp example
E,Xe=f['/PartType0/InternalEnergy'],f['/PartType0/ElectronAbundance']
mp=1.67e-24
Xh=0.76
gamma=5./3.
kb=1.38e-16
mu=4.*mp/(1.+3.*Xh+4.*Xh*Xe[0])
T=mu*(gamma-1.)*E[0]/kb
print("Temp bad units",T)
print("Temp K",T*1.e10)
'''
'''
#density example
Mass=f['/PartType0/Masses']
print("Masses 1e10, array",np.array(Mass))
print("Masses 1e10, array and dtype",np.array(Mass,dtype='float'))
print("Masses, array",np.array(Mass)*1.e10)
print("Masses,array and dtype",np.array(Mass,dtype='float')*1.e10)
'''
'''
base='/home/jovyan/home/illstack/CAMELS_example/NPZ_files/'
illpy=np.load(base+'gasdens_tng_1P_22_033_mh_unscaled.npz')
noillpy=np.load(base+'gasdens_tng_1P_22_033_noillpy.npz')
mh_illpy=illpy['M_Crit200']
mh_noillpy=noillpy['M_Crit200']

print("Using illpy, without specifying array",mh_illpy)
print("Using h5py, with specifying array",mh_noillpy)
'''
'''
basepath='/home/jovyan/Simulations/SIMBA/1P_10/'
f=h5py.File(basepath+'snap_033.hdf5','r')
keys=f['/PartType0'].keys()
print("SIMBA keys",keys)
haloids=f['/PartType0/HaloID']
print("SIMBA halos",np.shape(haloids),haloids[2])
basepath='/home/jovyan/Simulations/IllustrisTNG/1P_10/'
f=h5py.File(basepath+'snap_033.hdf5','r')
keys=f['/PartType0'].keys()
print("TNG keys",keys)
'''
'''
base_tng='/home/jovyan/Simulations/IllustrisTNG/1P_22/'
metals_tng=il.snapshot.loadSubset(base_tng,33,'gas',fields=['GFM_Metallicity'])
print(np.shape(metals_tng))
base_smb='/home/jovyan/Simulations/SIMBA/1P_22/'
metals_smb=il.snapshot.loadSubset(base_smb,33,'gas',fields=['Metallicity'])
print(np.shape(metals_smb[:,0]))
'''
'''
base_tng='/home/jovyan/Simulations/IllustrisTNG/1P_22/'
halos=il.groupcat.loadHalos(base_tng,33,fields=['Group_M_Crit200','GroupMassType','Group_R_Crit200'])
mh=halos['Group_M_Crit200']
rh=halos['Group_R_Crit200']
masses=halos['GroupMassType']
gas=masses[:,0]
dm=masses[:,1]
mstar=masses[:,4]
bh=masses[:,5]
print("total number halos",len(mstar))
counter_mh=0
counter_mstar=0
counter_rh=0
counter_bh=0
counter_dm=0
counter_gas=0
mh_idx=[]
mstar_idx=[]
rh_idx=[]
bh_idx=[]
dm_idx=[]
gas_idx=[]
for m in np.arange(len(mstar)):
    if mstar[m]==0.:
        counter_mstar+=1
        mstar_idx.append(m)
    if mh[m]==0.:
        counter_mh+=1
        mh_idx.append(m)
    if rh[m]==0:
        counter_rh+=1
        rh_idx.append(m)
    if gas[m]==0:
        counter_gas+=1
        gas_idx.append(m)
    if bh[m]==0:
        counter_bh+=1
        bh_idx.append(m)
    if dm[m]==0:
        counter_dm+=1
        dm_idx.append(m)
print(counter_mstar,"number of zeros in mstar")
#print(mstar_idx)
print(counter_mh,"number of zeros in mh")
#print(mh_idx)
print(counter_rh,"number of zeros in rh")
#print(rh_idx)
print(counter_bh,"number of zeros in bh")
#print(bh_idx)
print(counter_gas,"number of zeros in gas")
#print(gas_idx)
print(counter_dm,"number of zeros in dm")
#print(dm_idx)
'''
stacks=np.load('/home/jovyan/home/illstack/CAMELS_example/Batch_NPZ_files/SIMBA/SIMBA_1P_22_2.00259.npz',allow_pickle=True)
n=stacks['n']
print(np.shape(n))
print(n)

