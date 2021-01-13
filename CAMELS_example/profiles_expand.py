#!/usr/bin/env python
import sys
import numpy             as np
sys.path.insert(0,'/home/jovyan/home/illstack/')
import matplotlib.pyplot as plt
import illstack as istk
import mpi4py.rc
from decimal import Decimal
import istk_params as params

istk.init.initialize('istk_params.py')

prof1 = str(sys.argv[1])
prof2 = str(sys.argv[2])
prof3 = str(sys.argv[3])
prof4 = str(sys.argv[4])
prof5 = str(sys.argv[5])
snap_num= int(sys.argv[6])
sim=str(sys.argv[7])
simulation=str(sys.argv[8])

print("Simulation:",simulation)
print("Snapshot:",snap_num)
ntile = 3 # controls tiling -- optimal value not yet clear

mlow=params.mass_low
mhigh=params.mass_high
mhmin = mlow /1e10 # minimum mass in 1e10 Msun/h
mhmax = mhigh /1e10 # maximum mass in 1e10 Msun/h

scaled_radius=params.scaled_radius
mass_kind=params.mass_kind
save_direct=params.save_direct

Xh=0.76
mp=1.67e-24 #g
gamma=5./3.
kb=1.38e-16 #g*cm^2/s^2/K

#omega* only matter for dm profile
omegam=0.31
omegab=0.0486
omegadm =omegam-omegab

#prof=[prof1,prof2,prof3,prof4,prof5]
prof=[prof1,prof2,prof3]

volweight=[]
vals=[]
for p in prof:
    if p=='gasdens':
        print("Completing profiles for gasdens")
        part_type='gas'
        field_list = ['Coordinates','Masses']
        gas_particles = istk.io.getparticles(snap_num,part_type,field_list)
        posp = gas_particles['Coordinates'] #position, base unit ckpc/h 
        vals.append(gas_particles['Masses'])   #units 1e10 Msol/h
        volweight.append(True)
    elif p=='dmdens':
        print("Completing profiles for dmdens")
        part_type='dm'
        # HARD CODED BOX SIZE 2.5e4 kpc/h
        part_massf=2.775e2*omegadm*(2.5e4)**3/1e10 # particle mass in 1e10 Msun/h
        field_list = ['Coordinates'] #base unit ckpc/h
        posp = istk.io.getparticles(snap_num,part_type,field_list)
        vals = posp[:,0]*0 + part_massf / np.shape(posp)[0]
        print('setting dm particle mass to = ',vals[0]*1e10,'Msun/h')
    elif p=='gaspth':
        print("Completing profiles for gaspth")
        part_type='gas'
        field_list = ['Coordinates','Masses','InternalEnergy']
        gas_particles = istk.io.getparticles(snap_num,part_type,field_list)
        posp = gas_particles['Coordinates'] #base unit ckpc/h 
        vals.append(gas_particles['Masses']*gas_particles['InternalEnergy']*(gamma-1.))#unit 1e10Msol/h*(km/s)**2
        volweight.append(True)
    elif p=='metals_gmw': #gas mass-weighted metallicity
        print("Completing profiles for metallicity")
        part_type='gas'
        field_list=['GFM_Metallicity','Masses','Coordinates'] #Metallicity for SIMBA
        gas_particles=istk.io.getparticles(snap_num,part_type,field_list)
        posp=gas_particles['Coordinates']
        vals.append(gas_particles['GFM_Metallicity']*gas_particles['Masses']) #ratio of M_metals/M_tot,gas
        volweight.append(False)
    elif p=='gasmass': 
        #this is binned mass to multiply metals profile to get M_z instead of ratio
        print("Completing profiles for gas mass")
        part_type='gas'
        field_list=['Coordinates','Masses']
        gas_particles=istk.io.getparticles(snap_num,part_type,field_list)
        posp=gas_particles['Coordinates']
        vals.append(gas_particles['Masses'])
        volweight.append(False)
    elif p=='gastemp_gmw': #gmw = gas mass-weighted
        print("Completing profiles for gas temperature")
        part_type='gas'
        field_list=['Coordinates','Masses','InternalEnergy','ElectronAbundance']
        gas_particles=istk.io.getparticles(snap_num,part_type,field_list)
        posp=gas_particles['Coordinates']
        mu=(4.*mp/(1.+3.*Xh+4.*Xh*gas_particles['ElectronAbundance'])) #CGS
        vals.append(gas_particles['InternalEnergy']*mu*(gamma-1.)/kb) #K*(km/cm)^2, mult by 10^10 later
        volweight.append(False)
    else:
        print("Please enter an appropriate option for the profile")
        print("gasdens,dmdens,gaspth,metals_gmw,gasmass,gastemp_gmw")

field_list = ['GroupBHMass','GroupBHMdot','GroupFirstSub','GroupGasMetalFractions','GroupGasMetallicity','GroupLen','GroupMass','GroupMassType','GroupNsubs','GroupPos','GroupSFR','GroupStarMetalFractions','GroupStarMetallicity','GroupVel','GroupWindMass','Group_M_Crit200','Group_M_Crit500','Group_M_Mean200','Group_M_TopHat200','Group_R_Crit200','Group_R_Crit500','Group_R_Mean200','Group_R_TopHat200']
#units=[1e10 Msol/h,1e10 (Msol/h)/(0.978 Gyr/h),index,ratio of total mass of species/total gas mass,metallicity,count,1e10 Msol/h, 1e10 Msol/h, count, ckpc/h, Msol/yr, fraction, metallicity, (km/s)/a (get peculiar velocity by multiplying this by 1/a),1e10 Msol/h, 1e10 Msol/h, 1e10 Msol/h, 1e10 Msol/h, 1e10 Msol/h, ckpc/h, ckpc/h, ckpc/h, ckpc/h]
halos = istk.io.gethalos(snap_num,field_list)

GroupBHMass=halos['GroupBHMass']
GroupBHMdot=halos['GroupBHMdot']
GroupFirstSub=halos['GroupFirstSub']
gas_metal_fractions=halos['GroupGasMetalFractions']
Group_GasH=gas_metal_fractions[:,0]
Group_GasHe=gas_metal_fractions[:,1]
Group_GasC=gas_metal_fractions[:,2]
Group_GasN=gas_metal_fractions[:,3]
Group_GasO=gas_metal_fractions[:,4]
Group_GasNe=gas_metal_fractions[:,5]
Group_GasMg=gas_metal_fractions[:,6]
Group_GasSi=gas_metal_fractions[:,7]
Group_GasFe=gas_metal_fractions[:,8]
GroupGasMetallicity=halos['GroupGasMetallicity']
GroupLen=halos['GroupLen']
GroupMass=halos['GroupMass']
halo_mass= halos['GroupMassType']
mstar= halo_mass[:,4] 
GroupNsubs=halos['GroupNsubs']
posh=halos['GroupPos']
sfr  = halos['GroupSFR']
star_metal_fractions=halos['GroupStarMetalFractions']
Group_StarH=star_metal_fractions[:,0]
Group_StarHe=star_metal_fractions[:,1]
Group_StarC=star_metal_fractions[:,2]
Group_StarN=star_metal_fractions[:,3]
Group_StarO=star_metal_fractions[:,4]
Group_StarNe=star_metal_fractions[:,5]
Group_StarMg=star_metal_fractions[:,6]
Group_StarSi=star_metal_fractions[:,7]
Group_StarFe=star_metal_fractions[:,8]
GroupStarMetallicity=halos['GroupStarMetallicity']
vel=halos['GroupVel'] 
GroupVelx=vel[:,0]
GroupVely=vel[:,1]
GroupVelz=vel[:,2]
GroupWindMass=halos['GroupWindMass']
mh   = halos['Group_M_Crit200']
M_Crit500=halos['Group_M_Crit500']
M_Mean200=halos['Group_M_Mean200']
M_TopHat200=halos['Group_M_TopHat200']
rh   = halos['Group_R_Crit200']
R_Crit500=halos['Group_R_Crit500']
R_Mean200=halos['Group_R_Mean200']
R_TopHat200=halos['Group_R_TopHat200']

vals=np.array(vals) #here
volweight=np.array(volweight) #here

r, val, n, mh, rh, nprofs,GroupFirstSub,sfr,mstar,GroupBHMass,GroupBHMdot,Group_GasH,Group_GasHe,Group_GasC,Group_GasN,Group_GasO,Group_GasNe,Group_GasMg,Group_GasSi,Group_GasFe,GroupGasMetallicity,GroupLen,GroupMass,GroupNsubs,Group_StarH,Group_StarHe,Group_StarC,Group_StarN,Group_StarO,Group_StarNe,Group_StarMg,Group_StarSi,Group_StarFe,GroupStarMetallicity,GroupVelx,GroupVely,GroupVelz,GroupWindMass,M_Crit500,M_Mean200,M_TopHat200,R_Crit500,R_Mean200,R_TopHat200= istk.cyprof.stackonhalos(posp,vals,posh,mh,rh,GroupFirstSub,sfr,mstar,ntile,volweight,mhmin, mhmax,scaled_radius,mass_kind,GroupBHMass,GroupBHMdot,Group_GasH,Group_GasHe,Group_GasC,Group_GasN,Group_GasO,Group_GasNe,Group_GasMg,Group_GasSi,Group_GasFe,GroupGasMetallicity,GroupLen,GroupMass,GroupNsubs,Group_StarH,Group_StarHe,Group_StarC,Group_StarN,Group_StarO,Group_StarNe,Group_StarMg,Group_StarSi,Group_StarFe,GroupStarMetallicity,GroupVelx,GroupVely,GroupVelz,GroupWindMass,M_Crit500,M_Mean200,M_TopHat200,R_Crit500,R_Mean200,R_TopHat200)
r  =np.reshape(r,  (nprofs,istk.params.bins))
val=np.reshape(val,(len(volweight),nprofs,istk.params.bins)) #here
n  =np.reshape(n,  (nprofs,istk.params.bins))

#Change name of npz file here
np.savez(save_direct+'profs_'+sim+'_'+simulation+'_'+str(sys.argv[6])+'.npz',r=r[0],val=val,n=n,M_Crit200=mh,R_Crit200=rh,nprofs=nprofs,nbins=istk.params.bins,GroupFirstSub=GroupFirstSub,sfr=sfr,mstar=mstar,GroupBHMass=GroupBHMass,GroupBHMdot=GroupBHMdot,Group_GasH=Group_GasH,Group_GasHe=Group_GasHe,Group_GasC=Group_GasC,Group_GasN=Group_GasN,Group_GasO=Group_GasO,Group_GasNe=Group_GasNe,Group_GasMg=Group_GasMg,Group_GasSi=Group_GasSi,Group_GasFe=Group_GasFe,GroupGasMetallicity=GroupGasMetallicity,GroupLen=GroupLen,GroupMass=GroupMass,GroupNsubs=GroupNsubs,Group_StarH=Group_StarH,Group_StarHe=Group_StarHe,Group_StarC=Group_StarC,Group_StarN=Group_StarN,Group_StarO=Group_StarO,Group_StarNe=Group_StarNe,Group_StarMg=Group_StarMg,Group_StarSi=Group_StarSi,Group_StarFe=Group_StarFe,GroupStarMetallicity=GroupStarMetallicity,GroupVelx=GroupVelx,GroupVely=GroupVely,GroupVelz=GroupVelz,GroupWindMass=GroupWindMass,M_Crit500=M_Crit500,M_Mean200=M_Mean200,M_TopHat200=M_TopHat200,R_Crit500=R_Crit500,R_Mean200=R_Mean200,R_TopHat200=R_TopHat200)
