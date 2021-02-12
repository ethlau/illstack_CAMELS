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
prof6 = str(sys.argv[6])
prof7 = str(sys.argv[7])
prof8 = str(sys.argv[8])
prof9 = str(sys.argv[9])
snap_num= int(sys.argv[10])
sim=str(sys.argv[11])
simulation=str(sys.argv[12])

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
#prof=[prof1,prof2,prof3]
prof=[prof1,prof2,prof3,prof4,prof5,prof6,prof7,prof8,prof9]

volweight=[]
vals=[]
weights=[]
for p in prof:
    if p=='gasdens':
        print("Computing values for gasdens")
        part_type='gas'
        field_list = ['Coordinates','Masses']
        gas_particles = istk.io.getparticles(snap_num,part_type,field_list)
        posp = gas_particles['Coordinates'] #position, base unit ckpc/h 
        val=gas_particles['Masses']
        vals.append(val)   #units 1e10 Msol/h
        volweight.append(True)
        weights.append(1.0+0*val)
    elif p=='gaspth':
        print("Computing values for gaspth")
        part_type='gas'
        field_list = ['Coordinates','Masses','InternalEnergy']
        gas_particles = istk.io.getparticles(snap_num,part_type,field_list)
        posp = gas_particles['Coordinates'] #base unit ckpc/h 
        val=gas_particles['Masses']*gas_particles['InternalEnergy']*(gamma-1.)#unit 1e10Msol/h*(km/s)**2
        vals.append(val)
        volweight.append(True)
        weights.append(1.0+0*val)
    elif p=='metals_uw':
        print("Computing values for unweighted metallicity (metals_uw)")
        part_type='gas'
        
        if sim=='IllustrisTNG':
            field_list=['GFM_Metallicity','Masses','Coordinates']
            gas_particles=istk.io.getparticles(snap_num,part_type,field_list)
            posp=gas_particles['Coordinates']
            val=gas_particles['GFM_Metallicity'] #ratio
        elif sim=='SIMBA': 
            field_list=['Metallicity','Masses','Coordinates']
            gas_particles=istk.io.getparticles(snap_num,part_type,field_list)
            posp=gas_particles['Coordinates']
            val=gas_particles['Metallicity']  
            val=val[:,0] #total metals fraction, the rest are in order: He,C,N,O,Ne,Mg,Si,S,Ca,Fe
        vals.append(val)
        volweight.append(False)
        weights.append(1.0+0*val)
    elif p=='metals_gmw':
        print("Computing values for gas-mass-weighted metallicity (metals_gmw)")
        part_type='gas'
        if sim=='IllustrisTNG':
            field_list=['GFM_Metallicity','Masses','Coordinates']
            gas_particles=istk.io.getparticles(snap_num,part_type,field_list)
            posp=gas_particles['Coordinates']
            val=gas_particles['GFM_Metallicity'] #ratio
        elif sim=='SIMBA': 
            field_list=['Metallicity','Masses','Coordinates']
            gas_particles=istk.io.getparticles(snap_num,part_type,field_list)
            posp=gas_particles['Coordinates']
            val=gas_particles['Metallicity']  
            val=val[:,0]
        vals.append(val)
        volweight.append(False)
        weights.append(gas_particles['Masses'])
    elif p=='gasmass': 
        #this is binned mass to multiply metals profile to get M_z instead of ratio
        print("Computing values for unweighted gas mass")
        part_type='gas'
        field_list=['Coordinates','Masses']
        gas_particles=istk.io.getparticles(snap_num,part_type,field_list)
        posp=gas_particles['Coordinates']
        val=gas_particles['Masses']
        vals.append(val)
        volweight.append(False)
        weights.append(1.0+0*val)
    elif p=='gastemp_uw':
        print("Computing values for unweighted gas temperature (gastemp_uw)")
        part_type='gas'
        field_list=['Coordinates','Masses','InternalEnergy','ElectronAbundance']
        gas_particles=istk.io.getparticles(snap_num,part_type,field_list)
        posp=gas_particles['Coordinates']
        mu=(4.*mp/(1.+3.*Xh+4.*Xh*gas_particles['ElectronAbundance'])) #CGS
        val=gas_particles['InternalEnergy']*mu*(gamma-1.)/kb #K*(km/cm)^2, mult by 10^10 later
        vals.append(val)
        volweight.append(False)
        weights.append(1.0+0*val)
    elif p=='gastemp_gmw':
        print("Computing values for gas-mass-weighted gas temperature (gastemp_gmw)")
        part_type='gas'
        field_list=['Coordinates','Masses','InternalEnergy','ElectronAbundance']
        gas_particles=istk.io.getparticles(snap_num,part_type,field_list)
        posp=gas_particles['Coordinates']
        mu=(4.*mp/(1.+3.*Xh+4.*Xh*gas_particles['ElectronAbundance'])) #CGS
        val=gas_particles['InternalEnergy']*mu*(gamma-1.)/kb #K*(km/cm)^2, mult by 10^10 later
        vals.append(val)
        volweight.append(False)
        weights.append(gas_particles['Masses'])
    elif p=='metals_emm':
        print("Computing values for emission-measure-weighted metallicity (metals_emm)")
        part_type='gas'
        if sim=='IllustrisTNG':
            field_list=['GFM_Metallicity','Masses','Coordinates']
            gas_particles=istk.io.getparticles(snap_num,part_type,field_list)
            posp=gas_particles['Coordinates']
            val=gas_particles['GFM_Metallicity'] #ratio
        elif sim=='SIMBA': 
            field_list=['Metallicity','Masses','Coordinates']
            gas_particles=istk.io.getparticles(snap_num,part_type,field_list)
            posp=gas_particles['Coordinates']
            val=gas_particles['Metallicity']  
            val=val[:,0]
        vals.append(val)
        volweight.append(False)
        weights.append(1.0+0*val)
    elif p=='gastemp_emm':
        print("Computing values for emission measure-weighted gas temperature (gastemp_emm)")
        part_type='gas'
        field_list=['Coordinates','Masses','InternalEnergy','ElectronAbundance']
        gas_particles=istk.io.getparticles(snap_num,part_type,field_list)
        posp=gas_particles['Coordinates']
        mu=(4.*mp/(1.+3.*Xh+4.*Xh*gas_particles['ElectronAbundance'])) #CGS
        val=gas_particles['InternalEnergy']*mu*(gamma-1.)/kb #K*(km/cm)^2, mult by 10^10 later
        vals.append(val)
        volweight.append(False)
        weights.append(1.0+0*val)
    else:
        print("Please enter an appropriate option for the profile")
        print("gasdens,gaspth,metals_uw,metals_gmw,gasmass,gastemp_uw,gastemp_gmw,metals_emm,gastemp_emm")

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
