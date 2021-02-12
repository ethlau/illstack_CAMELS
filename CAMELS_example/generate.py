import numpy as np
import subprocess
import time

sim='IllustrisTNG' #IllustrisTNG
                          
prof1='gasdens'
prof2='gaspth'
prof3='metals_uw'
prof4='metals_gmw' #gas mass weighted
prof5='gasmass'
prof6='gastemp_uw'
prof7='gastemp_gmw'
prof8='metals_emm'
prof9='gastemp_emm'

red_dict={'000':6.0,'001':5.0,'002':4.0,'003':3.5,'004':3.0,'005':2.81329,'006':2.63529,'007':2.46560,'008':2.30383,'009':2.14961,'010':2.00259,'011':1.86243,'012':1.72882,'013':1.60144,'014':1.48001,'015':1.36424,'016':1.25388,'017':1.14868,'018':1.04838,'019':0.95276,'020':0.86161,'021':0.77471,'022':0.69187,'023':0.61290,'024':0.53761,'025':0.46584,'026':0.39741,'027':0.33218,'028':0.27,'029':0.21072,'030':0.15420,'031':0.10033,'032':0.04896,'033':0.0}


snap=['033']
#snap=red_dict.keys() #for all snaps

#adjust for which batch of simulations
#nums=np.linspace(0,65,66,dtype='int')
#simulations=[]
#for n in nums:
#    simulations.append('1P_'+str(n))
simulations=['1P_22']

for j in simulations:
    
    f=open('istk_params.py','r')
    lines=f.readlines()
    lines[0]="basepath = '/home/jovyan/Simulations/"+sim+"/"+j+"/'\n"
    f.close()
    
    f=open('istk_params.py','w')
    f.writelines(lines)
    f.close()
    
    #update the basepath for each sim/snap, clunky but works
    f=open('istk_params.py','r')
    lines=f.readlines()
    line0=lines[0]
    basepath=line0[12:-2]
    print("Basepath check:",basepath)
    

    for k in snap:
        g='getprof_temp_profiles.sh'
        print('#!/bin/bash',file=open(g,'w'))   
        print('python', 'profiles_expand.py',prof1,prof2,prof3,prof4,prof5,prof6,prof7,prof8,prof9,k,sim,j,file=open(g,'a'))
        start=time.time()
        subprocess.call(['./getprof_temp_profiles.sh'],shell=True)
        
end=time.time()
print("Time elapsed:",(end-start)/60.,"minutes")

