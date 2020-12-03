import sys
sys.path.insert(0,'/home/jovyan')
sys.path.append('/home/jovyan/illustris_python')
import illustris_python as il
import numpy as np
from . import params
import h5py

#commented out lines are the illustris-python way for TNG

def getparticles(snapshot_number,partType,field_list):

    basePath=params.basepath
    
    f=h5py.File(basePath+'snap_'+str(snapshot_number).zfill(3)+'.hdf5','r')
    
    particles={}
    for field in field_list:
        particles[field]=np.array(f['/PartType0/'+field]) #change PartType1 for dm
    #particles = il.snapshot.loadSubset(basePath,snapshot_number,partType,fields=field_list)
    return particles

def gethalos(snapshot_number,field_list):
    
    basePath=params.basepath
    #halos=il.groupcat.loadHalos(basePath,snapshot_number,fields=field_list)
    f=h5py.File(basePath+'fof_subhalo_tab_'+str(snapshot_number).zfill(3)+'.hdf5','r')
    halos={}
    for field in field_list:
        halos[field]=np.array(f['/Group/'+field])
    return halos

def getsubhalos(snapshot_number,field_list):
    basePath=params.basepath
    #subhalos=il.groupcat.loadSubhalos(basePath, snapshot_number,fields=field_list)
    f=h5py.File(basePath+'fof_subhalo_tab_'+str(snapshot_number).zfill(3)+'.hdf5')
    subhalos={}
    for field in field_list:
        subhalos[field]=np.array(f['/Subhalo/'+field])
    return subhalos


