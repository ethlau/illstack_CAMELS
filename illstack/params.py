basepath = '/home/jovyan/Simulations/IllustrisTNG/1P_22/'
serial      = True
search_radius = 20 #20 for full, 5 for testing
lims        = [10e-2,10e4]  #[0.01,10] scaled
bins        = 25
mass_low    = 10**10.0  #actually 10^10/h
mass_high   = 10**15.0  #actually 10^15/h 
scaled_radius = False #True == scaled, False == unscaled
mass_kind   = 'halo' #options='stellar','halo' 
save_direct = '/home/jovyan/home/illstack/CAMELS_example/NPZ_files/' 
