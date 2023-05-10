#############################################################################################################################################
#########################################----LALA Likelihood of Accordant Lineations Analysis -----##########################################
#############################################################################################################################################

#### Toy Example using LALA
#### (C) Rosie Archer, University of Sheffield.
#### For details see Archer et al., in prep, ESPL.

### License can be found in the LALA Tool repository

import numpy as np
import pandas as pd
import netCDF4 as nc
import LALA_functions as LALA

### Set-up using model simulation from BRITICE CHRONO
uvel = nc.Dataset('ensemble_5km_443_ex.nc').variables['uvel'][:,:,:,0].data
vvel = nc.Dataset('ensemble_5km_443_ex.nc').variables['vvel'][:,:,:,0].data
velocity = nc.Dataset('ensemble_5km_443_ex.nc').variables['velsurf_mag'][:,:,:].data
thickness = nc.Dataset('ensemble_5km_443_ex.nc').variables['thk'][:,:,:].data
mask = nc.Dataset('ensemble_5km_443_ex.nc').variables['mask'][:,:,:].data

times = thickness.shape[0]

### Calculates the modelled angles
direction = LALA.calc_angles(uvel, vvel)

### Imports flowset information
flowsets = pd.read_excel('flowset_info.xlsx')

### Define location and directions of lineations
lineation_location_x = flowsets['x']
lineation_location_y = flowsets['y']
lineation_direction = flowsets['angles']

number_lineations = len(lineation_location_x)

### Define conditions ice must meet to be considered plausible for formation
thickness_condition = 10
velocity_condition = 10
grounded_ice = 2

### Define kappa and p
kappa = 10
p = 0.01

### Calculates the area over time where formation is considered possible
formation_possible_study, A = LALA.possible_formation(thickness_condition, velocity_condition, grounded_ice, thickness, velocity, mask)

### Calculates the pre-study area where formation is considered possible
### In this code, the whole domain is considered plausible, but in Archer et al., (in prep), a different value is calculated, omitted here for simplicity
formation_possible_pre = velocity.shape[1] * velocity.shape[2]

### Calculates lambda and lambda star
lam, lam_star = LALA.calculate_lambdas(p, formation_possible_pre, A)

### Calculates the likelihood of the lineations forming at the given locations
likelihood_location = lam * A + lam_star * formation_possible_pre

### Calculates integrated time score for each lineation location
nu = [0] * number_lineations

for flowset in range(number_lineations):
    
    ### Finds the time steps where the flowset should be scored
    times = np.where(formation_possible_study[:,lineation_location_x[flowset],lineation_location_y[flowset]])[0]    
    
    nu[flowset] = sum([LALA.nu_t(t, lineation_location_x[flowset], lineation_location_y[flowset], lam, direction, lineation_direction, flowset, kappa) for t in times]) + lam_star / (2*np.pi)

### Calculates the final likelihood
final_likelihood = np.sum(np.log(nu)) - likelihood_location

flowset_scores = pd.DataFrame({'Flowset' : np.arange(number_lineations), 'Score' : nu})

flowset_scores.to_excel('flowset_scores.xlsx', index=False)

print(final_likelihood)
