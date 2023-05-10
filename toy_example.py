#############################################################################################################################################
#########################################----LALA Likelihood of Accordant Lineations Analysis -----##########################################
#############################################################################################################################################

#### Toy Example using LALA
#### (C) Rosie Archer, University of Sheffield.
#### For details see Archer et al., in prep, ESPL.

### License can be found in the LALA Tool repository

import numpy as np
from scipy.stats import vonmises as vm

num_flowsets = 1
flowset_direction = [45]
flowset_location = [2,3]

time_steps = 3

### Modelled flow directions at each time step at the location of the flowset
flow_direction = [15, 135, 45]
    
### Modelled flow directions put in an array (more similar to model output)
# Non-flowset locations are set to be zero for purposes of demonstration, but this is not needed with actual data
modelled_flow_t1 = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,15,0],[0,0,0,0,0],[0,0,0,0,0]])
modelled_flow_t2 = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,135,0],[0,0,0,0,0],[0,0,0,0,0]])
modelled_flow_t3 = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,45,0],[0,0,0,0,0],[0,0,0,0,0]])


### If in_radians equals True, angles are assumed to be in radians
### If in_radians equals False, angles are assumed to be in degrees and are converted to radians below
in_radians = False

### Converts LALA inputs into radians if not already
if not in_radians:
    
    flowset_direction = np.deg2rad(flowset_direction)
    flow_direction = np.deg2rad(flow_direction)
    
    modelled_flow_t1 = np.deg2rad(modelled_flow_t1)
    modelled_flow_t2 = np.deg2rad(modelled_flow_t2)
    modelled_flow_t3 = np.deg2rad(modelled_flow_t3)

### Creates a 3D array of modelled flow directions
# This is similar to potential model output
modelled_flow = np.stack((modelled_flow_t1,modelled_flow_t2,modelled_flow_t3))

plausible_times = [1,2] ### python indexing starts at 0, so this corresponds to time steps 2 and 3 shown in Figure 2 and 3

### Choose values for kappa and p - see Section 2.3
kappa = 5
p = 0.01

### Defines the number of grid cells deemed plausible for lineation formation in the pre-study time step, i.e. not deep ocean, and has adequate sediment
# In this example, we set this equal to the area of the domain, and such assuming that all grid cells have the potential to form lineations
prestudy_A = 25

### The time-integrated area of an ice sheet model that fits another metric
# 42 was arbitrarily chosen in this toy example
plausible_area_A = 42

### Representing Figure 2 in Archer et al., (in prep) as arrays for each time step
plausible_t1 = np.array([[0,0,0,0,0],[0,0,0,0,0],[1,0,0,0,0],[1,1,1,0,0],[1,1,1,1,0]])
plausible_t2 = np.array([[0,0,0,0,0],[0,1,0,0,0],[1,1,1,1,0],[1,1,1,1,1],[1,1,1,1,1]])
plausible_t3 = np.array([[0,0,0,0,0],[0,0,0,0,1],[0,0,0,1,1],[0,1,1,1,1],[1,1,1,1,1]])

### Summing the number of grid cells that have appropriate conditions for lineation formation
# This leads to A_M() = 35
formation_plausible_A = np.sum(plausible_t1 + plausible_t2 + plausible_t3)

### Calculate lambda and lambda_star
lam_star = (p * 1)/prestudy_A
lam = (1 - prestudy_A * lam_star)/plausible_area_A

### Now we can calculate the log-likelihood for the locations
loc_log_likelihood = formation_plausible_A * lam + prestudy_A * lam_star

### Also calculate the directional likelihood (not the log form at this point) for each plausible time step
directional_likelihood_t1 = 0.5 * (vm.pdf(flowset_direction - flow_direction[1], kappa) +
                                   vm.pdf(flowset_direction + np.deg2rad(180) - flow_direction[1], kappa))

directional_likelihood_t2 = 0.5 * (vm.pdf(flowset_direction - flow_direction[2], kappa) +
                                   vm.pdf(flowset_direction + np.deg2rad(180) - flow_direction[2], kappa))

### Calculate the value of nu - also can be used on its own to give an idea of which flowsets match the best
nu = lam * (directional_likelihood_t1 + directional_likelihood_t2) + lam_star/(2*np.pi)

### Calculates the log likelihood score for a model simulation
final_likelihood = np.log(nu) - loc_log_likelihood
print(final_likelihood)