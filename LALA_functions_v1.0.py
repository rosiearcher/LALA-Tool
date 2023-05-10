#############################################################################################################################################
#########################################----LALA Likelihood of Accordant Lineations Analysis -----##########################################
#############################################################################################################################################

#### Functions to aid in the use of LALA
#### (C) Rosie Archer, University of Sheffield.
#### For details see Archer et al., in prep, ESPL.

### License can be found in the LALA Tool repository

#### When using these functions, ensure this file is in the same location as the script you are working in
#### Include the following line in your script to link the files
#import LALA_functions as LALA

import numpy as np
from scipy.stats import vonmises as vm

def degrees_to_radians(angles, in_radians):
    
    ''' Converts angles currently in degrees to radians
    which is needed to use the von Mises distribution correctly '''
    
    if not in_radians:
        
        angles = np.deg2rad(angles)
    
    return(angles)

def calc_angles(uvel, vvel):
    
    ''' Calculate the modelled angles '''

    dir_angle = np.arctan2(vvel, uvel)
    
    return(dir_angle)

def possible_formation(thickness_condition, velocity_condition, grounded_ice, thickness, velocity, mask):
    
    ''' Finds the locations that meet the lineation formation criteria and 
    the total area of '''
    
    ### Find the locations where the thickness and velocity conditions are met    
    thickness_binary = thickness > thickness_condition
    velocity_binary = velocity > velocity_condition
    mask_binary = mask == grounded_ice

    ### Finds the locations and times that it is possible to form lineations
    formation_possible_study = thickness_binary * velocity_binary * mask_binary
    
    A = np.sum(formation_possible_study)
    
    return(formation_possible_study, A)

def calculate_lambdas(p, prestudy_A, plausible_area_A):
    
    ### Calculate lambda and lambda_star
    lam_star = (p * 1)/prestudy_A
    lam = (1 - prestudy_A * lam_star)/plausible_area_A
    
    return(lam, lam_star)

def likelihood_location(lam, A, lam_star, formation_possible_pre):
    
    ''' Calculates the likelihood of the lineations forming at the given locations'''

    return(lam * A + lam_star * formation_possible_pre)


def nu_t(t, lineation_location_x, lineation_location_y, lam, direction, lineation_direction, flow, kappa):
    
    ''' Function to calculate the directional likelihood
    at each location and time'''
    
    return(lam * (0.5 * vm.pdf(direction[t,lineation_location_x,lineation_location_y] - lineation_direction[flow], kappa) 
                  + 0.5 * vm.pdf(direction[t,lineation_location_x,lineation_location_y] - lineation_direction[flow] + np.pi, kappa)))


