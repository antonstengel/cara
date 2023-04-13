DEFAULT_HYPERPARAMETERS = {
    'type_class_sf'      : 4, # num sig figs to use for p_nc
    'masses_sf'          : 4, # num sig figs for discretized masses
    'final_mass_sf'      : 7, # num sig figs to use for the mass with largest value in overall support
    'support_sf'         : 4, # num sig figs to use for each value in support
    'final_support_sf'   : 7, # num sig figs to use for the largest value in the overall support
    'round_down_mass'    : True, # boolean to decide whether final masses should be rounded down
    'round_down_support' : True, # boolean for whether each value in support should be rounded down
    'roasolver'          : 'cara/roasolver.jar', # roasolver JAR to use
    'parallel'           : True # run discretized trials in parallel
}

# specifies the 
H_INTS = ('type_class_sf', 'masses_sf', 'final_mass_sf', 'support_sf', 'final_support_sf')
H_BOOLS = ('round_down_mass', 'round_down_support', 'parallel')
H_STRS = ('roasolver')