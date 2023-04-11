DEFAULT_HYPERPARAMETERS = {
    'type_class_sf'      : 4, # sf meaning "sig figs"
    'masses_sf'          : 4,
    'final_mass_sf'      : 7,
    'support_sf'         : 4,
    'final_support_sf'   : 7,
    'round_down_mass'    : True,
    'round_down_support' : True,
    'roasolver'          : 'modules/roasolver.jar'
}

H_INTS = ('type_class_sf', 'masses_sf', 'final_mass_sf', 'support_sf', 'final_support_sf')
H_BOOLS = ('round_down_mass', 'round_down_support')
H_STRS = ('roasolver')