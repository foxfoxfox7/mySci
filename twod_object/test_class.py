import numpy as np
import multiprocessing as mp

import quantarhei as qr
import aceto
from aceto.lab_settings import lab_settings

import agg_instance
import myFuncs


#######################################################################
# parameters
#######################################################################

iterations = 1

padding = 1000
t13 = qr.TimeAxis(0, 300, 1)
t2 = qr.TimeAxis(0, 2, 50)
num_mol = 3
spec_names = ['para', 'perp']

a_0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
a_90 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
lab_para = lab_settings(lab_settings.FOUR_WAVE_MIXING)
lab_para.set_laser_polarizations(a_0,a_0,a_0,a_0)
lab_perp = lab_settings(lab_settings.FOUR_WAVE_MIXING)
lab_perp.set_laser_polarizations(a_0,a_0,a_90,a_90)
labs = [lab_para, lab_perp]

#######################################################################
# setup
#######################################################################

final = agg_instance.AggregateAverage()
#final.twod_setup(ax2 = t2, ax13 = t13, pad = padding)
test = agg_instance.AggregateInstance()
test.get_molecules_circular(nM = num_mol, verbose = False)
#test.get_molecules_pdb('3eoj')
test.assign_spec_dens()

for i in range(iterations):
    test.assign_energies(method = 'different')
    test.build_agg(mult = 2)
    #test.test_dynamics('different')
    test.twod_setup(ax2 = t2, ax13 = t13)
    for i, lab in enumerate(labs):
        print(test.agg.get_Hamiltonian())
        test.twod_calculate(lab = lab, pad = padding, name = spec_names[i])
        final.twod_setup(ax2 = t2, ax13 = t13, pad = padding, name = spec_names[i])
        final.add_data(test.spec_cont_dict, name = spec_names[i])
'''

def calc_and_add():

    for i in range(iterations):
        test.assign_energies()
        test.build_agg(mult = 2)
        #eigen_dict = test.get_state_data()
        #test.test_dynamics('different')
        test.twod_setup(ax2 = t2, ax13 = t13)
        test.twod_calculate(lab = lab_para, pad = padding)
        final.add_data(test.spec_cont_dict)

    final.normalise(iterations)
    #final.save_as_numpy()
    final.plot_spec(disp_range = 1200)



n_per_loop = 2
n_cores = 1
n_loops = 1

#with mp.Pool(processes=n_cores) as pool:
#    pool.map(calc_and_add, [k for k in range(n_per_loop)])
'''


for name in spec_names:
    final.normalise(iterations, name = name)
    final.get_max_list(name = name)
    final.get_point_list(point = 12500, name = name)
    final.plot_spec(disp_range = 1200, name = name)
final.get_anisotropy('para', 'perp', _print_=True)

'''

spec_names = ['para', 'perp']
for i, lab in enumerate(labs):
    test.twod_calculate(lab = lab, pad = 1000, name = spec_names[i])
    test.print_spectra(name = spec_names[i])
    test.get_max_list(name = spec_names[i])
    with qr.energy_units("1/cm"):
        test.get_point_list(12500, name = spec_names[i])

test.get_anisotropy('para', 'perp')
test.print_data()
'''
