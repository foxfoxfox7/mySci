import numpy as np

import quantarhei as qr
import aceto
from aceto.lab_settings import lab_settings

import agg_instance
import myFuncs

test = agg_instance.AggregateInstance()
test.get_molecules_circular(nM = 3)
#test.get_molecules_pdb('3eoj')
test.assign_spec_dens()

#test.test_dynamics('different')

test.assign_energies()
test.build_agg(mult = 2)

eigen_dict = test.get_state_data()

t13 = qr.TimeAxis(0, 300, 1)
t2 = qr.TimeAxis(0, 2, 50)

spec_names = ['para', 'perp']
a_0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
a_90 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
lab_para = lab_settings(lab_settings.FOUR_WAVE_MIXING)
lab_para.set_laser_polarizations(a_0,a_0,a_0,a_0)
lab_perp = lab_settings(lab_settings.FOUR_WAVE_MIXING)
lab_perp.set_laser_polarizations(a_0,a_0,a_90,a_90)
labs = [lab_para, lab_perp]

test.twod_setup(ax2 = t2, ax13 = t13)
for i, lab in enumerate(labs):
	test.twod_calculate(lab = lab, pad = 1000, name = spec_names[i])
	test.print_spectra(name = spec_names[i])
	test.get_max_list(name = spec_names[i])
	with qr.energy_units("1/cm"):
		test.get_point_list(12500, name = spec_names[i])

test.print_data()

test.get_anisotropy('para', 'perp')

#print(test.anisotropy)

#anis = (test.point_list['para'] - test.point_list['perp']) / (test.point_list['para'] + (2 * test.point_list['perp']))
#print(anis)

#test.twod_calculate(lab = lab_para)

#test.print_spectra()
#save_dir = 'data/'
#test.save_spectra(save_dir)


#print(test.point_list)