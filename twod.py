#!/usr/bin/env python

'''
Code for generating 2D spectra on a circular aggregate of chromophores
Requires the use of quantarhei for 2D simulation and Aceto for laser
conditions and band levels
'''

import os
import math
import time
import random
import shutil
import copy
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp

import quantarhei as qr
from quantarhei.models.spectdens import SpectralDensityDB
import aceto
from aceto.lab_settings import lab_settings

import myFuncs


_test_ = True
_save_ = False
_show_ = True

dir_name = 'test4'
eigen_name = 'eigen'
para_resp_name = 'paraResp'
perp_resp_name = 'perpResp'

num_mol = 3
dipole_strength = 5
static_dis = 300
energy = 12500
reorganisation = 102
width = 300.0
cor_time = 100.0
temperature = 300

# Hwo to define the energies, all same, all different, all random
energies0 = [energy] * num_mol
#energies0 = [energy - (100 * num_mol / 2) + i * 100 for i in range(num_mol)]
#energies0 = [random.gauss(energy, static_dis) for i in range(num_mol)]
print('energies - ', energies0)

t13_ax_ax_step = 1
t13_ax_ax_len = 300
t2_ax_step = 100
t2_ax_len = 100
padding = 1000

#######################################################################
# Setup
#######################################################################

# The axis for time gaps 1 and 3
#t13_ax_ax_count = int(t13_ax_ax_len/t13_ax_ax_step)+1
#t13_ax = qr.TimeAxis(0.0, t13_ax_ax_count, t13_ax_ax_step)

# The axis for time gap 2
#t2_ax_count = int(t2_ax_len/t2_ax_step)+1
#t2_ax = qr.TimeAxis(0.0, t2_ax_count, t2_ax_step)

t13_ax = qr.TimeAxis(0, 300, 1)
t2_ax = qr.TimeAxis(0, 2, 50)

# Making the names of the directories and files to save stuff
save_dir = 'data/' + dir_name + '/'
eigen_file = save_dir + eigen_name + '.txt'
para_resp_dir = save_dir + para_resp_name + '/'
perp_resp_dir = save_dir + perp_resp_name + '/'
if _save_:
    try:
    	shutil.rmtree(save_dir)
    	print('Old directory removed')
    	os.mkdir(save_dir)
    except:
    	os.mkdir(save_dir)
        #print("Creation of the directory %s failed" % save_dir)

t1 = time.time()
print("Calculating spectra from 0 to ", t2_ax_len, " fs\n")

#######################################################################
# Creation of the aggregate positions and dipoles
#######################################################################

'''
A function to create a list of quantarhei molecules
with cicular coordinates evenly spaceed and dipole
vectors running round the circumference all in the same direction
'''
for_agg = myFuncs.circularAgg(num_mol, dipole_strength)

'''
Takes the name of a pdb file (without the extension) and converts it
into a list of molecules with positions and dipoles. The only molecle
it looks for is Bacteriochlorophyll
'''
#for_agg = myFuncs.bacteriochl_agg('3eoj')

'''
Just a simple dimer for testing. change the positions and dipoles
'''
#moleculeOne = qr.Molecule()
#moleculeTwo = qr.Molecule()
#moleculeOne.position = [0.0, 0.0, 0.0]
#moleculeTwo.position = [0.0, 10.0, 0.0]
#moleculeOne.set_dipole(0,1,[10.0, 0.0, 0.0])
#moleculeTwo.set_dipole(0,1,[0.0, 10.0, 0.0])
#for_agg = [moleculeOne, moleculeTwo]

#######################################################################
# Spectral density
#######################################################################

# Time axis for the spectral desnity
# Needs to have small step for the dynamics to converge
t_ax_sd = qr.TimeAxis(0.0, 10000, 1)
db = SpectralDensityDB()

# Parameters for spectral density. ODBO, Renger or Silbey
params = {
	"ftype": "OverdampedBrownian",
    #"ftype": "B777",
    "alternative_form": True,
    "reorg": reorganisation,
    "T":temperature,
    "cortime":cor_time
    }
with qr.energy_units('1/cm'):
    sd_low_freq = qr.SpectralDensity(t_ax_sd, params)

# Adding the high freq modes
sd_high_freq = db.get_SpectralDensity(t_ax_sd, "Wendling_JPCB_104_2000_5825")
ax = sd_low_freq.axis
sd_high_freq.axis = ax
sd_tot = sd_low_freq + sd_high_freq

cf = sd_tot.get_CorrelationFunction(temperature=temperature, ta=t_ax_sd)
# Assigning the correlation function to the list of molecules
for mol in for_agg:
    mol.set_transition_environment((0,1),cf)

if _test_:
    reorg = qr.convert(sd_low_freq.get_reorganization_energy(), "int", "1/cm")
    print("input_reorg - ", reorg)

    with qr.energy_units("1/cm"):
        sd_tot.plot(show=True, axis=[0, 2000, 0.0, np.max(sd_tot.data)])
    cf.plot(show=True)

#######################################################################
# Dynamics
#######################################################################

# Test the setup by calculating the dynamics at same and diff energies
if _test_:
    energies1 = [energy] * num_mol
    energies2 = [energy - (100 * num_mol / 2)\
     + i * 100 for i in range(num_mol)]
    myFuncs.test_dynamics(agg_list=for_agg, energies=energies1)
    myFuncs.test_dynamics(agg_list=for_agg, energies=energies2)

########################################################################
# TwoD Spec
########################################################################

# Arrays for the direction on the laser plane
a_0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
a_90 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
# Aceto lab setup (for twod calculator)
lab_para = lab_settings(lab_settings.FOUR_WAVE_MIXING)
lab_para.set_laser_polarizations(a_0,a_0,a_0,a_0)
lab_perp = lab_settings(lab_settings.FOUR_WAVE_MIXING)
lab_perp.set_laser_polarizations(a_0,a_0,a_90,a_90)

# Setting the energies to the moleucles (energies defined above)
with qr.energy_units("1/cm"):
    for i, mol in enumerate(for_agg):
        mol.set_energy(1, energies0[i])

# Setting up the aggregate for the 2D spectra. Multiplicity must be 2
agg = qr.Aggregate(molecules=for_agg)
agg.set_coupling_by_dipole_dipole(epsr=1.21)
agg.build(mult=2)

# Can save the hamiltoniam diag matrix, diaged hamiltonian and dipoles
if _save_:
	myFuncs.save_eigen_data(agg = agg, file = eigen_file)

agg.diagonalize()
rwa = agg.get_RWA_suggestion()
with qr.energy_units('1/cm'):
    print(agg.get_Hamiltonian())

# Initialising the twod response calculator for the paralell laser
resp_cal_para = qr.TwoDResponseCalculator(
	t1axis=t13_ax,
	t2axis=t2_ax,
	t3axis=t13_ax,
	system=agg
	)
# Copying the response calculator for the perpendicular laser
resp_cal_perp = copy.deepcopy(resp_cal_para)

# Bootstrap is the place to add 0-padding to the response signal
# printEigen=True prints eigenvalues, printResp='string' prints response
# Response is calculated Converted into spectrum Stored in a container
resp_cal_para.bootstrap(
	rwa,
	verbose=True,
	pad=padding,
	printResp = para_resp_dir,
	lab=lab_para
	)
resp_para_cont = resp_cal_para.calculate()
spec_cont_para = resp_para_cont.get_TwoDSpectrumContainer()

resp_cal_perp.bootstrap(
	rwa,
	verbose=True,
	pad=padding,
	printResp = perp_resp_dir,
	lab=lab_perp
	)
resp_perp_cont = resp_cal_perp.calculate()
spec_cont_perp = resp_perp_cont.get_TwoDSpectrumContainer()

t2 = time.time()
print('calculation completed in ', (t2-t1), 's' )

########################################################################
# Printing and Saving
########################################################################

# initialising empy numpy arrays for points on the spectra > anisotropy
para = np.empty(len(t2_ax.data))
perp = np.empty(len(t2_ax.data))
en1 = 11000
en2 = 13500

# Runs through timessteps on the t2 axis. Gets spectrum from container
for i, tt2 in enumerate(t2_ax.data):
	twodPara = spec_cont_para.get_spectrum(t2_ax.data[i])
	para[i] = twodPara.get_max_value()
	with qr.energy_units('1/cm'):
		twodPara.plot()
		plt.xlim(en1, en2)
		plt.ylim(en1, en2)
		plt.title('para' + str(int(tt2)))
		if _save_:
			plt.savefig(save_dir + 'para' + str(int(tt2)) + '.png')
		if _show_:
			plt.show()

# Does the same as above but for the perpendicular laser setup
for i, tt2 in enumerate(t2_ax.data):
	twodPerp = spec_cont_perp.get_spectrum(t2_ax.data[i])
	perp[i] = twodPerp.get_max_value()
	with qr.energy_units('1/cm'):
		twodPerp.plot()
		plt.xlim(en1, en2)
		plt.ylim(en1, en2)
		plt.title('perp' + str(int(tt2)))
		if _save_:
			plt.savefig(save_dir + 'perp' + str(int(tt2)) + '.png')
		if _show_:
			plt.show()

########################################################################
# Analysis
########################################################################

# Calculates anisotropy
anis = (para - perp) / (para + (2 * perp))
print('anisotropy - ', anis)

# Loads in the data about the states, energies, vectors, dipols, orders
try:
	pig_en, state_en, eig_vecs, state_dips, dip_order, en_order =\
	 myFuncs.extracting_eigen_data(eigen_file)
	print('dipoles')
	print(state_dips[0])
	print('order')
	print(dip_order)
	print('state energies')
	print(state_en[0])
	print('pigment energies')
	print(pig_en[0])
	print('order')
	print(en_order)
except Exception:
	print('no eigen data')

# loads in the response data if saved
try:
	resp_data_para = np.load(para_resp_dir + 'respT0Pad.npz')
	print('para response - ', resp_data_para.files)
except Exception:
	print('no para resp data at ' + para_resp_dir + 'respT0Pad.npz')

# loads in the response data if saved
try:
	resp_data_perp = np.load(perp_resp_dir + 'respT0Pad.npz')
	print('perp response - ', resp_data_perp.files)
except Exception:
	print('no perp resp data at ' + perp_resp_dir + 'respT0Pad.npz')

plt.plot(resp_data_para['time'], resp_data_para['rTot'], 'r--')
plt.show()
