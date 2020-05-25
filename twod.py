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
import tracemalloc
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp

import quantarhei as qr
import aceto
from aceto.lab_settings import lab_settings

import myFuncs


energy = 12500
num_mol = 3
# Creates a list of energies, centred around 'energy' and spaced by 100
#energies = [energy - (100 * num_mol / 2) + i * 100 for i in range(num_mol)]
#energies = [energy] * num_mol

dipole_strength = 5
static_dis = 300
reorganisation = 102
cor_time = 100.0
temperature = 300

# Setting up the time axises for the three time intervals of the spectra
t2_ax = qr.TimeAxis(0.0, 2, 100)
t13_ax = qr.TimeAxis(0.0, 500, 1)
padding = 500

########################################################################
# Creation of aggregate
########################################################################

# Setting up the time axis for the spectral density.
# Needs to be long and with small intervals.
t_ax_sd = qr.TimeAxis(0.0, 10000, 1)

# The paramaters for the spectral density
params = {"ftype": "B777",
              "alternative_form": True,
              "reorg": reorganisation,
              "T":temperature,
              "cortime":cor_time}

# Creation of the spectral density and correlation function
with qr.energy_units("1/cm"):
    sd_low_freq = qr.SpectralDensity(t_ax_sd, params)
cf = sd_low_freq.get_CorrelationFunction(temperature=temperature, ta=t_ax_sd)

# PLotting spectral density and correlation function.
with qr.energy_units("1/cm"):
	sd_low_freq.plot(show=True, axis=[0, 2000, 0.0, np.max(sd_low_freq.data)])
cf.plot(show=True)

# Generates positions of molecules in a ring with equal distances apart
# Also generates dipoles running along the circumference of the ring
mol_positions, mol_dipoles = myFuncs.circularAgg(num_mol, dipole_strength)
print('positions\n', mol_positions)
print('dipoles\n', mol_dipoles)

# Building the list of molecules which will make the aggregate
for_agg = []
for i in range(num_mol):
	molecule = qr.Molecule()
	with qr.energy_units("1/cm"):
		molecule.set_energy(1, energies[i])
	molecule.position = [mol_positions[i][0], mol_positions[i][1], 0.0]
	molecule.set_dipole(0,1,[mol_dipoles[i][0], mol_dipoles[i][1], 0.0])
	molecule.set_transition_environment((0,1),cf)
	for_agg.append(molecule)

########################################################################
# Dynamics
########################################################################

# Making an aggregate of multiplicity 1 for the dynamics
agg_dyn = qr.Aggregate(molecules=for_agg)
agg_dyn.set_coupling_by_dipole_dipole(epsr=1.21)
agg_dyn.build(mult=1)
agg_dyn.diagonalize()
with qr.energy_units('1/cm'):
	print(agg_dyn.get_Hamiltonian())

# Creating a propagation axis length t13 plus padding with intervals 1
t1_len = int(((t13_ax.length+padding-1)*t13_ax.step)+1)
t2_prop_axis = qr.TimeAxis(0.0, t1_len, 1)

# Generates the propagator to describe motion in the aggregate
prop_Redfield = agg_dyn.get_ReducedDensityMatrixPropagator(
	t2_prop_axis,
	relaxation_theory="stR",
	time_dependent=False,
	secular_relaxation=True
	)

# Obtaining the density matrix
shp = agg_dyn.get_Hamiltonian().dim
rho_i1 = qr.ReducedDensityMatrix(dim=shp, name="Initial DM")
# Setting initial conditions
rho_i1.data[shp-1,shp-1] = 1.0
# Propagating the system along the t13_ax time axis
rho_t1 = prop_Redfield.propagate(rho_i1, name="Redfield evo from agg")
rho_t1.plot(coherences=False, axis=[0,t1_len,0,1.0], show=True)

########################################################################
# TwoD Spec
########################################################################

# Setting up the aggregate for the 2D spectra. Multiplicity must be 2
agg = qr.Aggregate(molecules=for_agg)
agg.set_coupling_by_dipole_dipole(epsr=1.21)
agg.build(mult=2)
agg.diagonalize()
rwa = agg.get_RWA_suggestion()

# Setting up the laser parameters
a_0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
a_90 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
lab_para = lab_settings(lab_settings.FOUR_WAVE_MIXING)
lab_para.set_laser_polarizations(a_0,a_0,a_0,a_0)
lab_perp = lab_settings(lab_settings.FOUR_WAVE_MIXING)
lab_perp.set_laser_polarizations(a_0,a_0,a_90,a_90)

# Initialising the twod response calculator for the paralell laser
resp_cal_para = qr.TwoDResponseCalculator(
	t1axis=t13_ax,
	t2axis=t2_ax,
	t3axis=t13_ax,
	system=agg
	)
# Copying the sresponse calculator for the perpendicular laser
resp_cal_perp = resp_cal_para

# Bootstrap is the place to add 0-padding to the response signal
# printEigen=True prints eigenvalues, printResp='string' prints response
# Response is calculated Converted into spectrum Stored in a container
resp_cal_para.bootstrap(
	rwa,
	lab=lab_para,
	verbose=True,
	pad=padding
	)
resp_para_cont = resp_cal_para.calculate()
spec_cont_para = resp_para_cont.get_TwoDSpectrumContainer()

resp_cal_perp.bootstrap(
	rwa,
	lab=lab_perp,
	verbose=True,
	pad=padding
	)
resp_perp_cont = resp_cal_perp.calculate()
spec_cont_perp = resp_perp_cont.get_TwoDSpectrumContainer()

########################################################################
# Printing and Saving
########################################################################

# initialising empy numpy arrays for points on the spectra > anisotropy
para = np.empty(len(t2_ax.data))
perp = np.empty(len(t2_ax.data))

# Runs through timessteps on the t2 axis. Gets spectrum from container
for i, tt2 in enumerate(t2_ax.data):
	twodPara = spec_cont_para.get_spectrum(t2_ax.data[i])
	para[i] = twodPara.get_max_value()
	with qr.energy_units('1/cm'):
		twodPara.plot()
		plt.xlim(11400, 13100)
		plt.ylim(11400, 13100)
		#plt.savefig('para' + str(int(tt2)) + '.png')
		plt.show()

# Does the same as above but for the perpendicular laser setup
for i, tt2 in enumerate(t2_ax.data):
	twodPerp = spec_cont_perp.get_spectrum(t2_ax.data[i])
	perp[i] = twodPerp.get_max_value()
	with qr.energy_units('1/cm'):
		twodPerp.plot()
		plt.xlim(11400, 13100)
		plt.ylim(11400, 13100)
		#plt.savefig('perp' + str(int(tt2)) + '.png')
		plt.show()

# Calculates anisotropy
anis = (para - perp) / (para + (2 * perp))
print(anis)

