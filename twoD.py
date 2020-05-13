#!/usr/bin/env python

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
num_mol = 5
dipoleStrength = 5
# Creates a list of energies, centred around 'energy' and spaced by 100
energies = [energy-(100*num_mol/2) + i*100 for i in range(num_mol)]

staticDis = 300
reorganisation = 102
cTime = 100.0
temperature = 300

# Setting up the time axises for the three time intervals of the spectroscopy
t2s = qr.TimeAxis(0.0, 2, 100)
t13 = qr.TimeAxis(0.0, 500, 1)

#######################################################################
# Creation of aggregate
#######################################################################

# Setting up the time axis for the spectral density. 
# Needs to be long and with small intervals.
timeSD = qr.TimeAxis(0.0, 10000, 1)

# The paramaters for the spectral density
paramsReng = {"ftype": "B777",
              "alternative_form": True,
              "reorg": reorganisation,
              "T":temperature,
              "cortime":cTime}

# Creation of the spectral density and correlation function
with qr.energy_units("1/cm"):
    sdLowFreq = qr.SpectralDensity(timeSD, paramsReng)
cf = sdLowFreq.get_CorrelationFunction(temperature=temperature, ta=timeSD)

# PLotting spectral density and correlation function.
with qr.energy_units("1/cm"):
	sdLowFreq.plot(show=True, axis=[0, 2000, 0.0, np.max(sdLowFreq.data)])
cf.plot(show=True)

# Generates positions of molecules in a ring with equal distances apart
# Also generates dipoles running along the circumference of the ring
mol_positions, mol_dipoles = myFuncs.circularAgg(num_mol, dipoleStrength)

print('positions\n', mol_positions)
print('dipoles\n', mol_dipoles)

# Building the list of molecules which will make the aggregate
forAggregate = []
for i in range(num_mol):
	molecule = qr.Molecule()
	with qr.energy_units("1/cm"):
		molecule.set_energy(1, energies[i])
	molecule.position = [mol_positions[i][0], mol_positions[i][1], 0.0]
	molecule.set_dipole(0,1,[mol_dipoles[i][0], mol_dipoles[i][1], 0.0])
	molecule.set_transition_environment((0,1),cf)
	forAggregate.append(molecule)

#######################################################################
# Dynamics
#######################################################################

# Making an aggregate of multiplicity 1 for the dynamics
aggDyn = qr.Aggregate(molecules=forAggregate)
aggDyn.set_coupling_by_dipole_dipole(epsr=1.21)
aggDyn.build(mult=1)
aggDyn.diagonalize()
with qr.energy_units('1/cm'):
	print(aggDyn.get_Hamiltonian())

# Generates the propagator to describe motion in the aggregate
prop_Redfield = aggDyn.get_ReducedDensityMatrixPropagator(
	t13,
	relaxation_theory="stR",
	time_dependent=False, 
	secular_relaxation=True
	)

# Obtaining the density matrix
shp = aggDyn.get_Hamiltonian().dim
rho_i1 = qr.ReducedDensityMatrix(dim=shp, name="Initial DM")
# Setting initial conditions
rho_i1.data[shp-1,shp-1] = 1.0
# Propagating the system along the t13 time axis
rho_t1 = prop_Redfield.propagate(rho_i1, name="Redfield evo from agg")
rho_t1.plot(coherences=False, axis=[0,t13.length,0,1.0], show=True)

#######################################################################
# TwoD Spec
#######################################################################

agg = qr.Aggregate(molecules=forAggregate)
agg.set_coupling_by_dipole_dipole(epsr=1.21)
agg.build(mult=2)
agg.diagonalize()
rwa = agg.get_RWA_suggestion()

a_0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
a_90 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
labPara = lab_settings(lab_settings.FOUR_WAVE_MIXING)
labPara.set_laser_polarizations(a_0,a_0,a_0,a_0)
labPerp = lab_settings(lab_settings.FOUR_WAVE_MIXING)
labPerp.set_laser_polarizations(a_0,a_0,a_90,a_90)

tcalc_para = qr.TwoDResponseCalculator(
	t1axis=t13, 
	t2axis=t2s, 
	t3axis=t13, 
	system=agg
	)
tcalc_para.bootstrap(rwa, pad=500, verbose=True, lab=labPara)#
twods_para = tcalc_para.calculate()
paraContainer = twods_para.get_TwoDSpectrumContainer()

tcalc_perp = qr.TwoDResponseCalculator(
	t1axis=t13, 
	t2axis=t2s, 
	t3axis=t13, 
	system=agg
	)

########################## COPY IT

tcalc_perp.bootstrap(rwa, verbose=True, lab=labPerp, pad=500)#, pad=500
twods_perp = tcalc_perp.calculate()
perpContainer = twods_perp.get_TwoDSpectrumContainer()

paraPoint = []
anis = []
for i, tt2 in enumerate(t2s.data):
	twodPara = paraContainer.get_spectrum(t2s.data[i])
	paraPoint.append(twodPara.get_max_value())
	with qr.energy_units('1/cm'):
		twodPara.plot()
		plt.xlim(11400, 13100)
		plt.ylim(11400, 13100)
		#plt.savefig('para' + str(int(tt2)) + '.png')
		plt.show()

for i, tt2 in enumerate(t2s.data):
	twodPerp = perpContainer.get_spectrum(t2s.data[i])
	anis.append((paraPoint[i] - twodPerp.get_max_value())/(paraPoint[i] + (2 * twodPerp.get_max_value())))
	with qr.energy_units('1/cm'):
		twodPerp.plot()
		plt.xlim(11400, 13100)
		plt.ylim(11400, 13100)
		#plt.savefig('perp' + str(int(tt2)) + '.png')
		plt.show()

print(anis)
