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

#import myFuncs


numMol = 3
dipoleStrength = 5
energies = [12100, 12200, 12300, 12400, 12500, 12600]#, 12250, 12350]

staticDis = 300
reorganisation = 102
cTime = 100.0
temperature = 300

t2s = qr.TimeAxis(0.0, 2, 100)
t13 = qr.TimeAxis(0.0, 500, 1)

#######################################################################
# Creation of aggregate
#######################################################################

timeSD = qr.TimeAxis(0.0, 10000, 1)
paramsReng = {"ftype": "B777",
            "alternative_form": True,
            "reorg": reorganisation,
            "T":temperature,
            "cortime":cTime}   
with qr.energy_units("1/cm"):
    sdLowFreq = qr.SpectralDensity(timeSD, paramsReng)
cf = sdLowFreq.get_CorrelationFunction(temperature=temperature, ta=timeSD)

with qr.energy_units("1/cm"):
	sdLowFreq.plot(show=True, axis=[0, 2000, 0.0, np.max(sdLowFreq.data)])
cf.plot(show=True)

#molPositions = myFuncs.circularAgg(numMol, dipoleStrength)

# This code just makes a ring of positions where each is roughly 8.7A appart
# And then gives dipole vectors at tangents to the circle, with dipoleStrnegth magnitude
proteinDis = 8.7
difference = 0.6
r = 5
while difference > 0.1:
	t = np.linspace(0, np.pi * 2, numMol+1)
	x = r * np.cos(t)
	y = r * np.sin(t)
	circle = np.c_[x, y]

	artificialDis = math.sqrt(((circle[0][0]-circle[1][0])**2)+((circle[0][1]-circle[1][1])**2))
	difference = abs(artificialDis-proteinDis)

	if artificialDis > proteinDis:
		r = r - 0.1
	elif artificialDis < proteinDis:
		r = r + 0.1
circle2 = np.delete(circle, numMol, 0)

dipoles = np.empty([numMol,2])
mag = math.sqrt((circle[0][0]**2)+(circle[0][1]**2))
for i in range(numMol):
	dipoles[i][0] = -circle2[i][1]
	dipoles[i][1] = circle2[i][0]
	dipoles[i][0] = dipoles[i][0] / mag
	dipoles[i][1] = dipoles[i][1] / mag
	dipoles[i][0] = dipoles[i][0] * dipoleStrength
	dipoles[i][1] = dipoles[i][1] * dipoleStrength

print('positions\n', circle2)
print('dipoles\n', dipoles)

forAggregate = []
for i in range(numMol):
	molName = qr.Molecule()
	with qr.energy_units("1/cm"):
		molName.set_energy(1, energies[i])
	#molName.position = [molPositions[0][i][0], molPositions[0][i][1], 0.0]
	#molName.set_dipole(0,1,[molPositions[1][i][0], molPositions[1][i][1], 0.0])
	molName.position = [circle2[i][0], circle2[i][1], 0.0]
	molName.set_dipole(0,1,[dipoles[i][0], dipoles[i][1], 0.0])
	molName.set_transition_environment((0,1),cf)
	forAggregate.append(molName)

#######################################################################
# Dynamics
#######################################################################

aggDyn = qr.Aggregate(molecules=forAggregate)
aggDyn.set_coupling_by_dipole_dipole(epsr=1.21)
aggDyn.build(mult=1)
aggDyn.diagonalize()
with qr.energy_units('1/cm'):
	print(aggDyn.get_Hamiltonian())

#with eigenbasis_of(H2):
prop_Redfield = aggDyn.get_ReducedDensityMatrixPropagator(t13,
                                                       relaxation_theory="stR",
                                                       time_dependent=False, 
                                                       secular_relaxation=True)
shp = aggDyn.get_Hamiltonian().dim
rho_i1 = qr.ReducedDensityMatrix(dim=shp, name="Initial DM")
rho_i1.data[shp-1,shp-1] = 1.0  
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

tcalc_para = qr.TwoDResponseCalculator(t1axis=t13, t2axis=t2s, t3axis=t13, system=agg)
tcalc_para.bootstrap(rwa, pad=500, verbose=True, lab=labPara)#
twods_para = tcalc_para.calculate()
paraContainer = twods_para.get_TwoDSpectrumContainer()

tcalc_perp = qr.TwoDResponseCalculator(t1axis=t13, t2axis=t2s, t3axis=t13, system=agg)
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
		plt.savefig('para' + str(int(tt2)) + '.png')
		plt.show()

for i, tt2 in enumerate(t2s.data):
	twodPerp = perpContainer.get_spectrum(t2s.data[i])
	anis.append((paraPoint[i] - twodPerp.get_max_value())/(paraPoint[i] + (2 * twodPerp.get_max_value())))
	with qr.energy_units('1/cm'):
		twodPerp.plot()
		plt.xlim(11400, 13100)
		plt.ylim(11400, 13100)
		plt.savefig('perp' + str(int(tt2)) + '.png')
		plt.show()

print(anis)
