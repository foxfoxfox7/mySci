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
from quantarhei.models.spectdens import SpectralDensityDB

import aceto
from aceto.lab_settings import lab_settings

_save_ = False
_show_ = True
_mock_ = False

saveDir = './exampleTrimerSep/'

numMol = 3
dipoleStrength = 5
staticDis = 300
energy = 12500
reorganisation = 102
width = 300.0
cTime = 100.0
temperature = 300

t13Step = 1
t13Length = 300
t2Step = 100
t2Length = 100
padding = 1000

coreN = 1
paraRepeatN = 1
loopN = 1
totalSpec = int(paraRepeatN*loopN)

#######################################################################
# Setup paramaters
#######################################################################

# Choose the energies (if its random, this is for the dynamics only)
energies = [energy] * numMol
#for i in range(numMol):
#    energies[i] = random.gauss(energy, staticDis)
# If you want to specifiy the enegies, remember to comment out the random 
# energy for each loop of the 2d calculation in 'calcTwoD' or 'calcTwoDMock'
#energies = [12100, 12200, 12300, 12400, 12500, 12600]#, 12250, 12350]

t2Count = int(t2Length/t2Step)+1
t13Count = int(t13Length/t13Step)+1
t2s = qr.TimeAxis(0.0, t2Count, t2Step)
t13 = qr.TimeAxis(0.0, t13Count, t13Step)

if _save_:
	try:
		os.mkdir(saveDir)
	except OSError:
		print ("Creation of the directory %s failed" % saveDir)

t1 = time.time()
print("Calculating spectra from 0 to ", t2Length, " fs\n")

#######################################################################
# Spectral density
#######################################################################

# Needs to have small step for the dynamics to converge
timeSD = qr.TimeAxis(0.0, 10000, 1)
db = SpectralDensityDB()

# Renger form of spec dens
#paramsReng = {"ftype": "B777",
#            "alternative_form": False,
#            "reorg": reorganisation,
#            "T":temperature,
#            "cortime":cTime}   
#with qr.energy_units("1/cm"):
#    sdLowFreq = qr.SpectralDensity(timeSD, paramsReng)
'''
# Over Damped Brownian Oscillaotr
params = {"ftype":"OverdampedBrownian", 
          "T":temperature, 
          "reorg":reorganisation, 
          "cortime":cTime}
with qr.energy_units('1/cm'):
    sdLowFreq = qr.SpectralDensity(timeSD, params)
'''
# Silbey form of spec dens
sdLowFreq = db.get_SpectralDensity(timeSD, "Renger_JCP_2002")

# Adding the high freq modes
sdWend = db.get_SpectralDensity(timeSD, "Wendling_JPCB_104_2000_5825")
ax = sdLowFreq.axis
sdWend.axis = ax
sdTot = sdLowFreq + sdWend
cf = sdTot.get_CorrelationFunction(temperature=temperature, ta=timeSD)

if _show_:
	with qr.energy_units("1/cm"):
	    sdTot.plot(show=True, axis=[0, 2000, 0.0, np.max(sdTot.data)])
	cf.plot(show=True)

#######################################################################
# Creation of the aggregate ring
#######################################################################

# The gap between pigments in LH2
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
print('energies\n', energies)

forAggregate = []
forAggregateMock = []
for i in range(numMol):
	molName = qr.Molecule()
	with qr.energy_units("1/cm"):
		molName.set_energy(1, energies[i])
	#molName.position = [0.0, i*10.0, 0.0]
	molName.position = [circle2[i][0], circle2[i][1], 0.0]
	#molName.set_dipole(0,1,[0.0, 10.0, 0.0])
	molName.set_dipole(0,1,[dipoles[i][0], dipoles[i][1], 0.0])
	forAggregateMock.append(molName)
	molName.set_transition_environment((0,1),cf)
	forAggregate.append(molName)

# Mock calculator uses sinple lineshapes instead of the correlation function (cf)
with qr.energy_units("1/cm"):
    for mol in forAggregateMock:
        mol.set_transition_width((0,1), width)

#######################################################################
# Creation of the aggregate protein
#######################################################################
'''
pdb_name = '3eoj'
file = PDBFile(pdb_name + ".pdb")

bcl_model = BacterioChlorophyll(model_type="PDB")
molecules = file.get_Molecules(model=bcl_model)
# The PDB names for the pigments
bcl_names = []
for m in molecules:
    bcl_names.append(m.name)
bcl_names.sort()

# Creates the input file if one is not there to be read
if not os.path.exists(pdb_name + "_input.txt"):
    file_input = open(pdb_name + "_input.txt", "w")
    for c, n in enumerate(bcl_names):
        file_input.write(n + "\tBChl" + str(c + 1) + "\t12500.0\n")
    file_input.close()

pigment_name = []
pigment_type = []
energies = []
with open(pdb_name + "_input.txt") as file_names:
    for line in file_names:
        pigment_name.append(line.split('\t')[0])
        pigment_type.append(line.split('\t')[1])
        energies.append(float(line.split('\t')[2]))
naming_map = dict(zip(pigment_name, pigment_type))

# Make a list of the molecules and set their molecule type
forAggregate = []
for name in pigment_name:
    for m in molecules:
        if m.name == name:
            m.set_name(naming_map[name])
            forAggregate.append(m)

for j, mol in enumerate(forAggregate):
    mol.set_transition_environment((0,1), cf)
    with energy_units("1/cm"):
        mol.set_energy(1, energies[j])
'''
#######################################################################
# Creation of the aggregate dimer
#######################################################################
'''
forAggregate = []

moleculeOne = qr.Molecule()
moleculeTwo = qr.Molecule()
energies = [12400, 12600]
with qr.energy_units("1/cm"):
	moleculeOne.set_energy(1, energies[0])
	moleculeTwo.set_energy(1, energies[1])
	#molName.set_energy(1, random.gauss(energies[i], 500))

moleculeOne.position = [0.0, 0.0, 0.0]
moleculeTwo.position = [0.0, 10.0, 0.0]
moleculeOne.set_dipole(0,1,[10.0, 0.0, 0.0])
moleculeTwo.set_dipole(0,1,[0.0, 10.0, 0.0])
forAggregateMock = [moleculeOne, moleculeTwo]

moleculeOne.set_transition_environment((0,1),cf)
moleculeTwo.set_transition_environment((0,1),cf)
forAggregate = [moleculeOne, moleculeTwo]

# Mock calculator uses sinple lineshapes instead of the correlation function (cf)
with qr.energy_units("1/cm"):
    for mol in forAggregateMock:
        mol.set_transition_width((0,1), width)
'''
#######################################################################
# Dynamics
#######################################################################

aggDyn = qr.Aggregate(molecules=forAggregate)
aggDyn.set_coupling_by_dipole_dipole(epsr=1.21)
aggDyn.build(mult=1)
aggDyn.diagonalize()
with qr.energy_units('1/cm'):
	print(aggDyn.get_Hamiltonian())

#t1Len = int(((t13.length-1)*t13.step)+1+padding)
t1Len = int(((t13.length+padding-1)*t13.step)+1)
print(t1Len)
t2propAxis = qr.TimeAxis(0.0, t1Len, 1)

#with eigenbasis_of(H2):
prop_Redfield = aggDyn.get_ReducedDensityMatrixPropagator(t2propAxis,
                                                       relaxation_theory="stR",
                                                       time_dependent=False, 
                                                       secular_relaxation=True)
shp = aggDyn.get_Hamiltonian().dim
rho_i1 = qr.ReducedDensityMatrix(dim=shp, name="Initial DM")
rho_i1.data[shp-1,shp-1] = 1.0  
rho_t1 = prop_Redfield.propagate(rho_i1, name="Redfield evo from agg")

rho_t1.plot(coherences=False, axis=[0,t1Len,0,1.0], show=False)
if _save_:
	plt.savefig(saveDir + 'dynamics.png')
if _show_:
	plt.show()

#######################################################################
# Calculation of the twoD PARA
#######################################################################

# Aggregate for twodcalculator
agg = qr.Aggregate(molecules=forAggregate)
agg.set_coupling_by_dipole_dipole(epsr=1.21)
agg.build(mult=2)
agg.diagonalize()
rwa = agg.get_RWA_suggestion()

#aggMock = qr.Aggregate(molecules=forAggregateMock)
#aggMock.set_coupling_by_dipole_dipole(epsr=1.21)
#aggMock.build(mult=2)
#aggMock.diagonalize()
#rwa = aggMock.get_RWA_suggestion()

a_0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
a_90 = np.array([0.0, 1.0, 0.0], dtype=np.float64)

# Aceto lab setup (for twod calculator)
labPara = lab_settings(lab_settings.FOUR_WAVE_MIXING)
labPara.set_laser_polarizations(a_0,a_0,a_0,a_0)
labPerp = lab_settings(lab_settings.FOUR_WAVE_MIXING)
labPerp.set_laser_polarizations(a_0,a_0,a_90,a_90)

# quantarhei lab setup (for mock calculator)
labParaMock = qr.LabSetup()
labPerpMock = qr.LabSetup()
labParaMock.set_polarizations(pulse_polarizations=[a_0,a_0,a_0], detection_polarization=a_0)
labPerpMock.set_polarizations(pulse_polarizations=[a_0,a_0,a_90], detection_polarization=a_90)


def calcTwoD(loopNum):

	container = []

	agg = qr.Aggregate(molecules=forAggregate)

	#with qr.energy_units('1/cm'):
	#	for i in range(len(forAggregate)):
	#		agg.monomers[i].set_energy(1, random.gauss(energy, staticDis))

	agg.set_coupling_by_dipole_dipole(epsr=1.21)
	agg.build(mult=2)
	agg.diagonalize()

	tcalc_para = qr.TwoDResponseCalculator(t1axis=t13, t2axis=t2s, t3axis=t13, system=agg)
	tcalc_para.bootstrap(rwa, pad=padding, verbose=True, lab=labPara, printEigen=False, printResp='paraResp')#, printResp='paraResp'
	twods_para = tcalc_para.calculate()
	paraContainer = twods_para.get_TwoDSpectrumContainer()
	container.append(paraContainer)

	tcalc_perp = qr.TwoDResponseCalculator(t1axis=t13, t2axis=t2s, t3axis=t13, system=agg)
	tcalc_perp.bootstrap(rwa, pad=padding, verbose=True, lab=labPerp, printEigen=False)#, printResp='perpResp'
	twods_perp = tcalc_perp.calculate()
	perpContainer = twods_perp.get_TwoDSpectrumContainer()
	container.append(perpContainer)

	return container

def calcTwoDMock(loopNum):

    container = []
    
    agg = qr.Aggregate(forAggregateMock)

    with qr.energy_units('1/cm'):
     	for i in range(len(forAggregateMock)):
     		agg.monomers[i].set_energy(1, random.gauss(energy, staticDis))

    agg.set_coupling_by_dipole_dipole()
    agg.build(mult=2)
    agg.diagonalize()

    rT = agg.get_RelaxationTensor(t2s, relaxation_theory='stR')
    eUt = qr.EvolutionSuperOperator(t2s, rT[1], rT[0]) # rT[1] = Hamiltonian
    eUt.set_dense_dt(t2Step)
    eUt.calculate(show_progress=False)

    mscPara = qr.MockTwoDResponseCalculator(t13, t2s, t13)
    mscPara.bootstrap(rwa=rwa, shape="Gaussian")
    contPara = mscPara.calculate_all_system(agg, eUt, labParaMock, show_progress=True)
    cont2DPara = contPara.get_TwoDSpectrumContainer(stype=qr.signal_TOTL)
    container.append(cont2DPara)

    mscPerp = qr.MockTwoDResponseCalculator(t13, t2s, t13)
    mscPerp.bootstrap(rwa=rwa, shape="Gaussian")
    contPerp = mscPerp.calculate_all_system(agg, eUt, labPerpMock, show_progress=True)
    cont2DPerp = contPerp.get_TwoDSpectrumContainer(stype=qr.signal_TOTL)
    container.append(cont2DPerp)

    return container


#######################################################################
# Creation of the spectra
#######################################################################

for l in range(loopN):
	tracemalloc.start()
	t2 = time.time()
	print('\nCalculating loop ' + str(l+1) + '...\n')

	grandContainer = []
	pool = mp.Pool(processes=coreN)
	if _mock_:
		grandContainer = pool.map(calcTwoDMock, [k for k in range(paraRepeatN)])
	else:
		grandContainer = pool.map(calcTwoD, [k for k in range(paraRepeatN)])
	pool.close()
	pool.join()

	for i, tt2 in enumerate(t2s.data):
		paraName = 'para' + str(i) + '.npy'
		perpName = 'perp' + str(i) + '.npy'

		twodPara = grandContainer[0][0].get_spectrum(t2s.data[i])
		twodPerp = grandContainer[0][1].get_spectrum(t2s.data[i])

		for j in range(paraRepeatN-1):
			twodPara.add_data(grandContainer[j+1][0].get_spectrum(t2s.data[i]).data)
			twodPerp.add_data(grandContainer[j+1][1].get_spectrum(t2s.data[i]).data)

		try:
			twodPara2 = qr.TwoDSpectrum()
			twodPara2.load_data(paraName)
			twodPara.add_data(twodPara2.data)

			twodPerp2 = qr.TwoDSpectrum()
			twodPerp2.load_data(perpName)
			twodPerp.add_data(twodPerp2.data)
		except:
			print('first loop')

		twodPara.save_data(paraName)
		twodPerp.save_data(perpName)

	grandContainer.clear()

	t2 = time.time()
	current, peak = tracemalloc.get_traced_memory()
	print("\n... calculated in " + str(t2-t1) + " sec")
	print(str(loopN - l - 1) + ' loops left')
	print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB\n")
	tracemalloc.stop()

#######################################################################
# display
#######################################################################

paraList = []
perpList = []

en1 = 11000
en2 = 13500
for i, tt2 in enumerate(t2s.data):
	paraName = 'para' + str(i) + '.npy'

	twodParaLoad = qr.TwoDSpectrum()
	twodParaLoad.load_data(paraName)
	twodParaLoad.set_data_type()
	twodParaLoad.set_axis_1(twodPara.xaxis)
	twodParaLoad.set_axis_3(twodPara.yaxis)
	twodParaLoad.devide_by(totalSpec)
	paraList.append(twodParaLoad)

	with qr.energy_units('1/cm'):
		twodParaLoad.plot()
		plt.xlim(en1, en2)
		plt.ylim(en1, en2)
		if _save_:
			plt.savefig(saveDir + 'para' + str(int(i)) + '.png')
		if _show_:
			plt.show()

	os.remove(paraName)
		
for i, tt2 in enumerate(t2s.data):
	perpName = 'perp' + str(i) + '.npy'

	twodPerpLoad = qr.TwoDSpectrum()
	twodPerpLoad.load_data(perpName)
	twodPerpLoad.set_data_type()
	twodPerpLoad.set_axis_1(twodPerp.xaxis)
	twodPerpLoad.set_axis_3(twodPerp.yaxis)
	twodPerpLoad.devide_by(totalSpec)
	perpList.append(twodPerpLoad)

	with qr.energy_units('1/cm'):
		twodPerpLoad.plot()
		plt.xlim(en1, en2)
		plt.ylim(en1, en2)
		if _save_:
			plt.savefig(saveDir + 'perp' + str(int(i)) + '.png')
		if _show_:
			plt.show()

	os.remove(perpName)

#######################################################################
# Anisotropy
#######################################################################

if _save_:
	with open(saveDir + 'anis.log', 'w') as anisFile:
			anisFile.write('Time step = ' + str(t2s.step) + '\n')

anisMax = []
for j, tt2 in enumerate(t2s.data):
	paraPoint = paraList[j].get_max_value()
	perpPoint = perpList[j].get_max_value()
	anisMax.append((paraPoint - perpPoint)/(paraPoint + (2 * perpPoint)))

print('anisMax = ' + str(anisMax) + '\n')

if _save_:
	with open(saveDir + 'anis.log', 'a') as anisFile:
		anisFile.write('anisMax = ' + str(anisMax) + '\n')

with qr.energy_units("1/cm"):
	for i in range(en1, en2, 100):
		anis = []
		for j, tt2 in enumerate(t2s.data):
			paraPoint = paraList[j].get_value_at(i, i).real
			perpPoint = perpList[j].get_value_at(i, i).real
			anis.append((paraPoint - perpPoint)/(paraPoint + (2 * perpPoint)))

		print('anis' + str(i) + ' = ' + str(anis))

		if _save_:
			with open(saveDir + 'anis.log', 'a') as anisFile:
				anisFile.write('anis' + str(i) + ' = ' + str(anis) + '\n')

