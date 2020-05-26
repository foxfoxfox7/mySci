#!/usr/bin/env python

import os
import math
import time
import random
import copy
import re
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp

import quantarhei as qr
from quantarhei.models.spectdens import SpectralDensityDB

#import aceto
#from aceto.lab_settings import lab_settings

import myFuncs

_test_ = False
_save_ = False
_show_ = True

save_dir = './exampleTrimerSep/'

num_mol = 8
dipole_strength = 3
static_dis = 300
energy = 12500
reorganisation = 102
width = 300.0
cor_time = 100.0
temperature = 300

# Number of 2d containers calculated each loop
n_per_loop = 2
# Number of cores used for each loop
n_cores = 1
# Number of times this is repeated
n_loops = 1
totalSpec = int(n_per_loop*n_loops)

t13_ax_ax_step = 1
t13_ax_ax_len = 300
t2_ax_step = 100
t2_ax_len = 100
padding = 1000

#######################################################################
# Setup paramaters
#######################################################################

# The axis for time gaps 1 and 3
t13_ax_ax_count = int(t13_ax_ax_len/t13_ax_ax_step)+1
t13_ax = qr.TimeAxis(0.0, t13_ax_ax_count, t13_ax_ax_step)

# The axis for time gap 2
t2_ax_count = int(t2_ax_len/t2_ax_step)+1
t2_ax = qr.TimeAxis(0.0, t2_ax_count, t2_ax_step)

if _save_:
    try:
        os.mkdir(save_dir)
    except OSError:
        print("Creation of the directory %s failed" % save_dir)

t1 = time.time()
print("Calculating spectra from 0 to ", t2_ax_len, " fs\n")


#######################################################################
# Creation of the aggregate ring
#
# A function to create a list of quantarhei molecules
# withcicular coordinates evenly spaceed and dipole
# vectors running round the circumference all in the same direction
#######################################################################

for_agg = myFuncs.circularAgg(num_mol, dipole_strength)

#######################################################################
# Creation of the aggregate protein
#
# Takes the name of a pdb file (without the extension) and converts it
# into a list of molecules with positions and dipoles. The only molecle
# it looks for is Bacteriochlorophyll
#######################################################################

#for_agg = myFuncs.bacteriochl_agg('3eoj')

#######################################################################
# Creation of the aggregate dimer

# Just a simple dimer for testing. change the positions and dipoles
#######################################################################
'''
moleculeOne = qr.Molecule()
moleculeTwo = qr.Molecule()
moleculeOne.position = [0.0, 0.0, 0.0]
moleculeTwo.position = [0.0, 10.0, 0.0]
moleculeOne.set_dipole(0,1,[10.0, 0.0, 0.0])
moleculeTwo.set_dipole(0,1,[0.0, 10.0, 0.0])
for_agg = [moleculeOne, moleculeTwo]
'''
#######################################################################
# Spectral density
#######################################################################

# Needs to have small step for the dynamics to converge
t_ax_sd = qr.TimeAxis(0.0, 10000, 1)
db = SpectralDensityDB()

# Parameters for spectral density. ODBO, Renger or Silbey
params = {"ftype": "OverdampedBrownian",
            #"ftype": "B777",
            "alternative_form": True,
            "reorg": reorganisation,
            "T":temperature,
            "cortime":cor_time}
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
    with qr.energy_units("1/cm"):
        sd_tot.plot(show=True, axis=[0, 2000, 0.0, np.max(sd_tot.data)])
    cf.plot(show=True)

#######################################################################
# Dynamics
#######################################################################

def test_dynamics(agg_list, energies):
    # Adding the energies to the molecules. Neeed to be done before agg
    with qr.energy_units("1/cm"):
        for i, mol in enumerate(agg_list):
            mol.set_energy(1, energies[i])

    # Creation of the aggregate for dynamics. multiplicity can be 1
    agg = qr.Aggregate(molecules=agg_list)
    agg.set_coupling_by_dipole_dipole(epsr=1.21)
    agg.build(mult=1)
    agg.diagonalize()
    with qr.energy_units('1/cm'):
        print(agg.get_Hamiltonian())

    # Creating a propagation axis length t13_ax plus padding with intervals 1
    t1_len = int(((t13_ax.length+padding-1)*t13_ax.step)+1)
    t2_prop_axis = qr.TimeAxis(0.0, t1_len, 1)

    # Generates the propagator to describe motion in the aggregate
    prop_Redfield = agg.get_ReducedDensityMatrixPropagator(
        t2_prop_axis,
        relaxation_theory="stR",
        time_dependent=False,
        secular_relaxation=True
        )

    # Obtaining the density matrix
    shp = agg.get_Hamiltonian().dim
    rho_i1 = qr.ReducedDensityMatrix(dim=shp, name="Initial DM")
    # Setting initial conditions
    rho_i1.data[shp-1,shp-1] = 1.0
    # Propagating the system along the t13_ax_ax time axis
    rho_t1 = prop_Redfield.propagate(rho_i1, name="Redfield evo from agg")
    rho_t1.plot(coherences=False, axis=[0,t1_len,0,1.0], show=True)

# Test the setup by calculating the dynamics at same and diff energies
if _test_:
    energies1 = [energy] * num_mol
    energies2 = [energy - (100 * num_mol / 2)\
     + i * 100 for i in range(num_mol)]
    test_dynamics(agg_list=for_agg, energies=energies1)
    test_dynamics(agg_list=for_agg, energies=energies2)

#######################################################################
# state analysis
#######################################################################

def getLine(fileName, thisLine, plus = 0):

	lineList = []
	with open(fileName, 'r') as f:
		for lineNum, line in enumerate(f):
			line=line.strip()
			if re.search('^' + thisLine, line):
				lineList.append(lineNum + plus)

	return lineList

def getDataBetween(fileName, lineStart, lineFin):

	dataInput=[]
	with open(fileName, 'r') as f:
		for line_num, line in enumerate(f):
			line=line.strip()
			if (line_num >= lineStart) and (line_num < lineFin):
				data=line.split()
				dataInput+=data

	return dataInput

def save_eigen_data(agg, file):
	nM = agg.nmono
	print('nM', nM)

	H = agg.get_Hamiltonian()
	SS = H.diagonalize()
	trans1 = SS[1:nM+1,1:nM+1]
	H.undiagonalize()
	hamil = agg.HH[1:nM+1,1:nM+1]
	
	with open(file, 'a') as f:
		f.write('Hamiltonian\n')
		np.savetxt(f, hamil)

		f.write('Transformation Matrix\n')
		np.savetxt(f, trans1)

	agg.diagonalize()
	diag = agg.HH[1:nM+1,1:nM+1]

	with open(file, 'a') as f:
		f.write('Diagonalized\n')
		np.savetxt(f, agg.HH[1:nM+1,1:nM+1])

		f.write('Dipoles\n')
		np.savetxt(f, agg.D2[0][1:].reshape(1, nM))

def extracting_eigen_data(file):
	startLine = getLine(file, 'Hamiltonian')
	endLine = getLine(file, 'Transformation')
	nM = endLine[0] - startLine[0] - 1
	numIter = len(startLine)
	print('number of Iterations - ' + str(numIter))
	print('number of molecules/sites - ' + str(nM))

	hamilStartLine = getLine(file, 'Hamiltonian', plus = 1)
	hamilEndLine = getLine(file, 'Hamiltonian', plus = nM+1)
	diagStartLine = getLine(file, 'Diagonalized', plus = 1)
	diagEndLine = getLine(file, 'Diagonalized', plus = nM+1)
	eigenStartLine = getLine(file, 'Transformation', plus = 1)
	eigenEndLine = getLine(file, 'Transformation', plus = nM+1)
	dipoleStartLine = getLine(file, 'Dipoles', plus = 1)
	dipoleEndLine = getLine(file, 'Dipoles', plus = 2)

	hamilList = []
	hamil = []
	pig_en = []

	diagList = []
	diag = []
	state_en = []

	eigenList = []
	eig_vecs =[]

	state_dipsStr = []
	state_dips = []

	for i in range(numIter):
		hamilList.append(getDataBetween(file, hamilStartLine[i], hamilEndLine[i]))
		hamil.append(np.array(hamilList[i]).reshape(int(len(hamilList[i])/nM),nM))
		pig_en.append(np.diagonal(hamil[i]).astype(np.float))

		diagList.append(getDataBetween(file, diagStartLine[i], diagEndLine[i]))
		diag.append(np.array(diagList[i]).reshape(int(len(diagList[i])/nM),nM))
		state_en.append(np.diagonal(diag[i]).astype(np.float))

		eigenList.append(getDataBetween(file, eigenStartLine[i], eigenEndLine[i]))
		eig_vecs.append(np.transpose(np.array(eigenList[i]).reshape(int(len(eigenList[i])/nM),nM)))

		state_dipsStr.append(getDataBetween(file, dipoleStartLine[i], dipoleEndLine[i]))
		state_dips.append((np.array(state_dipsStr[i]).astype(np.float)))

	dip_order = np.flip(np.argsort(state_dips[0]))
	en_order = np.argsort(pig_en[0])

	return pig_en, state_en, eig_vecs, state_dips, dip_order, en_order

def displayState(func, order, col = 'r', plot = False):

	print(col)
	print('state ', order)
	print('Dipole ', state_dips[0][order])
	print('Energy ', state_en[0][order])

	colour = col + '--'
	plt.plot(func[order].astype(np.float), colour)
	if plot:
		plt.show()

def complexify(func):
	
	Nvec = len(func)

	vecf = np.fft.fft(func)
	vecf[0] = vecf[0]/2.0
	vecf[int(Nvec/2)] = vecf[int(Nvec/2)]/2.0
	vecf[int(Nvec/2)+1:] = 0.0
	vecf = 2.0*vecf
	veci = np.fft.ifft(vecf)

	Norm = np.dot(np.conj(veci), veci)
	vecb = copy.deepcopy(veci)
	veci = veci/np.sqrt(Norm)

	return veci

def plotAState(func):

	plt.plot(func.imag, 'm--')
	plt.plot(func.real, 'k--')
	plt.plot(np.absolute(func.real)+ np.absolute(func.imag), 'r--')

	plt.show()

def IPR(vec):

	I2r = []
	I1r = []
	I2r_sum = []

	I2r[Ne-1] = np.sum(np.abs(vec)**4)
	I1r[Ne-1] = np.sum(np.abs(vec)**2)


	print(I1r, 1.0/I2r)

	I2r_sum[Ne-1] += 1.0/I2r[Ne-1]


#######################################################################
# Getting aggregate data
#######################################################################

file_name = 'data/eigen_data.txt'
f = open(file_name,"w+")
f.close()

for i in range(n_per_loop):

	for_agg_copy = for_agg
	with qr.energy_units("1/cm"):
		for i, mol in enumerate(for_agg_copy):
			mol.set_energy(1, random.gauss(energy, static_dis))

	agg = qr.Aggregate(molecules=for_agg_copy)
	agg.set_coupling_by_dipole_dipole(epsr=1.21)
	agg.build()

	save_eigen_data(agg = agg, file = file_name)

pig_en, state_en, eig_vecs, state_dips, dip_order, en_order = extracting_eigen_data(file_name)

#######################################################################
# Analysis
#######################################################################

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

ll = 100
cos_func = [np.cos(((2*math.pi)/ll)*i) for i in range(ll)]
cos_complex = complexify(cos_func)
plotAState(cos_complex)

plt.plot(state_dips[0], 'y--')
plt.show()

colours = ['b','g','r','c','m','y','k']
complexEigen = []
complexEigen2 = []
modEigen = []
modEigenNorm = []
for i in range(num_mol):
	complexEigen2.append(complexify(eig_vecs[0][i].astype(np.float)))
	#modEigen.append(np.absolute(complexEigen[i]))
	#modEigenNorm.append(modEigen[i]/sum(modEigen[i]))

# range shoudl be less or equal to than number of nolecules
for i in range(4):
	#displayState(np.absolute(complexEigen2), dip_order[i], colours[i])
	displayState(np.absolute(complexEigen2), i, colours[i])
plt.show()


'''
#complexEigen3 = np.zeros(shape=(numIter,num_mol))
complexEigen3 = [[0 for x in range(num_mol)] for y in range(numIter)] 
for j in range(numIter):
	for i in range(num_mol):
		complexEigen3[j][i] = complexify(eig_vecs[j][i].astype(np.float))

print(complexEigen3)
print('space')
print(complexEigen3[0])

#plt.plot(eig_vecs[0][0].astype(np.float), 'k--')

for i in range(4):
	colour = colours[i] + '--'
	plt.plot(complexEigen2[i].real, 'k--')
	plt.plot(complexEigen2[i].imag, 'm--')
	plt.plot(np.absolute(complexEigen2[i]), colour)
	plt.show()



print('state -', dip_order[0])
finalEigen = complexify(eig_vecs[0][dip_order[0]].astype(np.float))
print('state -', dip_order[-1])
finalEigen = complexify(eig_vecs[0][dip_order[-1]].astype(np.float))



for i in range (4):
	print(colours[i])
	#complexIPR = np.sum(np.abs(complexEigen2[i])**2)
	#realIPR = np.sum(np.abs(complexEigen2[i].real)**2)
	print('complexIPR')
	
	IPR(complexEigen2[i])
	print('realIPR')
	IPR(complexEigen2[i].real)



for i in range(4):
	#displayState(state_en[0], i, colours[i])
	displayState(modEigenNorm, dip_order[i], colours[i], plot = True)
plt.show()




plt.plot(modEigenNorm[dip_order[0]].real)
plt.plot(modEigenNorm[dip_order[0]].imag)
plt.plot(modEigenNorm[dip_order[0]])
plt.show()



listTest = []
listFinal = []
for i in range(4):
	listTest.append(eig_vecs[0][dip_order[i]].astype(np.float))
	listFinal.append(complexify(listTest[i]))
'''


