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
from quantarhei import Aggregate
from quantarhei import energy_units
from quantarhei import eigenbasis_of 
from quantarhei import CorrelationFunction
from quantarhei import TimeAxis
from quantarhei import Molecule
from quantarhei import SpectralDensity
from quantarhei import DFunction
from quantarhei import Molecule

from quantarhei.builders.pdb import PDBFile
from quantarhei.core.units import kB_int 
from quantarhei.core.units import convert
from quantarhei.models.bacteriochlorophylls import BacterioChlorophyll
from quantarhei.models.spectdens import SpectralDensityDB
from quantarhei.spectroscopy.twodcalculator import TwoDResponseCalculator
from quantarhei.spectroscopy.twodcontainer import TwoDResponseContainer
#from quantarhei.spectroscopy.twod22 import TwoDSpectrumCalculator
from quantarhei.spectroscopy.twod22 import TwoDSpectrum

#import aceto
#from aceto.lab_settings import lab_settings


r = 5
numMol = 3
coreN = 1
repeatN = 2
dipoleStrength = 5
staticDis = 300
reorganisation = 100

t2Step = 200
t2Count = 2
t13Step = 2
t13Count = 300
t2s = TimeAxis(0.0, t2Count, t2Step)
t13 = TimeAxis(0.0, t13Count, t13Step)
timeTotal = TimeAxis(0.0, 3*t13Count, t13Step)

two_pi = np.pi * 2
proteinDis = 8.7
difference = 0.6
temperature = 300

params = dict(ftype="OverdampedBrownian", 
				T=temperature, 
				reorg=reorganisation, 
				cortime=100.0)
with energy_units('1/cm'):
	cf = CorrelationFunction(timeTotal, params)
	#cf = CorrelationFunction(ta, params)

#######################################################################
# Creation of the aggregate basic
#######################################################################

while difference > 0.1:
	t = np.linspace(0, two_pi, numMol+1)
	x = r * np.cos(t)
	y = r * np.sin(t)
	circle = np.c_[x, y]

	artificialDis = math.sqrt(((circle[0][0]-circle[1][0])**2)+((circle[0][1]-circle[1][1])**2))
	difference = abs(artificialDis-proteinDis)

	if artificialDis > proteinDis:
		r = r - 0.1
	elif artificialDis < proteinDis:
		r = r + 0.1

dipoles = np.empty([numMol,2])
mag = math.sqrt((circle[0][0]**2)+(circle[0][1]**2))
for i in range(numMol):
	dipoles[i][0] = -circle[i][1]
	dipoles[i][1] = circle[i][0]
	dipoles[i][0] = dipoles[i][0] / mag
	dipoles[i][1] = dipoles[i][1] / mag
	dipoles[i][0] = dipoles[i][0] * dipoleStrength
	dipoles[i][1] = dipoles[i][1] * dipoleStrength

energies = [12500] * numMol
#energies = [12000, 12100, 12200, 12300, 12400, 12150, 12250, 12350]
forAggregate = []
for i in range(numMol):
	molName = 'molecule' + str(i)
	molName = Molecule()
	with energy_units("1/cm"):
		molName.set_energy(1, energies[i])
		#molName.set_energy(1, random.gauss(energies[i], staticDis))
	molName.set_transition_environment((0,1),cf)
	#molName.position = [0.0, i*10.0, 0.0]
	molName.position = [circle[i][0], circle[i][1], 0.0]
	#molName.set_dipole(0,1,[0.0, 10.0, 0.0])
	molName.set_dipole(0,1,[dipoles[i][0], dipoles[i][1], 0.0])
	forAggregate.append(molName)

#######################################################################
# Creation of the aggregate protein
#######################################################################
'''
pdb_name = 'LH1'
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

molOneName = 'molecule1'
moleculeOne = Molecule()
moleculeTwo = Molecule()
with energy_units("1/cm"):
	moleculeOne.set_energy(1, 12000)
	moleculeTwo.set_energy(1, 13000)
	#molName.set_energy(1, random.gauss(energies[i], 500))
moleculeOne.set_transition_environment((0,1),cf)
moleculeTwo.set_transition_environment((0,1),cf)
moleculeOne.position = [0.0, 0.0, 0.0]
moleculeTwo.position = [0.0, 10.0, 0.0]
moleculeOne.set_dipole(0,1,[0.0, -10.0, 0.0])
moleculeTwo.set_dipole(0,1,[0.0, 10.0, 0.0])

forAggregate = [moleculeOne, moleculeTwo]
'''
#######################################################################
# Dynamics
#######################################################################
'''
agg2 = Aggregate(molecules=forAggregate)
agg2.set_coupling_by_dipole_dipole(epsr=1.21)
agg2.build()
H2 = agg2.get_Hamiltonian()
with eigenbasis_of(H2):
	prop_Redfield = agg2.get_ReducedDensityMatrixPropagator(timeTotal,
															relaxation_theory="stR",
															time_dependent=False, 
															secular_relaxation=True)

	shp2 = H2.dim
	rho_i1 = qr.ReducedDensityMatrix(dim=shp2, name="Initial DM")
	rho_i1.data[shp2-1,shp2-1] = 1.0  
	rho_t1 = prop_Redfield.propagate(rho_i1, name="Redfield evolution from aggregate")
	rho_t1.plot(coherences=False, axis=[0,t13Step*t13Count,0,1.0], show=True)
'''

#######################################################################
# state analysis
#######################################################################

def getEigen():

	print('nM', nM)
	agg = Aggregate(molecules=forAggregate)
	for j in range(nM):
		with energy_units('1/cm'):
			agg.monomers[j].set_energy(1, random.gauss(energies[j], staticDis))
	agg.set_coupling_by_dipole_dipole(epsr=1.21)
	agg.build() #mult=2

	N = len(forAggregate)

	H = agg.get_Hamiltonian()
	print(H)
	SS = H.diagonalize()
	print(H)
	trans1 = SS[1:N+1,1:N+1]
	H.undiagonalize()
	hamil = agg.HH[1:N+1,1:N+1]
	
	with open('transData.txt', 'a') as f:

		f.write('Hamiltonian\n')
		#np.savetxt(f, agg.HH[1:N+1,1:N+1])
		np.savetxt(f, hamil)

		f.write('Transformation Matrix\n')
		np.savetxt(f, trans1)

	agg.diagonalize()
	diag = agg.HH[1:N+1,1:N+1]

	with open('transData.txt', 'a') as f:
		
		f.write('Diagonalized\n')
		np.savetxt(f, agg.HH[1:N+1,1:N+1])

		f.write('Dipoles\n')
		np.savetxt(f, agg.D2)

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

def getImag(func, show = False, plot = True):

	length = int(len(func))
	half = int(len(func)/2)

	if show:
		print('eigenvector')
		print(func)
	vecFT = (np.fft.ifft(func))
	if show:
		print('fourier transformed vector')
		print(vecFT)
	vecFT[0:half] = 0
	vecIfftProc = (vecFT *2)
	if show:
		print('half elements to 0 and doubled')
		print(vecIfftProc)
	vecFinal = (np.fft.fft(vecIfftProc))
	if show:
		print('final')
		print(vecFinal)

	if plot:
		plt.plot(func, 'b-')
		plt.plot(vecFinal.real, 'r--')
		plt.plot(vecFinal.imag, 'g--')
		plt.plot(np.absolute(vecFinal), 'm--')
		plt.show()

	return vecFinal

def displayState(func, order, col = 'r', plot = False):

	print(col)
	print('state ', order)
	print('Dipole ', stateDipoleList[0][order])
	print('Energy ', stateEnList[0][order])

	colour = col + '--'
	plt.plot(func[order].astype(np.float), colour)
	if plot:
		plt.show()

def Tomas(func):
	
	Nvec = len(func)
	'''
	vecf = np.fft.ifft(func)
	vecf[0] = vecf[0]/2.0
	vecf[int(Nvec/2):] = 0.0
	vecf = 2.0*vecf
	veci = np.fft.fft(vecf)
	'''

	vecf = np.fft.fft(func)
	vecf[0] = vecf[0]/2.0
	vecf[int(Nvec/2)] = vecf[int(Nvec/2)]/2.0
	vecf[int(Nvec/2)+1:] = 0.0
	vecf = 2.0*vecf
	veci = np.fft.ifft(vecf)

	Norm = np.dot(np.conj(veci), veci)
	#print("Norm: ", Norm)
	vecb = copy.deepcopy(veci)
	veci = veci/np.sqrt(Norm)

	return veci

def plotAState(func, num):
	Nvec = len(func)
	'''
	vecf = np.fft.ifft(func)
	vecf[0] = vecf[0]/2.0
	vecf[int(Nvec/2):] = 0.0
	vecf = 2.0*vecf
	veci = np.fft.fft(vecf)
	'''

	vecf = np.fft.fft(func)
	vecf[0] = vecf[0]/2.0
	vecf[int(Nvec/2)] = vecf[int(Nvec/2)]/2.0
	vecf[int(Nvec/2)+1:] = 0.0
	vecf = 2.0*vecf
	veci = np.fft.ifft(vecf)

	Norm = np.dot(np.conj(veci), veci)
	#print("Norm: ", Norm)
	vecb = copy.deepcopy(veci)
	veci = veci/np.sqrt(Norm)


	plt.plot([i for i in range(len(vec))], vec)
	#complexEigentest = (getImag(vec, plot = False))
	plt.plot(veci.imag, 'm--')
	plt.plot(veci.real, 'k--')
	colour = colours[i] + '--'
	plt.plot(np.absolute(veci.real)+ np.absolute(veci.imag), colour)

	plt.show()


#######################################################################
# Getting aggregate data
#######################################################################

f = open("transData.txt","w+")
f.close()

nM = len(forAggregate)
numRepeat = 1
for i in range(numRepeat):
	getEigen()

#######################################################################
# Extracting information
#######################################################################

startLine = getLine('transData.txt', 'Hamiltonian')
endLine = getLine('transData.txt', 'Transformation')
nM = endLine[0] - startLine[0] - 1
numIter = len(startLine)
print('number of Iterations - ' + str(numIter))
print('number of molecules/sites - ' + str(nM))

hamilStartLine = getLine('transData.txt', 'Hamiltonian', plus = 1)
hamilEndLine = getLine('transData.txt', 'Hamiltonian', plus = nM+1)
diagStartLine = getLine('transData.txt', 'Diagonalized', plus = 1)
diagEndLine = getLine('transData.txt', 'Diagonalized', plus = nM+1)
eigenStartLine = getLine('transData.txt', 'Transformation', plus = 1)
eigenEndLine = getLine('transData.txt', 'Transformation', plus = nM+1)
dipoleStartLine = getLine('transData.txt', 'Dipoles', plus = 1)
dipoleEndLine = getLine('transData.txt', 'Dipoles', plus = 2)

hamilList = []
hamil = []
pigEnList = []

diagList = []
diag = []
stateEnList = []

eigenList = []
eigenVecStr =[]

stateDipoleListStr = []
stateDipoleList = []

for i in range(numIter):
	hamilList.append(getDataBetween('transData.txt', hamilStartLine[i], hamilEndLine[i]))
	hamil.append(np.array(hamilList[i]).reshape(int(len(hamilList[i])/nM),nM))
	pigEnList.append(np.diagonal(hamil[i]).astype(np.float))

	diagList.append(getDataBetween('transData.txt', diagStartLine[i], diagEndLine[i]))
	diag.append(np.array(diagList[i]).reshape(int(len(diagList[i])/nM),nM))
	stateEnList.append(np.diagonal(diag[i]).astype(np.float))

	eigenList.append(getDataBetween('transData.txt', eigenStartLine[i], eigenEndLine[i]))
	eigenVecStr.append(np.transpose(np.array(eigenList[i]).reshape(int(len(eigenList[i])/nM),nM)))

	stateDipoleListStr.append(getDataBetween('transData.txt', dipoleStartLine[i], dipoleEndLine[i]))
	del stateDipoleListStr[i][0]
	stateDipoleList.append((np.array(stateDipoleListStr[i]).astype(np.float)))

dipoleOrder = np.flip(np.argsort(stateDipoleList[0]))
pigEnergyOrder = np.argsort(pigEnList[0])

print('dipoles')
print(stateDipoleList[0])
print('order')
print(dipoleOrder)
print('state energies')
print(stateEnList[0])
print('pigment energies')
print(pigEnList[0])
print('order')
print(pigEnergyOrder)

#######################################################################
# Analysis
#######################################################################

colours = ['b','g','r','c','m','y','k']
complexEigen = []
complexEigen2 = []
modEigen = []
modEigenNorm = []
for i in range(nM):
	#complexEigen.append(getImag(eigenVecStr[0][i].astype(np.float), plot = False))
	complexEigen2.append(Tomas(eigenVecStr[0][i].astype(np.float)))
	#modEigen.append(np.absolute(complexEigen[i]))
	#modEigenNorm.append(modEigen[i]/sum(modEigen[i]))


complexEigen3 = np.zeros(shape=(numIter,nM))
#complexEigen3 = [[0 for x in range(nM)] for y in range(numIter)] 
for j in range(numIter):
	for i in range(nM):
		complexEigen3[j][i] = Tomas(eigenVecStr[j][i].astype(np.float))

print(complexEigen3)
print('space')
print(complexEigen3[0])


'''
for i in range(4):
	#displayState(stateEnList[0], i, colours[i])
	#displayState(modEigenNorm, dipoleOrder[i], colours[i])
	#displayState(np.absolute(complexEigen), dipoleOrder[i], 'r')
	#displayState(np.absolute(complexEigen2), dipoleOrder[i], colours[i])
	displayState(np.absolute(complexEigen2), i, colours[i])
	
plt.show()

#plt.plot(eigenVecStr[0][0].astype(np.float), 'k--')

for i in range(4):
	colour = colours[i] + '--'
	plt.plot(complexEigen2[i].real, 'k--')
	plt.plot(complexEigen2[i].imag, 'm--')
	plt.plot(np.absolute(complexEigen2[i]), colour)
	plt.show()


plt.plot(stateDipoleList[0], 'y--')
plt.show()


def IPR(vec):

	I2r = []
	I1r = []
	I2r_sum = []

	I2r[Ne-1] = np.sum(np.abs(vec)**4)
	I1r[Ne-1] = np.sum(np.abs(vec)**2)


	print(I1r, 1.0/I2r)

	I2r_sum[Ne-1] += 1.0/I2r[Ne-1]


	#print()


for i in range (4):
	print(colours[i])
	#complexIPR = np.sum(np.abs(complexEigen2[i])**2)
	#realIPR = np.sum(np.abs(complexEigen2[i].real)**2)
	print('complexIPR')
	
	IPR(complexEigen2[i])
	print('realIPR')
	IPR(complexEigen2[i].real)
'''


#for i in range(4):
#	#displayState(stateEnList[0], i, colours[i])
#	displayState(modEigenNorm, dipoleOrder[i], colours[i], plot = True)
#plt.show()




#plt.plot(modEigenNorm[dipoleOrder[0]].real)
#plt.plot(modEigenNorm[dipoleOrder[0]].imag)
#plt.plot(modEigenNorm[dipoleOrder[0]])
#plt.show()



#listTest = []
#listFinal = []
#for i in range(4):
#	listTest.append(eigenVecStr[0][dipoleOrder[i]].astype(np.float))
#	listFinal.append(Tomas(listTest[i]))





#cosThing = []
#for i in range(nM):
#	cosThing.append(np.cos(((2*math.pi)/length)*i))
#vecCosImag = getImag(cosThing, show = True)

#print('state -', dipoleOrder[0])
#finalEigen = getImag(eigenVecStr[0][dipoleOrder[0]].astype(np.float))
#print('state -', dipoleOrder[-1])
#finalEigen = getImag(eigenVecStr[0][dipoleOrder[-1]].astype(np.float))




#######################################################################
#																	  #
# 					Calculation of the TwoD 						  #
#																	  #
#######################################################################
'''
a_0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
a_90 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
lab_para = lab_settings(lab_settings.FOUR_WAVE_MIXING)
lab_para.set_laser_polarizations(a_0,a_0,a_0,a_0)
lab_perp = lab_settings(lab_settings.FOUR_WAVE_MIXING)
lab_perp.set_laser_polarizations(a_0,a_0,a_90,a_90)

#######################################################################
# the function
#######################################################################

def calcTwoD(loopNum):

	t1 = time.time()

	container = []
	containerPara = []
	containerPerp = []

	agg = Aggregate(molecules=forAggregate)
	for j in range(numMol):
		with energy_units('1/cm'):
			agg.monomers[j].set_energy(1, random.gauss(energies[j], staticDis))
	agg.set_coupling_by_dipole_dipole(epsr=1.21)
	agg.build(mult=2)

	H = agg.get_Hamiltonian()
	with energy_units("1/cm"):
		print(H)

	tcalc_para = TwoDResponseCalculator(t1axis=t13, t2axis=t2s, t3axis=t13, system=agg)
	tcalc_para.bootstrap(rwa, verbose=True, lab=lab_para)
	twods_para = tcalc_para.calculate()

	tcalc_perp = TwoDResponseCalculator(t1axis=t13, t2axis=t2s, t3axis=t13, system=agg)
	tcalc_perp.bootstrap(rwa, verbose=True, lab=lab_perp)
	twods_perp = tcalc_perp.calculate()

	with energy_units("1/cm"):
		for tt2 in t2s.data:
			sp_para = twods_para.get_spectrum(tt2)
			container.append(sp_para)

	with energy_units("1/cm"):
		for tt2 in t2s.data:
			sp_perp = twods_perp.get_spectrum(tt2)
			container.append(sp_perp)

	t2 = time.time()
	print('loop ' + str(loopNum) + 'completed in ' + str(t2-t1) + 's')
	print('approx ' + str(((repeatN/coreN)-1)*(t2-t1)) + 's left')

	return container


grandContainer = []
pool = mp.Pool(processes=coreN)
grandContainer = pool.map(calcTwoD, [k for k in range(repeatN)])
pool.close()
pool.join()

print(grandContainer)

#######################################################################
# Creation of the spectra
#######################################################################

spectraPara = []
spectraPerp = []
with energy_units("1/cm"):
	for i, tt2 in enumerate(t2s.data):
		specPara = TwoDSpectrum()
		specPerp = TwoDSpectrum()
		
		specPara.set_axis_1(grandContainer[0][i].xaxis)
		specPara.set_axis_3(grandContainer[0][i].yaxis)
		specPerp.set_axis_1(grandContainer[0][i+t2Count].xaxis)
		specPerp.set_axis_3(grandContainer[0][i+t2Count].yaxis)

		for j in range(len(grandContainer)):
			specPara.add_data(grandContainer[j][i].reph2D, dtype="Reph")
			specPara.add_data(grandContainer[j][i].nonr2D, dtype="Nonr")
			specPerp.add_data(grandContainer[j][i+t2Count].reph2D, dtype="Reph")
			specPerp.add_data(grandContainer[j][i+t2Count].nonr2D, dtype="Nonr")
		specPara.devide_by(len(grandContainer))
		specPerp.devide_by(len(grandContainer))

		spectraPara.append(specPara)
		spectraPerp.append(specPerp)

aPara = ['square', 11000, 13800, 11000, 13800]
aPerp = ['square', 11000, 13800, 11000, 13800]

pointPara1 = []
pointPerp1 = []
pointPara2 = []
pointPerp2 = []
areaPara = []
maxPara = []
areaPerp = []
maxPerp = []
with energy_units("1/cm"):
	for i, tt2 in enumerate(t2s.data):

		pointPara1.append(round(spectraPara[i].get_value_at(12620, 12600),1))
		pointPara2.append(round(spectraPara[i].get_value_at(12490, 12510),1))
		maxPara.append(round(spectraPara[i].get_max_value(),1))
		areaPara.append(spectraPara[i].get_area_integral(aPara))

		pointPerp1.append(round(spectraPerp[i].get_value_at(12620, 12600),1))
		pointPerp2.append(round(spectraPerp[i].get_value_at(12490, 12510),1))
		maxPerp.append(round(spectraPerp[i].get_max_value(),1))
		areaPerp.append(spectraPerp[i].get_area_integral(aPara))

with energy_units("1/cm"):
	for i, tt2 in enumerate(t2s.data):
		spectraPara[i].plot()
		plt.xlim(11000, 14000)
		plt.ylim(11000, 14000)
		plt.show()
with energy_units("1/cm"):
	for i, tt2 in enumerate(t2s.data):
		spectraPerp[i].plot()
		plt.xlim(11000, 14000)
		plt.ylim(11000, 14000)
		plt.show()
with energy_units("1/cm"):
	for i, tt2 in enumerate(t2s.data):
		spectraPara[i].plot()
		plt.xlim(11000, 14000)
		plt.ylim(11000, 14000)
		plt.savefig('para' + str(tt2) + '.png')
with energy_units("1/cm"):
	for i, tt2 in enumerate(t2s.data):
		spectraPerp[i].plot()
		plt.xlim(11000, 14000)
		plt.ylim(11000, 14000)
		plt.savefig('perp' + str(tt2) + '.png')

print('Max Para - ', maxPara)
print('area Para - ', areaPara)
print('Max Perp - ', maxPerp)
print('area Perp - ', areaPerp, '\n')

anisMax = []
anisPoint1 = []
anisPoint2 = []
for i in range(t2Count):
	#anis.append((areaPara[i] - areaPerp[i])/(areaPara[i] + (2 * areaPerp[i])))
	anisMax.append((maxPara[i] - maxPerp[i])/(maxPara[i] + (2 * maxPerp[i])))
	anisPoint1.append((pointPara1[i] - pointPerp1[i])/(pointPara1[i] + (2 * pointPerp1[i])))
	anisPoint2.append((pointPara2[i] - pointPerp2[i])/(pointPara2[i] + (2 * pointPerp2[i])))

print('Max anis - ', anisMax)
print('point 1 anis - ', anisPoint1)
print('point 2 anis - ', anisPoint2)
print('number of pigmetns - ', len(forAggregate))

'''