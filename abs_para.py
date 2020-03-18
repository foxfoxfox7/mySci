# -*- coding: utf-8 -*-
import os
import time
import random

import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp

from quantarhei import Aggregate
from quantarhei import energy_units
from quantarhei import CorrelationFunction
from quantarhei import TimeAxis

from quantarhei.spectroscopy.absbase import AbsSpectrumBase
from quantarhei.builders.pdb import PDBFile
from quantarhei.core.units import convert
from quantarhei.models.bacteriochlorophylls import BacterioChlorophyll
from quantarhei.models.spectdens import SpectralDensityDB

import quantarhei as qr

#***********************************************************************
#*                                                                     *
#*                    Parameters and abs function                      *
#*                                                                     *
#***********************************************************************

t1 = time.time()

_save_ = False
_show_plots_ = True

# Parameters for the correlation function
time_length = 500
time_step = 1.0
temperature = 300.0

# Static disorder parameters
absSpecCount = 40
gWidth875 = 400 #400
gWidth800 = 300
gWidth850 = 300

# Parallel parameters ((mp.cpu_count())
proPower = 4


def abs_calculate(loopNum = 0, prop = True):
    # Goes through the list of molecules in agg and assigns them a new energy
    if pdb_name == 'LH2':
        for j, mol in enumerate(agg.monomers):
            with energy_units("1/cm"):
                if (j+1)%3 == 0:
                    mol.set_energy(1, random.gauss(energies[j], gWidth800))
                else:
                    mol.set_energy(1, random.gauss(energies[j], gWidth850))
    else:
        for j, mol in enumerate(agg.monomers):
            with energy_units("1/cm"):
                mol.set_energy(1, random.gauss(energies[j], gWidth875))
        

    # rebuild cleans the object before building it again
    agg.rebuild()

    if prop:
        # Returning the propagator (tensor is included). stR=standard Redfield
        prop_Redfield = agg.get_ReducedDensityMatrixPropagator(timea,
                                                    relaxation_theory="stR",
                                                    time_dependent=False,
                                                    secular_relaxation=True)
        
        # setting up the abs object ready to be calculated with(out) propagater
        calc = qr.AbsSpectrumCalculator(timea, system=agg,
                              relaxation_tensor=prop_Redfield.RelaxationTensor)
    else:
        calc = qr.AbsSpectrumCalculator(timea, system=agg)

    calc.bootstrap(rwa=rwa)
    abs_calc = calc.calculate()

    return abs_calc


#***********************************************************************
#*                                                                     *
#*        Getting molecules from PDB and building input file           *
#*                                                                     *
#***********************************************************************

print("Getting molecules from PDB and building input file...\n")
# Activates the BCl finding funciton to extract molecules from PDB
bcl_model = BacterioChlorophyll(model_type="PDB")

# Exctract the molecules from pdb file
#pdb_name = input("file name? - ")
pdb_name = "LH1"
try:
    file = PDBFile(pdb_name + ".pdb")
except:
    raise ValueError("Input file name (", pdb_name, ") is not in directory")
molecules = file.get_Molecules(model=bcl_model)

# The PDB names for the pigments
bcl_names = []
for m in molecules:
    bcl_names.append(m.name)
bcl_names.sort()

# Creates the input file if one is not there to be read
if not os.path.exists(pdb_name + "_input.txt"):
    file_input = open(pdb_name + "_input.txt", "w")
    #file_input.write("name\tpigment\tenergy\n")
    for c, n in enumerate(bcl_names):
        # need to work out how to make the energy that of the default
        file_input.write(n + "\tBChl" + str(c + 1) + "\t12500.0\n")
    file_input.close()

#***********************************************************************
#*                                                                     *
#*                       Creating the Aggregate                        *
#*                                                                     *
#***********************************************************************

print("\nCreating the aggregate...\n")
# Getting the molecules and energies from the input file
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
for_aggregate = []
for name in pigment_name:
    for m in molecules:
        if m.name == name:
            m.set_name(naming_map[name])
            for_aggregate.append(m)

# Create the aggregate of Bacteriochlorophylls and a coupling matrix
agg = Aggregate(name=pdb_name, molecules=for_aggregate)
agg.set_coupling_by_dipole_dipole(epsr=1.21)

#***********************************************************************
#*                                                                     *
#*                     Setting the spectral density                    *
#*                                                                     *
#***********************************************************************

print("\nMaking the spectral density and setting it to the molecules...\n")
timea = qr.TimeAxis(0.0, time_length, time_step)
params = {"cortime":          100.0,
          "T":                temperature,
          "matsubara":        20}

db = SpectralDensityDB(verbose=False)
'''
# Using the Renger2002 paper for the base of the spec density
sd_renger = db.get_SpectralDensity(timea, "Renger_JCP_2002")
# Using Wendling2000 to incude high frequency modes
sd_wendling = db.get_SpectralDensity(timea, "Wendling_JPCB_104_2000_5825")

# Combining them to make our spectral density
ax = sd_renger.axis
sd_wendling.axis = ax
sd_tot = sd_renger + sd_wendling
'''

sd_tot = db.get_SpectralDensity(timea, "Renger_JCP_2002")

input_reorg = convert(sd_tot.get_reorganization_energy(), "int", "1/cm")
print("input_reorg -", input_reorg)
calc_reog = convert(sd_tot.measure_reorganization_energy(), "int", "1/cm")
print("calc_reog -", calc_reog)

# Obtaining the correlation function (and plotting)
with energy_units("1/cm"):
    corr_fun = sd_tot.get_CorrelationFunction(temperature)

# Assigning the correlation function to the molecules
for j, mol in enumerate(agg.monomers):
    mol.set_transition_environment((0,1), corr_fun)
    # Initially sets the energies without static disorder
    with energy_units("1/cm"):
        mol.set_energy(1, energies[j])

#***********************************************************************
#*                                                                     *
#*                    Parallel abs spec calculator                     *
#*                                                                     *
#***********************************************************************

print("\nStarting the abs calculation loop...\n")
print(time.asctime())
# Builds the aggregate with standard energies to calculate the RWA
agg.build()
rwa = agg.get_RWA_suggestion()
with energy_units("1/cm"):
    print(agg.get_Hamiltonian())
print(energies)
# Sets up object with number of processers used (mp.cpu_count() is max on comp)
pool = mp.Pool(proPower)
spectra = pool.map(abs_calculate, [i for i in range(absSpecCount)])
pool.close()
pool.join()

# Creates the final abs object and adds each calculated spec to it
abs_tot = qr.AbsSpectrum()
for i in range(absSpecCount):
    abs_tot.add_to_data(spectra[i])
# Normalise to take the average
abs_tot.normalize2()

#***********************************************************************
#*                                                                     *
#*                   Reading the measured abs spec                     *
#*                                                                     *
#***********************************************************************

print("\nReading loaded abs spectrum...\n")
data_load = np.loadtxt("meas_data.txt")
energy, seventy, room = np.hsplit(data_load, 3)
abs_load = AbsSpectrumBase()
# gives the correct interval step after conversion from nm to int
if temperature == 300:
    abs_load.set_by_interpolation(x=energy, y=room, xaxis="wavelength")
elif temperature == 77:
    abs_load.set_by_interpolation(x=energy, y=seventy, xaxis="wavelength")
else:
    raise ValueError("The temperature should be room (300) or 77K")

abs_load.normalize2()

print("calculated -", abs_tot.axis.data[np.argmax(abs_tot.data)])
print("measured -", abs_load.axis.data[np.argmax(abs_load.data)])

t2 = time.time()
print("Job took", round((t2-t1)/60, 3), "minutes")
print("or", round((t2-t1)/3600, 3), "hours to run")
print("It looped", absSpecCount, "times and used", proPower, "cores")

#***********************************************************************
#*                                                                     *
#*                       Plotting and Saving                           *
#*                                                                     *
#***********************************************************************

print("\n...and printing")

with energy_units("1/cm"):
    fig = plt.figure()
    if _show_plots_:
        plt.plot(abs_load.axis.data, abs_load.data)
        plt.plot(abs_tot.axis.data, abs_tot.data)
        #plt.xlim(1.5,3.0)
        #plt.ylim(0,1.1)
        plt.show()

if _save_:
    outFile = pdb_name.lower() + "_W"
    if pdb_name == 'LH2':
        outFile = outFile + str(gWidth800)
    else:
        outFile = outFile + str(gWidth875)
    outFile = outFile + "_E" + str(int(energies[0]))
    directory = './Abs/'
    if temperature == 77:
        directory = directory + 'Seventy/'

    fig.savefig(directory + outFile + '.png')
