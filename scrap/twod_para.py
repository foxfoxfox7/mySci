#!/usr/bin/env python

import os
import math
import time
import random
import copy
import tracemalloc
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp

import quantarhei as qr
from quantarhei.models.spectdens import SpectralDensityDB
import aceto
from aceto.lab_settings import lab_settings

import myFuncs

_test_ = False
_save_ = False
_show_ = True

save_dir = './exampleTrimerSep/'

num_mol = 10
dipole_strength = 5
static_dis = 100
energy = 12500
reorganisation = 102
width = 300.0
cor_time = 100.0
temperature = 300

# Number of 2d containers calculated each loop
n_per_loop = 1
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

#######################################################################
# Setup of the calculation
#######################################################################

# Arrays for the direction on the laser plane
a_0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
a_90 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
# Aceto lab setup (for twod calculator)
lab_para = lab_settings(lab_settings.FOUR_WAVE_MIXING)
lab_para.set_laser_polarizations(a_0,a_0,a_0,a_0)
lab_perp = lab_settings(lab_settings.FOUR_WAVE_MIXING)
lab_perp.set_laser_polarizations(a_0,a_0,a_90,a_90)
labs = [lab_para, lab_perp]


def calcTwoD(n_loopsum):#
    ''' Caculates a 2d spectrum for both perpendicular and parael lasers
    using aceto bands and accurate lineshapes'''

    container = []

    #energies0 = [energy - (100 * num_mol / 2) + i * 100 for i in range(num_mol)]
    #energies0 = [energy] * num_mol
    # Giving random energies to the moleucles according to a gauss dist
    with qr.energy_units("1/cm"):
        for i, mol in enumerate(for_agg):
            mol.set_energy(1, random.gauss(energy, static_dis))
            #mol.set_energy(1, energies0[i])

    agg = qr.Aggregate(molecules=for_agg)
    agg.set_coupling_by_dipole_dipole(epsr=1.21)
    agg.build(mult=2)
    print(np.diagonal(agg.HH[1:num_mol+1,1:num_mol+1]))
    agg.diagonalize()
    rwa = agg.get_RWA_suggestion()

    print(np.diagonal(agg.HH[1:num_mol+1,1:num_mol+1]))

    # Initialising the twod response calculator for the paralell laser
    resp_calc_temp = qr.TwoDResponseCalculator(
        t1axis=t13_ax,
        t2axis=t2_ax,
        t3axis=t13_ax,
        system=agg
        )

    # Bootstrap is the place to add 0-padding to the response signal
    # printEigen=True prints eigenvalues, printResp='string' prints response
    # Response is calculated Converted into spectrum Stored in a container
    for lab in labs:
        resp_calc = copy.deepcopy(resp_calc_temp)
        resp_calc.bootstrap(rwa, pad=padding, lab=lab)
        resp_cont = resp_calc.calculate()
        spec_cont = resp_cont.get_TwoDSpectrumContainer()
        container.append(spec_cont)

    return container
    #return spec_cont_para, spec_cont_perp

def sum_twod(m, laser):
    for i, tt2 in enumerate(t2_ax.data):
        name = laser + '_' + str(int(tt2)) + '.npy'

        global twod

        # Create the spectrum for the timestep as the first in the list
        twod = tot_cont[0][m].get_spectrum(t2_ax.data[i])

        # for all the rest of the list, add it to the first spectrum
        for j in range(n_per_loop-1):
            twod.add_data(tot_cont[j+1][m].get_spectrum(t2_ax.data[i]).data)

        # If there is data already saved toa file, load it and add it
        try:
            twod_load = qr.TwoDSpectrum()
            twod_load.load_data(name)
            twod.add_data(twod_load.data)
        except:
            print('first loop')

        twod.save_data(name)

#######################################################################
# Creation of the spectra
#######################################################################

# list of laser polarities
las_pol = ['para', 'perp']

for l in range(n_loops):
    tracemalloc.start()
    t2 = time.time()
    print('\nCalculating loop ' + str(l+1) + '...\n')

    # Function calculates the spectrum containers
    with mp.Pool(processes=n_cores) as pool:
        tot_cont = pool.map(calcTwoD, [k for k in range(n_per_loop)])

    # Function sums the spectra in the conts and writes to file
    for m, laser in enumerate(las_pol):
        sum_twod(m, laser)

    # necessary to clear the container for the next loop
    tot_cont.clear()

    t2 = time.time()
    current, peak = tracemalloc.get_traced_memory()
    print("\n... calculated in " + str(t2-t1) + " sec")
    print(str(n_loops - l - 1) + ' loops left')
    print(f"Current memory usage is {current / 10**6}MB")
    print(f"Peak was {peak / 10**6}MB\n")
    tracemalloc.stop()

#######################################################################
# display
#######################################################################

para = []
perp = []
spectra = [para, perp]
en1 = 11000
en2 = 13500

for m, laser in enumerate(las_pol):
    for i, tt2 in enumerate(t2_ax.data):
        name = laser + '_' + str(int(tt2)) + '.npy'

        td_final = qr.TwoDSpectrum()
        td_final.load_data(name)
        td_final.set_data_type()
        td_final.set_axis_1(twod.xaxis)
        td_final.set_axis_3(twod.yaxis)
        td_final.devide_by(totalSpec)
        spectra[m].append(td_final)

        with qr.energy_units('1/cm'):
            td_final.plot()
            plt.xlim(en1, en2)
            plt.ylim(en1, en2)
            if _save_:
                plt.savefig(save_dir + name + '.png')
            if _show_:
                plt.show()

        os.remove(name)

#######################################################################
# Anisotropy
#######################################################################

if _save_:
    with open(save_dir + 'anis.log', 'w') as anis_file:
            anis_file.write('Time step = ' + str(t2_ax.step) + '\n')
'''
with qr.energy_units("1/cm"):
    for i in range(energy - 200, energy + 200, 50):
        anis = []
        for j, tt2 in enumerate(t2_ax.data):
            para_val = spectra[0][j].get_value_at(i, i).real
            perp_val = spectra[1][j].get_value_at(i, i).real
            anis.append((para_val - perp_val)/(para_val + (2 * perp_val)))

        print('anis' + str(i) + ' = ' + str(anis))

        if _save_:
            with open(save_dir + 'anis.log', 'a') as anis_file:
                anis_file.write('anis' + str(i) + ' = ' + str(anis) + '\n')
'''
anis_max = []
for j, tt2 in enumerate(t2_ax.data):
    para_val = spectra[0][j].get_max_value()
    perp_val = spectra[1][j].get_max_value()
    anis_max.append((para_val - perp_val)/(para_val + (2 * perp_val)))

print('\nanis_max = ' + str(anis_max) + '\n')
if _save_:
    with open(save_dir + 'anis.log', 'a') as anis_file:
        anis_file.write('anis_max = ' + str(anis_max) + '\n')
