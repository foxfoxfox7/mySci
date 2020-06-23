# -*- coding: utf-8 -*-
import os
import time
import random
import copy
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import scipy.constants as const

from quantarhei.models.bacteriochlorophylls import BacterioChlorophyll
from quantarhei.models.spectdens import SpectralDensityDB

import quantarhei as qr

import myFuncs

#***********************************************************************
#*                                                                     *
#*                    Parameters and abs function                      *
#*                                                                     *
#***********************************************************************

t1 = time.time()

_save_ = True
_show_ = True

static_dis = 450
energy = 12217
reorganisation = 102
width = 300.0
cor_time = 100.0
temperature = 300


# Number of 2d containers calculated each loop
absSpecCount = 50
# Number of cores used for each loop
n_cores = 1


def abs_calculate(loopNum = 0, prop = True):
    # Goes through the list of molecules in agg and assigns them a new energy
    with qr.energy_units("1/cm"):
        for i, mol in enumerate(for_agg):
            mol.set_energy(1, random.gauss(energy, static_dis))
    

    agg = qr.Aggregate(molecules=for_agg)
    agg.set_coupling_by_dipole_dipole(epsr=1.21)
    agg.build(mult=1)
    KK = agg.get_RedfieldRateMatrix()

    HH = agg.get_Hamiltonian()
    pig_ens = np.diagonal(HH.data[1:num_mol+1,1:num_mol+1])\
     / (2.0*const.pi*const.c*1.0e-13)

    calc = qr.AbsSpectrumCalculator(timea, system=agg, rate_matrix=KK)
    calc.bootstrap(rwa=rwa)
    abs_calc = calc.calculate()

    t4 = time.time()
    print('time taken ', (t4-t3))
    print('loops done ', loopNum)

    return abs_calc


length = 1000
timea = qr.TimeAxis(0.0, length, 1)

for_agg = myFuncs.bacteriochl_agg('LH1')
#for_agg = myFuncs.bacteriochl_agg('3eoj')
num_mol = len(for_agg)

# Parameters for spectral density. ODBO, Renger or Silbey
params = {#"ftype": "OverdampedBrownian",
            "ftype": "B777",
            "alternative_form": True,
            "reorg": 30,
            "T":temperature,
            "cortime":cor_time}

myFuncs.set_cor_func(for_agg, params, ax_len = length)
rwa = qr.convert(energy, '1/cm', 'int')

#***********************************************************************
#*                                                                     *
#*                    Parallel abs spec calculator                     *
#*                                                                     *
#***********************************************************************

print("\nStarting the abs calculation loop...\n")
print(time.asctime())
t3 = time.time()
# Sets up object with number of processers used (mp.cpu_count() is max on comp)
pool = mp.Pool(n_cores)
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
data_load = np.loadtxt("LH1_exp_abs.txt")
energy, seventy, room = np.hsplit(data_load, 3)
abs_load = qr.AbsSpectrum()
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
print("It looped", absSpecCount, "times and used", n_cores, "cores")

#***********************************************************************
#*                                                                     *
#*                       Plotting and Saving                           *
#*                                                                     *
#***********************************************************************

print("\n...and printing")

with qr.energy_units("1/cm"):
    plt.plot(abs_load.axis.data, abs_load.data)
    plt.plot(abs_tot.axis.data, abs_tot.data)
    if _save_:
        plt.savefig('./abs_spec/LH1abs.png')
    if _show_:
        plt.show()

with qr.energy_units("1/cm"):
    plt.plot(abs_load.axis.data, abs_load.data)
    plt.plot(abs_tot.axis.data, abs_tot.data)
    plt.xlim(10000,15000)
    if _save_:
        plt.savefig('./abs_spec/LH1abs_narrow.png')
    if _show_:
        plt.show()

