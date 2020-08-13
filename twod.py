#!/usr/bin/env python

import os
import math
import time
import random
import copy
import pickle
import tracemalloc
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import scipy.constants as const

import quantarhei as qr
from quantarhei.models.spectdens import SpectralDensityDB
import aceto
from aceto.lab_settings import lab_settings

import sys
sys.path.append('/home/kieran/Work/mySci')
import myFuncs


_forster_ = False
_LH1_ = False

save_dir = './test_new/'

num_mol = 2
dipole_strength = 5
static_dis = 50
energy = 12216
reorganisation = 102
cor_time = 100.0
temperature = 300

# Number of 2d containers calculated each loop
n_per_loop = 1
# Number of cores used for each loop
n_cores = 1
# Number of times this is repeated
n_loops = 1
totalSpec = int(n_per_loop*n_loops)

t13_ax_step = 1
t13_ax_len = 300
t2_ax_step = 60
t2_ax_len = 60

#######################################################################
# Setup paramaters
#######################################################################

# The axis for time gaps 1 and 3
t13_ax_count = int(t13_ax_len/t13_ax_step)+1
t13_ax = qr.TimeAxis(0.0, t13_ax_count, t13_ax_step)

# The axis for time gap 2
t2_ax_count = int(t2_ax_len/t2_ax_step)+1
t2_ax = qr.TimeAxis(0.0, t2_ax_count, t2_ax_step)

try:
    os.mkdir(save_dir)
except OSError:
    print("Creation of the directory %s failed" % save_dir)


#######################################################################
# Creation of the aggregate ring
#######################################################################

if _LH1_:
    myFuncs.show_dipoles(for_agg)
else:
    for_agg = myFuncs.circularAgg(num_mol, dipole_strength)


# Parameters for spectral density. ODBO, Renger or Silbey
params = {"ftype": "OverdampedBrownian",
            #"ftype": "B777",
            #"alternative_form": True,
            "reorg": reorganisation,
            "T":temperature,
            "cortime":cor_time}
myFuncs.set_cor_func(for_agg, params)

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
las_pol = ['para', 'perp']

def calcTwoD(n_loopsum):#
    ''' Caculates a 2d spectrum for both perpendicular and parael lasers
    using aceto bands and accurate lineshapes'''

    resp_container = []

    #energies0 = [energy - (100 * num_mol / 2) + i * 100 for i in range(num_mol)]
    #energies0 = [energy] * num_mol
    # Giving random energies to the moleucles according to a gauss dist
    with qr.energy_units("1/cm"):
        for i, mol in enumerate(for_agg):
            mol.set_energy(1, random.gauss(energy, static_dis))
            #mol.set_energy(1, energies0[i])

    agg = qr.Aggregate(molecules=for_agg)

    agg_rates = copy.deepcopy(agg)
    agg_rates.set_coupling_by_dipole_dipole(epsr=1.21)
    agg_rates.build(mult=1)

    if _forster_:
        with qr.energy_units('1/cm'):
            print(agg_rates.get_Hamiltonian())
        KK = agg_rates.get_FoersterRateMatrix()
    else:
        with qr.energy_units('1/cm'):
            print(agg_rates.get_Hamiltonian())
        KK = agg_rates.get_RedfieldRateMatrix()
    agg.set_coupling_by_dipole_dipole(epsr=1.21)
    agg.build(mult=2)
    rwa = agg.get_RWA_suggestion()

    HH = agg.get_Hamiltonian()
    pig_ens = np.diagonal(HH.data[1:num_mol+1,1:num_mol+1])\
     / (2.0*const.pi*const.c*1.0e-13)
    en_order = np.argsort(pig_ens)
    SS = HH.diagonalize()
    eig_vecs = np.transpose(SS[1:num_mol+1,1:num_mol+1])
    state_ens = np.diagonal(HH.data[1:num_mol+1,1:num_mol+1])\
     / (2.0*const.pi*const.c*1.0e-13)
    agg.diagonalize()
    dips = agg.D2[0][1:num_mol+1]
    dip_order = np.flip(np.argsort(dips))
    
    # Initialising the twod response calculator for the paralell laser
    resp_calc_temp = qr.TwoDResponseCalculator(
        t1axis = t13_ax,
        t2axis = t2_ax,
        t3axis = t13_ax,
        system = agg,
        rate_matrix = KK
        )

    # keep_resp saves the reponse int he object. write_resp writes to numpy
    # Response is calculated Converted into spectrum Stored in a container
    for i, lab in enumerate(labs):
        resp_calc = copy.deepcopy(resp_calc_temp)
        resp_calc.bootstrap(rwa, lab=lab, verbose = True,
            keep_resp = True)#write_resp = save_dir + las_pol[i] + '_resp', 
        resp_cont = resp_calc.calculate()
        resp_container.append(resp_calc.responses)

    state_data = {
        'pig_ens': pig_ens,
        'en_order': en_order,
        'eig_vecs': eig_vecs,
        'state_ens': state_ens,
        'dips': dips,
        'dip_order': dip_order,
        'rwa': rwa
        }

    return resp_container, state_data

#######################################################################
# Creation of the spectra
#######################################################################

Nr13 = t13_ax.length

resp_names = ['rGSB', 'nGSB', 'rSE', 'nSE','rESA', 'nESA', 'rSEWT', 
 'nSEWT', 'rESAWT', 'nESAWT']
resp_dict = {}
for i, resp in enumerate(resp_names):
    total_resp_list = []
    for j, lab in enumerate(labs):
        lab_list = []
        for k, tt2 in enumerate(t2_ax.data):
            lab_list.append(np.zeros((Nr13, Nr13), dtype=np.complex128, order='F'))
        total_resp_list.append(lab_list)
    resp_dict.update({resp: total_resp_list})

state_dict = {
    'pig_ens': [],
    'en_order': [],
    'eig_vecs': [],
    'state_ens': [],
    'dips': [],
    'dip_order': []
    }

for l in range(n_loops):
    tracemalloc.start()
    t3 = time.time()
    print('\nCalculating loop ' + str(l+1) + '...\n')

    # Function calculates the spectrum containers
    with mp.Pool(processes=n_cores) as pool:
        tot_cont = pool.map(calcTwoD, [k for k in range(n_per_loop)])

    for i, resp in enumerate(resp_names):
        for j, lab in enumerate(labs):
            for k, tt2 in enumerate(t2_ax.data):
                for n in range(n_per_loop):
                    resp_dict[resp][j][k] += tot_cont[n][0][j][k][resp]

    for key in state_dict:
        for i in range(n_per_loop):
            state_dict[key].append(tot_cont[i][1][key])

    global resp_time
    resp_time = tot_cont[0][0][0][0]['time']
    global rwa
    rwa = tot_cont[0][1]['rwa']

    # necessary to clear the container for the next loop
    tot_cont.clear()

    t4 = time.time()
    current, peak = tracemalloc.get_traced_memory()
    print("\n... calculated in " + str(t4-t3) + " sec")
    print(str(n_loops - l - 1) + ' loops left')
    print(f"Current memory usage is {current / 10**6}MB")
    print(f"Peak was {peak / 10**6}MB\n")
    tracemalloc.stop()

for i, resp in enumerate(resp_names):
    for j, laser in enumerate(las_pol):
        for k, tt2 in enumerate(t2_ax.data):
            resp_dict[resp][j][k] = resp_dict[resp][j][k] / (n_loops * n_per_loop) 
resp_dict.update({'rwa': rwa, 't2axis': t2_ax.data, 'time': resp_time})

params_dict = {
    'LH1': _LH1_,
    'num_mol': num_mol,
    'dipole_strength': dipole_strength,
    'static_dis': static_dis,
    'energy': energy,
    'totalSpec': totalSpec,
    't13_ax_step': t13_ax_step,
    't13_ax_len': t13_ax_len,
    't2_ax_step': t2_ax_step,
    't2_ax_len': t2_ax_len,
    'ftype': params['ftype']
    }


with open(save_dir + 'resp.pkl', 'wb') as f:
    pickle.dump(resp_dict, f, pickle.HIGHEST_PROTOCOL)
with open(save_dir + 'states.pkl', 'wb') as f:
    pickle.dump(state_dict, f, pickle.HIGHEST_PROTOCOL)
with open(save_dir + 'params.pkl', 'wb') as f:
    pickle.dump(params_dict, f, pickle.HIGHEST_PROTOCOL)
