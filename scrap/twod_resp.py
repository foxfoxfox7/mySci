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

import myFuncs

_test_ = False
_save_ = True
_show_ = True
_forster_ = False

save_dir = './testing/'

num_mol = 2
dipole_strength = 5
static_dis = 50
energy = 12500
reorganisation = 102
width = 300.0
cor_time = 100.0
temperature = 300

# Number of 2d containers calculated each loop
n_per_loop = 2
# Number of cores used for each loop
n_cores = 2
# Number of times this is repeated
n_loops = 2
totalSpec = int(n_per_loop*n_loops)

t13_ax_step = 1
t13_ax_len = 200
t2_ax_step = 100
t2_ax_len = 100

#######################################################################
# Setup paramaters
#######################################################################

# The axis for time gaps 1 and 3
t13_ax_count = int(t13_ax_len/t13_ax_step)+1
t13_ax = qr.TimeAxis(0.0, t13_ax_count, t13_ax_step)

# The axis for time gap 2
t2_ax_count = int(t2_ax_len/t2_ax_step)+1
t2_ax = qr.TimeAxis(0.0, t2_ax_count, t2_ax_step)

if _save_:
    try:
        os.mkdir(save_dir)
        os.mkdir(save_dir+'spec')
    except OSError:
        print("Creation of the directory %s failed" % save_dir)

t1 = time.time()
print("Calculating spectra from 0 to ", t2_ax_len, " fs\n")


#######################################################################
# Creation of the aggregate ring
#######################################################################

for_agg = myFuncs.circularAgg(num_mol, dipole_strength)
#myFuncs.show_dipoles(for_agg)

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

myFuncs.set_cor_func(for_agg, params)

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
    agg_rates.build(mult=2)
    
    if _forster_:
        KK = agg_rates.get_FoersterRateMatrix()
    else:
        KK = agg_rates.get_RedfieldRateMatrix()
        agg.set_coupling_by_dipole_dipole(epsr=1.21)
    agg.build(mult=2)
    rwa = agg.get_RWA_suggestion()

    HH = agg_rates.get_Hamiltonian()
    pig_ens = np.diagonal(HH.data[1:num_mol+1,1:num_mol+1])\
     / (2.0*const.pi*const.c*1.0e-13)
    en_order = np.argsort(pig_ens)
    SS = HH.diagonalize()
    eig_vecs = np.transpose(SS[1:num_mol+1,1:num_mol+1])
    state_ens = np.diagonal(HH.data[1:num_mol+1,1:num_mol+1])\
     / (2.0*const.pi*const.c*1.0e-13)
    agg_rates.diagonalize()
    dips = agg_rates.D2[0][1:num_mol+1]
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
        resp_calc.bootstrap(rwa, lab=lab, verbose = False,
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
            print(key)
            print(tot_cont[i][1][key])
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

if _save_:
    with open(save_dir + 'resp.pkl', 'wb') as f:
        pickle.dump(resp_dict, f, pickle.HIGHEST_PROTOCOL)

if _save_:
    with open(save_dir + 'states.pkl', 'wb') as f:
        pickle.dump(state_dict, f, pickle.HIGHEST_PROTOCOL)

test_pad = 800
signal_names = ['GSB', 'SE', 'ESA', 'SEWT', 'ESAWT']
resp_containers_part = []

t2start = int(resp_dict['t2axis'][0])
t2len = int(len(resp_dict['t2axis']))
t2step = int(resp_dict['t2axis'][1])
t2_dict = qr.TimeAxis(t2start, t2len, t2step)

t13start = resp_dict['time'][0]
t13step = resp_dict['time'][1]
t13end = resp_dict['time'][-1] + 1 + (t13step * test_pad)
t13_fr = np.arange(t13start, t13end, t13step)

t13_fr_len = len(resp_dict['time'])
for j, laser in enumerate(las_pol):
    laser_cont = qr.TwoDResponseContainer(t2_dict)
    for i, tt2 in enumerate(resp_dict['t2axis']):

        reph = np.zeros((t13_fr_len, t13_fr_len), dtype=np.complex128, order='F')
        nonr = np.zeros((t13_fr_len, t13_fr_len), dtype=np.complex128, order='F')

        for name in signal_names:
            reph += resp_dict['r'+name][j][i]
            nonr += resp_dict['n'+name][j][i]

        from scipy import signal as sig
        window = 20
        tuc = sig.tukey(window * 2, 1, sym = False)
        for k in range(len(reph)):
            reph[len(reph)-window:,k] *= tuc[window:]
            reph[k,len(reph)-window:] *= tuc[window:]
            nonr[len(nonr)-window:,k] *= tuc[window:]
            nonr[k,len(nonr)-window:] *= tuc[window:]

        reph = np.hstack((reph, np.zeros((reph.shape[0], test_pad))))
        reph = np.vstack((reph, np.zeros((test_pad, reph.shape[1]))))
        nonr = np.hstack((nonr, np.zeros((nonr.shape[0], test_pad))))
        nonr = np.vstack((nonr, np.zeros((test_pad, nonr.shape[1]))))

        onetwod_p = myFuncs.get_resp_object(
            respR = reph,
            respN = nonr,
            time_ax = t13_fr,
            rwa = resp_dict['rwa'],
            time_t2 = tt2
            )
        laser_cont.set_spectrum(onetwod_p)
    resp_containers_part.append(laser_cont)


#######################################################################
# display
#######################################################################

t2 = time.time()
print('"\n...' + str(totalSpec) +  ' done in '  + str(t2-t1) + " sec\n")

para = []
perp = []
spectra = [para, perp]
en1 = 11000
en2 = 13500

for m, laser in enumerate(las_pol):
    spec_cont_p = resp_containers_part[m].get_TwoDSpectrumContainer()
    for i, tt2 in enumerate(t2_ax.data):
        twod_p = spec_cont_p.get_spectrum(tt2)
        spectra[m].append(twod_p)
        with qr.energy_units('1/cm'):
            twod_p.plot()
            plt.xlim(en1, en2)
            plt.ylim(en1, en2)
            if _save_:
                plt.savefig(save_dir+'spec/'+laser+str(int(tt2))+'.png')
            if _show_:
                plt.show()

#######################################################################
# Anisotropy
#######################################################################

if _save_:
    with open(save_dir + 'anis.log', 'w') as anis_file:
            anis_file.write('Time step = ' + str(t2_ax.step) + '\n')

list_of_points = []
with qr.energy_units("1/cm"):
    for i in list_of_points:
        anis = []
        for j, tt2 in enumerate(t2_ax.data):
            para_val = spectra[0][j].get_value_at(i, i).real
            perp_val = spectra[1][j].get_value_at(i, i).real
            anis.append((para_val - perp_val)/(para_val + (2 * perp_val)))

        print('anis' + str(i) + ' = ' + str(anis))
        if _save_:
            with open(save_dir + 'anis.log', 'a') as anis_file:
                anis_file.write('anis' + str(i) + ' = ' + str(anis) + '\n')

anis_max = []
for j, tt2 in enumerate(t2_ax.data):
    para_val = spectra[0][j].get_max_value()
    perp_val = spectra[1][j].get_max_value()
    anis_max.append((para_val - perp_val)/(para_val + (2 * perp_val)))

print('anis_max = ' + str(anis_max) + '\n')
if _save_:
    with open(save_dir + 'anis.log', 'a') as anis_file:
        anis_file.write('anis_max = ' + str(anis_max) + '\n')
