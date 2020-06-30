import os
import random
import pickle
import numpy as np
import multiprocessing as mp
import scipy.constants as const

import quantarhei as qr

import myFuncs

num_mol = 8
dipole_strength = 5
static_dis = 0
energy = 12216
reorganisation = 102
cor_time = 100.0
temperature = 300

# Number of 2d containers calculated each loop
n_per_loop = 1
# Number of cores used for each loop
n_cores = 1
# Number of times this is repeated
n_loops = 500
totalSpec = int(n_per_loop*n_loops)

#######################################################################
# Setup paramaters
#######################################################################

save_dir = './state_data/'
file_name = 'N'+str(num_mol)+'_D'+str(dipole_strength)+'_SD'+str(static_dis)+'_L'+str(totalSpec)
#file_name = 'LH1_SD0_L50'

try:
    os.mkdir(save_dir)
except OSError:
    print("Creation of the directory %s failed" % save_dir)

#######################################################################
# Creation of the aggregate ring
#######################################################################

for_agg = myFuncs.circularAgg(num_mol, dipole_strength)
#for_agg = myFuncs.bacteriochl_agg('LH1')
#myFuncs.show_dipoles(for_agg)
nM = len(for_agg)

#######################################################################
# Spectral density
#######################################################################

# Parameters for spectral density. ODBO, Renger or Silbey
params = {#"ftype": "OverdampedBrownian",
            "ftype": "B777",
            "alternative_form": True,
            "reorg": reorganisation,
            "T":temperature,
            "cortime":cor_time}

myFuncs.set_cor_func(for_agg, params)

#######################################################################
# Setup of the calculation
#######################################################################

def calcTwoD(n_loopsum):#
    ''' Caculates a 2d spectrum for both perpendicular and parael lasers
    using aceto bands and accurate lineshapes'''

    #energies0 = [energy - (100 * nM / 2) + i * 100 for i in range(nM)]
    #energies0 = [energy] * nM
    # Giving random energies to the moleucles according to a gauss dist
    with qr.energy_units("1/cm"):
        for i, mol in enumerate(for_agg):
            mol.set_energy(1, random.gauss(energy, static_dis))
            #mol.set_energy(1, energies0[i])

    agg = qr.Aggregate(molecules=for_agg)
    agg.set_coupling_by_dipole_dipole(epsr=1.21)
    agg.build(mult=1)

    HH = agg.get_Hamiltonian()
    pig_ens = np.diagonal(HH.data[1:nM+1,1:nM+1])\
     / (2.0*const.pi*const.c*1.0e-13)
    en_order = np.argsort(pig_ens)
    SS = HH.diagonalize()
    eig_vecs = np.transpose(SS[1:nM+1,1:nM+1])
    state_ens = np.diagonal(HH.data[1:nM+1,1:nM+1])\
     / (2.0*const.pi*const.c*1.0e-13)
    agg.diagonalize()
    dips = agg.D2[0][1:nM+1]
    dip_order = np.flip(np.argsort(dips))

    state_data = {
        'pig_ens': pig_ens,
        'en_order': en_order,
        'eig_vecs': eig_vecs,
        'state_ens': state_ens,
        'dips': dips,
        'dip_order': dip_order
        }

    return state_data

#######################################################################
# Creation of the spectra
#######################################################################

state_dict = {
    'pig_ens': [],
    'en_order': [],
    'eig_vecs': [],
    'state_ens': [],
    'dips': [],
    'dip_order': []
    }

for l in range(n_loops):
    print('\nCalculating loop ' + str(l+1) + '...\n')

    # Function calculates the spectrum containers
    with mp.Pool(processes=n_cores) as pool:
        tot_cont = pool.map(calcTwoD, [k for k in range(n_per_loop)])

    for key in state_dict:
        for i in range(n_per_loop):
            state_dict[key].append(tot_cont[i][key])

    # necessary to clear the container for the next loop
    tot_cont.clear()

state_dict.update({'dip_str': dipole_strength, 'static_dis': static_dis, 'energy_point': energy})

with open(save_dir + file_name + '.pkl', 'wb') as f:
    pickle.dump(state_dict, f, pickle.HIGHEST_PROTOCOL)
