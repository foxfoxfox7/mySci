import pickle
import copy
import math
import matplotlib.pyplot as plt
import numpy as np

import quantarhei as qr

import sys
sys.path.append('/home/kieran/Work/mySci')
import myFuncs




def displayState(func, order):

    plt.plot(func[order].imag, 'b--')
    plt.plot(func[order].real, 'k--')
    plt.plot(np.absolute(func[order].real)+ np.absolute(func[order].imag), 'r--')
    plt.show()



save_dir = './state_data/'
#file_name = 'N32_D5_SD250_L500'
file_name = 'LH1_SD50_L10000'

with open(save_dir + file_name +'.pkl', 'rb') as f:
    state_dict = pickle.load(f)

print(state_dict.keys())

num_loop = len(state_dict['pig_ens'])
num_mol = len(state_dict['pig_ens'][0])

print('ens')
print(state_dict['pig_ens'][0])

print('dips')
print(state_dict['dips'][0])


complex_eigen = []
for l in range(num_loop):
    complex_eigen_mol = []
    for n in range(num_mol):
        complex_eigen_mol.append(myFuncs.complexify(state_dict['eig_vecs'][l][n].astype(np.float)))
    complex_eigen.append(complex_eigen_mol)

ipr_list = np.zeros(num_mol)
ipr_list_real = np.zeros(num_mol)
for l in range(num_loop):
    for state in range(num_mol):
        eigen_ipr, eigen_ipr_real = myFuncs.calc_ipr(complex_eigen[l][state])
        ipr_list[state] += eigen_ipr
        ipr_list_real[state] += eigen_ipr_real
ipr_list = ipr_list / num_loop
ipr_list_real = ipr_list_real / num_loop

dip_av = np.zeros(num_mol)
for l in range(num_loop):
    dip_av += state_dict['dips'][l]
dip_av = dip_av / num_loop

dip_av_commas = ', '.join(map(str, dip_av)) 
print('dips av')
print(dip_av_commas)

ipr_list_commas = ', '.join(map(str, ipr_list)) 
print('ipr')
print(ipr_list_commas)


#for n in range(num_mol):
#    displayState(complex_eigen[0], n)


