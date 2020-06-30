import pickle
import copy
import math
import matplotlib.pyplot as plt
import numpy as np

import quantarhei as qr

import myFuncs




#save_dir = './state_data/'
#file_name = 'N8_D5_SD0_L500'




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

def calc_ipr(vec):

    nM = len(vec)

    abs_vec = np.absolute(vec)
    by_four_vec = abs_vec ** 4
    sum_vec = np.sum(by_four_vec)
    ipr = 1 / sum_vec

    abs_vec_r = vec.real
    by_four_vec_r = abs_vec_r ** 4
    sum_vec_r = np.sum(by_four_vec_r)
    ipr_r = 1 / sum_vec_r

    return ipr, ipr_r

def displayState(func, order):

    plt.plot(func[order].imag, 'b--')
    plt.plot(func[order].real, 'k--')
    plt.plot(np.absolute(func[order].real)+ np.absolute(func[order].imag), 'r--')
    plt.show()

'''

with open(save_dir + file_name +'.pkl', 'rb') as f:
    state_dict = pickle.load(f)

print(state_dict.keys())

num_loop = len(state_dict['pig_ens'])
num_mol = len(state_dict['pig_ens'][0])

print(state_dict['pig_ens'][0])

complex_eigen = []
for l in range(num_loop):
    complex_eigen_mol = []
    for n in range(num_mol):
        complex_eigen_mol.append(complexify(state_dict['eig_vecs'][l][n].astype(np.float)))
    complex_eigen.append(complex_eigen_mol)

ipr_list = np.zeros(num_mol)
ipr_list_real = np.zeros(num_mol)
for l in range(num_loop):
    for state in range(num_mol):
        eigen_ipr, eigen_ipr_real = calc_ipr(complex_eigen[l][state])
        ipr_list[state] += eigen_ipr
        ipr_list_real[state] += eigen_ipr_real
ipr_list = ipr_list / num_loop
ipr_list_real = ipr_list_real / num_loop

print(ipr_list)
print(ipr_list_real)

#for n in range(num_mol):
#    displayState(complex_eigen[0], n)


'''

save_dir = './N8_D5_SD100_T140_20_L100/'
file_name = 'resp'


las_pol = ['para']#, 'perp'
resp_names = ['rGSB', 'nGSB', 'rSE', 'nSE','rESA', 'nESA', 'rSEWT', 
 'nSEWT', 'rESAWT', 'nESAWT']
signal_names = ['GSB', 'SE', 'ESA', 'SEWT', 'ESAWT']

resp_containers_part, t2_dict = myFuncs.load_resp_dict(
    save_dir + 'resp',
    las_pol,
    signal_names,
    test_pad = 1200
    )


en1 = 11000
en2 = 13500
for m, laser in enumerate(las_pol):
    spec_cont_p = resp_containers_part[m].get_TwoDSpectrumContainer()
    for i, tt2 in enumerate(t2_dict.data):
        twod_p = spec_cont_p.get_spectrum(tt2)
        with qr.energy_units('1/cm'):
            print('angle - ', myFuncs.get_peak_angle(twod_p, 11550, 11900))
            twod_p.plot()
            plt.xlim(en1, en2)
            plt.ylim(en1, en2)
            plt.show()

