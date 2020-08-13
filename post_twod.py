import pickle
import copy
import math
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('/home/kieran/Work/mySci')

import quantarhei as qr

import myFuncs


_save_ = False

def displayState(func, order):

    plt.plot(func[order].imag, 'b--')
    plt.plot(func[order].real, 'k--')
    plt.plot(np.absolute(func[order].real)+ np.absolute(func[order].imag), 'r--')
    plt.show()

########################################################################
# Loading the dictionaries
########################################################################

save_dir = './N8_D5_SD150_T200_40_500/'

las_pol = ['para', 'perp']
resp_names = ['rGSB', 'nGSB', 'rSE', 'nSE','rESA', 'nESA', 'rSEWT', 
 'nSEWT', 'rESAWT', 'nESAWT']
signal_names = ['GSB', 'SE', 'ESA', 'SEWT', 'ESAWT']
#signal_names = ['SEWT']
#signal_names = ['GSB', 'SE']

resp_containers_part, t2_dict = myFuncs.load_resp_dict(
    save_dir + 'resp',
    las_pol,
    signal_names,
    test_pad = 800
    )

#with open(save_dir + 'params' + '.pkl', 'rb') as f:
#    params_dict = pickle.load(f)
#print(params_dict.keys())

########################################################################
# Get spectra
########################################################################

para = []
perp = []
spectra = [para, perp]
en1 = 10500
en2 = 14000

for m, laser in enumerate(las_pol):
    spec_cont_p = resp_containers_part[m].get_TwoDSpectrumContainer()
    for i, tt2 in enumerate(t2_dict.data):
        twod_p = spec_cont_p.get_spectrum(tt2)
        spectra[m].append(twod_p)
        with qr.energy_units('1/cm'):
            twod_p.plot()
            plt.xlim(en1, en2)
            plt.ylim(en1, en2)
            #plt.savefig(save_dir+'spec/'+laser+str(int(tt2))+'.png')
            plt.show()


########################################################################
# Analysing the peak (spectral shift)
########################################################################

angle2 = []
widths = []
en1 = 11000
en2 = 12500
for m, laser in enumerate(las_pol):
    spec_cont_p = resp_containers_part[m].get_TwoDSpectrumContainer()
    for i, tt2 in enumerate(t2_dict.data):
        twod_p = spec_cont_p.get_spectrum(tt2)
        with qr.energy_units('1/cm'):
            
            V = myFuncs.get_cross_points(twod_p, 11600, 12080)

            angle2.append(myFuncs.get_angle_from_points(V[0], V[1]))
            #widths.append(myFuncs.get_peak_cross_sections(V, twod_p))

            #twod_p.plot()
            #plt.scatter(V[:,0], V[:,1])
            #plt.xlim(en1, en2)
            #plt.ylim(en1, en2)
            #plt.show()

print('angle')                  
print(angle2)
print('widths')
print(widths)
print(widths[:,0])
print(widths[:,1])

########################################################################
# Anisotropy (points)
########################################################################

if _save_:
    with open(save_dir + 'anis.log', 'w') as anis_file:
        anis_file.write('Time step = ' + str(t2_dict.step) + '\n')

# LEAVE LIST OF POINTS BLANK TO ONLY GET MAX ANISOTROPY
list_of_points = []
with qr.energy_units("1/cm"):
    for i in list_of_points:
        anis = []
        for j, tt2 in enumerate(t2_dict.data):
            para_val = spectra[0][j].get_value_at(i, i).real
            perp_val = spectra[1][j].get_value_at(i, i).real
            anis.append((para_val - perp_val)/(para_val + (2 * perp_val)))

        print('anis' + str(i) + ' = ' + str(anis))
        if _save_:
            with open(save_dir + 'anis.log', 'a') as anis_file:
                anis_file.write('anis' + str(i) + ' = ' + str(anis) + '\n')

anis_max = []
for j, tt2 in enumerate(t2_dict.data):
    para_val = spectra[0][j].get_max_value()
    perp_val = spectra[1][j].get_max_value()
    anis_max.append((para_val - perp_val)/(para_val + (2 * perp_val)))

print('anis_max = ' + str(anis_max) + '\n')
if _save_:
    with open(save_dir + 'anis.log', 'a') as anis_file:
        anis_file.write('anis_max = ' + str(anis_max) + '\n')

########################################################################
# Anisotropy (2d plot)
########################################################################

#with qr.energy_units('1/cm'):
#    xaxis = np.array(spectra[0][0].xaxis.data[lim1:lim2])
#    yaxis = np.array(spectra[0][0].yaxis.data[lim1:lim2])

for i, tt2 in enumerate(t2_dict.data):
    hh = np.array(spectra[0][i].data)#[lim1:lim2,lim1:lim2])
    hv = np.array(spectra[1][i].data)#[lim1:lim2,lim1:lim2])
    anis_array = (hh - hv)/(hh + (2 * hv))
    print(anis_array.shape)
    print(anis_array[500,500])

    with qr.energy_units('1/cm'):
        one_twod = qr.TwoDSpectrum()
        one_twod.set_axis_1(spectra[0][0].xaxis)
        one_twod.set_axis_3(spectra[0][0].yaxis)
        one_twod.set_t2(tt2)
        one_twod.set_data(anis_array)

        print(one_twod.get_max_value())

        one_twod.plot()
        plt.xlim(en1, en2)
        plt.ylim(en1, en2)
        plt.show()

        di_cut = one_twod.get_diagonal_cut()
        di_cut.plot()
        plt.show()







