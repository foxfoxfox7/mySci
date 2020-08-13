import pickle
import copy
import math
import matplotlib.pyplot as plt
import numpy as np

import quantarhei as qr

import sys
sys.path.append('/home/kieran/Work/mySci')
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

save_dir = './test_new/'
file_name = 'resp'

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
            widths.append(myFuncs.get_peak_cross_sections(V, twod_p))

            #twod_p.plot()
            #plt.scatter(V[:,0], V[:,1])
            #plt.xlim(en1, en2)
            #plt.ylim(en1, en2)
            #plt.show()

print('angle')                  
print(angle2)
print('widths')
print(widths)

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
'''
data_for_anis = [[],[]]
x_for_anis = [[],[]]
y_for_anis = [[],[]]


for m, laser in enumerate(las_pol):
    spec_cont_p = resp_containers_part[m].get_TwoDSpectrumContainer()
    for i, tt2 in enumerate(t2_dict.data):
        twod_p = spec_cont_p.get_spectrum(tt2)
        with qr.energy_units('1/cm'):

            data_for_anis[m].append(twod_p.data)
            x_for_anis[m].append(twod_p.xaxis.data)
            y_for_anis[m].append(twod_p.yaxis.data)

data_para = np.array(data_for_anis[0])
data_perp = np.array(data_for_anis[1])
data_para = data_para.real
data_perp = data_perp.real

print('para')
print(data_para)
print('perp')
print(data_perp)


anis_data = (data_para - data_perp)/(data_para + (2 * data_perp))
print('anis')
print(anis_data[0])
print(x_for_anis[0])

fig, ax = plt.subplots(1,1)
cp = ax.contourf(x_for_anis[0][0], y_for_anis[0][0], anis_data[0])
plt.xlim(10000,14000)
plt.ylim(10000,14000)
fig.colorbar(cp) # Add a colorbar to a plot
#ax.set_title('Filled Contours Plot')
#ax.set_xlabel('x (cm)')
#ax.set_ylabel('y (cm)')
plt.show()
'''
