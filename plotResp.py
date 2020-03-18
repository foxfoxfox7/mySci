import matplotlib.pyplot as plt
import numpy as np

import quantarhei as qr


#
# The responses
#
directory = 'exampleTrimerSep/paraResp'

#time = np.loadtxt('./' + directory + '/timeData.txt')
#respN = np.loadtxt('./' + directory + '/respN_t0.txt').view(complex)
#respR = np.loadtxt('./' + directory + '/respR_t0.txt').view(complex)

#time = np.load('./' + directory + '/timeData.npy')
#respN = np.load('./' + directory + '/respN_t0.npy')
#respR = np.load('./' + directory + '/respR_t0.npy')

data = np.load('./' + directory + '/respT0Pad.npz')

print(data.files)

plt.plot(data['time'], data['nonr'], 'r--')
plt.show()
plt.plot(data['time'], data['reph'], 'b--')
plt.show()

#
# The setup
#
rwa = 2.354
en1 = 11500
en2 = 13500

t13Step = 1
t13Length = 1000
t13Count = int(t13Length/t13Step)+1
t13 = qr.TimeAxis(0.0, t13Count, t13Step)
t13.atype = 'complete'
t13Freq = t13.get_FrequencyAxis()
t13Freq.data += rwa
t13Freq.start += rwa

#
# Separate pathways spectrum
#
ftresp = np.fft.fft(data['reph'],axis=1)
ftresp = np.fft.ifft(ftresp,axis=0)
reph2D = np.fft.fftshift(ftresp)

ftresp = np.fft.ifft(data['nonr'],axis=1)
ftresp = np.fft.ifft(ftresp,axis=0)*ftresp.shape[1]
nonr2D = np.fft.fftshift(ftresp)

onetwod = qr.TwoDResponse()
onetwod.set_axis_1(t13Freq)
onetwod.set_axis_3(t13Freq)
onetwod.set_resolution("signals")
onetwod._add_data(reph2D, dtype=qr.signal_REPH)
onetwod._add_data(nonr2D, dtype=qr.signal_NONR)
spectrum1 = onetwod.get_TwoDSpectrum()

with qr.energy_units('1/cm'):
	spectrum1.plot()
	plt.xlim(en1, en2)
	plt.ylim(en1, en2)
	plt.show()

#
# Printout
#
print('count - ', t13Count)
print('data len - ', len(data['nonr']))
print('respN - ', data['nonr'][67][5])
print('respR - ', data['reph'][67][5])

# Editted
'''
respNtest = respN
respRtest = respR

respNtest[300:][:] = 0
respRtest[300:][:] = 0
respNtest[:][300:] = 0
respRtest[:][300:] = 0

plt.plot(time, respNtest, 'r--')
plt.show()
plt.plot(time, respRtest, 'b--')
plt.show()

ftresp = np.fft.fft(respRtest,axis=1)
ftresp = np.fft.ifft(ftresp,axis=0)
reph2D = np.fft.fftshift(ftresp)

ftresp = np.fft.ifft(respNtest,axis=1)
ftresp = np.fft.ifft(ftresp,axis=0)*ftresp.shape[1]
nonr2D = np.fft.fftshift(ftresp)

onetwodtest = qr.TwoDResponse()
onetwodtest.set_axis_1(t13)
onetwodtest.set_axis_3(t13)
onetwodtest.set_resolution("signals")
onetwodtest._add_data(reph2D, dtype=qr.signal_REPH)
onetwodtest._add_data(nonr2D, dtype=qr.signal_NONR)

spectrumTest = onetwodtest.get_TwoDSpectrum()

print('Test max - ', spectrumTest.get_max_value())

spectrumBase.plot()
plt.xlim(490, 510)
plt.ylim(490, 510)
plt.show()
'''