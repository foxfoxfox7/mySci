import math
import matplotlib.pyplot as plt
import numpy as np

import quantarhei as qr

from scipy.optimize import curve_fit

import sys
sys.path.append('/home/kieran/Work/mySci')
import myFuncs


def _extract2dSpectra(fileName, timeStamp, countNum):
	
	file1 = fileName + timeStamp + '_fs.txt'
	file2 = fileName + timeStamp + 'fs.txt'

	try:
		with open(file1) as file:
		    array2d = [[float(digit) for digit in line.split()] for line in file]
		array2d = np.array(array2d)
		#print(countNum, ' - ', file1)
	except:
		with open(file2) as file:
		    array2d = [[float(digit) for digit in line.split()] for line in file]
		array2d = np.array(array2d)
		#print(countNum, ' - ', file2)

	X = []
	Y = []
	for i in range(len(array2d[0])):
		X.append(array2d[0][i])		
	for j in range(len(array2d)):
		Y.append(array2d[j][0])

	del X[0]
	del Y[0]
	array2d = np.delete(array2d, (0), axis=0)
	array2d = np.delete(array2d, (0), axis=1)

	return array2d, X, Y

def get_list_of_spec(fileRoot):

	xList = []
	yList = []
	dataList = []
	timeStamps = []

	timeGap = 0
	time = -1
	end = False
	count = 0
	while not end and time < cutOff:
		try:
			time = time + 1
			timeString = str(time)
			data, xAxis, yAxis = _extract2dSpectra(fileRoot, timeString, count)
			xList.append(xAxis)
			yList.append(yAxis)
			dataList.append(data)
			timeStamps.append(time)
			count = count+1
			timeInc = 0
		except:
			timeInc = timeInc + 1
			if timeInc > 200:
				end = True
				print('No more timestamps')

	return xList, yList, dataList, timeStamps, count

def get_spectrum_from_data(xax, yax, data, timestamp):

	x_ax_n = len(xax)
	x_ax_start = xax[0]
	x_ax_fin = xax[-1]
	x_ax_len = x_ax_fin - x_ax_start
	x_ax_step = x_ax_len / x_ax_n
	x_ax = qr.FrequencyAxis(x_ax_start, x_ax_n, x_ax_step)

	y_ax_n = len(yax)
	y_ax_start = yax[0]
	y_ax_fin = yax[-1]
	y_ax_len = y_ax_fin - y_ax_start
	y_ax_step = y_ax_len / y_ax_n
	y_ax = qr.FrequencyAxis(y_ax_start, y_ax_n, y_ax_step)

	onetwod = qr.TwoDResponse()
	onetwod.set_axis_1(x_ax)
	onetwod.set_axis_3(y_ax)
	onetwod._add_data(data, resolution="off")
	onetwod.set_t2(timestamp)

	return onetwod

def get_exp_fit(y, timestep, title='dummy', show=False):

	x = []
	for i in range(len(y)):
		x.append(i*timestep)

	xn = np.array(x, dtype=float)
	yn = np.array(y, dtype=float)

	def func(xn, a, b, c):
	    return a * np.exp(-b * xn) + c

	popt, pcov = curve_fit(func, xn, yn, p0 = (xn[0], 0.01, xn[-1]))#, 

	if show:
		plt.figure()
		plt.plot(xn, yn, 'ko', label="Original Data")
		plt.plot(xn, func(xn, *popt), 'r-', label="Fitted Curve")
		plt.legend()
		plt.show()

	return popt


# LH1/LH1_2D_Data/Non-rephasing/Imag/Non-Rephasing_Imaginary_
# LH1/LH1_2D_Data/Non-rephasing/Real/Non-Rephasing_Real_
# LH1/LH1_2D_Data/Rephasing/Imag/Rephasing_Imaginary_
# LH1/LH1_2D_Data/Rephasing/Real/Rephasing_Real_
# LH1/LH1_2D_Data/Total/Imag/Total_Imaginary_
####### LH1/LH1_2D_Data/Total/Real/Total_Real_
# LH1/LH1_2D_RephPol/HV/Imag/2D_Rephasing_Imaginary_
# LH1/LH1_2D_RephPol/HV/real/2D_Rephasing_Real_
# LH1/LH1_2D_RephPol/VV/Imag/2D_Rephasing_Imaginary_
# LH1/LH1_2D_RephPol/VV/real/2D_Rephasing_Real_
####### LH1/Total_Real_MA_Population_Time/2D_Total_Real_

# LH2/LH2_2D_Pola/Ascii/HV_real_interpolated/2D_Total_Real_
# LH2/LH2_2D_Pola/Ascii/VV_real_interpolated/2D_Total_Real_200_

#####  to_kier/TotReal_VV_shorttime/2D_Total_Real_
# to_kier/RephReal_VV_shorttime/2D_Rephasing_Real_

fileRoot = 'to_kier/TotReal_VV_shorttime/2D_Total_Real_'

fileRoot_vv = 'to_kier/TotReal_VV_shorttime/2D_Total_Real_'
fileRoot_hv = 'to_kier/TotReal_HV_shorttime/2D_Total_Real_'

cutOff = 800

showSet = False
showSingle = False
slide = 200



xList_vv, yList_vv, dataList_vv, timeStamps_vv, count_vv = get_list_of_spec(fileRoot_vv)
xList_hv, yList_hv, dataList_hv, timeStamps_hv, count_hv = get_list_of_spec(fileRoot_hv)

print('Number of time steps - ',  count_vv)
print('Time steps are - ', timeStamps_vv)

print('x - ', len(xList_vv[0]))
print('y - ', len(yList_vv[0]))

twod_vv = []
twod_hv = []
angles = []
for i in range(count_vv):
	onetwod_vv = get_spectrum_from_data(xList_vv[i], yList_vv[i], dataList_vv[i], timeStamps_vv[i])
	onetwod_hv = get_spectrum_from_data(xList_hv[i], yList_hv[i], dataList_hv[i], timeStamps_hv[i])

	twod_vv.append(onetwod_vv)
	twod_hv.append(onetwod_hv)
	#onetwod_vv.plot(window = [10.5, 12.0, 10.5, 12.0])#
	#plt.title(str(timeStamps_vv[i]) + 'fs')
	#plt.show()

	angles.append(myFuncs.get_peak_angle(onetwod_vv, 10.9, 11.6))

anis_max = []
for i in range(count_vv):
    para_val = twod_vv[i].get_max_value()
    perp_val = twod_hv[i].get_max_value()
    anis_max.append((para_val - perp_val)/(para_val + (2 * perp_val)))

plt.plot(timeStamps_vv, angles)
plt.title('angles')
plt.show()

plt.plot(timeStamps_vv, anis_max)
plt.title('anisotropy')
plt.show()

params = get_exp_fit(anis_max, timestep=10.0, show=True)

print(params)
print(1/params[1])

'''

xList = []
yList = []
dataList = []
timeStamps = []

timeGap = 0
time = -1
end = False
count = 0
while not end and time < cutOff:
	try:
		time = time + 1
		timeString = str(time)
		data, xAxis, yAxis = _extract2dSpectra(fileRoot, timeString, count)
		xList.append(xAxis)
		yList.append(yAxis)
		dataList.append(data)
		timeStamps.append(time)
		count = count+1
		timeInc = 0
	except:
		timeInc = timeInc + 1
		if timeInc > 200:
			end = True
			print('No more timestamps')



for i in range(count):
	x_ax_n = len(xList[i])
	x_ax_start = xList[i][0]
	x_ax_fin = xList[i][-1]
	x_ax_len = x_ax_fin - x_ax_start
	x_ax_step = x_ax_len / x_ax_n
	x_ax = qr.FrequencyAxis(x_ax_start, x_ax_n, x_ax_step)

	y_ax_n = len(yList[i])
	y_ax_start = yList[i][0]
	y_ax_fin = yList[i][-1]
	y_ax_len = y_ax_fin - y_ax_start
	y_ax_step = y_ax_len / y_ax_n
	y_ax = qr.FrequencyAxis(y_ax_start, y_ax_n, y_ax_step)

	onetwod = qr.TwoDResponse()
	onetwod.set_axis_1(x_ax)
	onetwod.set_axis_3(y_ax)
	onetwod._add_data(dataList[i], resolution="off")
	onetwod.set_t2(timeStamps[i])
	#with qr.energy_units('1/cm'):
	onetwod.plot(window = [10.5, 12.0, 10.5, 12.0])#
	plt.title(str(timeStamps[i]) + 'fs')
	#plt.savefig('LH1_t' + str(int(timeStamps[i]))+'.png')
	plt.show()

	print(myFuncs.get_peak_angle(onetwod, 11.0, 11.3))
'''

cmap = plt.cm.rainbow
if showSet:
	for i in range(count-1):
		cs = plt.contourf(xList[i], yList[i], dataList[i])
		#cs = plt.imshow( dataList[i], origin='lower',
        #           interpolation='bilinear', cmap=cmap)
		plt.title(str(timeStamps[i]) + 'fs')
		#plt.savefig(str(timeStamps[i]) + 'fs.png')
		plt.show()

if showSingle:
	print('Showing spectra number - ', slide)
	cs = plt.contourf(xList[slide], yList[slide], dataList[slide])
	plt.title(str(timeStamps[slide]) + 'fs')
	#plt.savefig(str(timeStamps[slide]) + 'fs.png')
	plt.show()
