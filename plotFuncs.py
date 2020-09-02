import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import quantarhei as qr


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
		plt.plot(xn, yn, 'ko', label="Original Noised Data")
		plt.plot(xn, func(xn, *popt), 'r-', label="Fitted Curve")
		plt.legend()
		plt.show()

	return popt

def plot_depo(data, labels, timestep, title=None, show=True, save=False):

	x = []
	for i in range(len(data[0])):
		x.append(i*timestep)

	fig, ax = plt.subplots()
	for i, data_set in enumerate(data):
		ax.plot(x, data_set, label=labels[i])
	ax.set_title(title)
	leg = ax.legend()
	if save:
		plt.savefig(title+'.png')
	if show:
		plt.show()

def plot_exponentials2(data, x, timestep, title=None, show=True, save=False):

	exp_param = []
	for curve in data:
		fit_params = get_exp_fit(curve, timestep)
		exp_param.append(fit_params[1])

	plt.plot(x, exp_param)
	if save:
		plt.savefig(title+'.png')
	if show:
		plt.show()

def plot_exponentials(data, x, timestep, xtitle = None):

    exp_param = []
    exp_param2 = []
    for curve in data:
        fit_params = get_exp_fit(curve, timestep)
        exp_param.append(1/fit_params[1])
        exp_param2.append(fit_params[0]+fit_params[2])

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,4), dpi= 80)
    fig.suptitle("The fitting parameters of the depolarization curves", fontsize=14)
    
    ax[0].plot(x, exp_param)
    #ax[0].set_title('how quickly the deplarization decays')
    ax[0].set_ylabel('lifetime (fs)')
    ax[0].set_xlabel(xtitle)
    
    ax[1].plot(x, exp_param2)
    #ax[1].set_title('the start position of the anisotropy')
    ax[1].set_ylabel('start position')
    ax[1].set_xlabel(xtitle)
    
    plt.show()

def plot_lists(data, labels, title=None, show=True, save=False):

	fig, ax = plt.subplots()
	for i, data_set in enumerate(data):
		ax.plot(data_set, label=labels[i])
	ax.set_title(title)
	leg = ax.legend()

	if save:
		plt.savefig(title+'.png')
	if show:
		plt.show()

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

def get_list_of_spec(fileRoot, cutOff = 200):

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

def get_spectrum_from_data2(xax, yax, data, timestamp):

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

    onetwod = qr.TwoDSpectrum()
    onetwod.set_axis_1(x_ax)
    onetwod.set_axis_3(y_ax)
    onetwod.set_data(data)
    onetwod.set_t2(timestamp)

    return onetwod

