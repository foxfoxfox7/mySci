import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


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

def plot_exponentials(data, x, timestep, title=None, show=True, save=False):

	exp_param = []
	for curve in data:
		fit_params = get_exp_fit(curve, timestep)
		exp_param.append(fit_params[1])

	plt.plot(x, exp_param)
	if save:
		plt.savefig(title+'.png')
	if show:
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