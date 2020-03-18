import math
import numpy as np


def circularAgg(numMol, dipoleStrength):
	proteinDis = 8.7
	difference = 0.6
	r = 5
	while difference > 0.1:
		t = np.linspace(0, np.pi * 2, numMol+1)
		x = r * np.cos(t)
		y = r * np.sin(t)
		circle = np.c_[x, y]

		artificialDis = math.sqrt(((circle[0][0]-circle[1][0])**2)+((circle[0][1]-circle[1][1])**2))
		difference = abs(artificialDis-proteinDis)

		if artificialDis > proteinDis:
			r = r - 0.1
		elif artificialDis < proteinDis:
			r = r + 0.1
	circle2 = np.delete(circle, numMol, 0)

	dipoles = np.empty([numMol,2])
	mag = math.sqrt((circle[0][0]**2)+(circle[0][1]**2))
	for i in range(numMol):
		dipoles[i][0] = -circle2[i][1]
		dipoles[i][1] = circle2[i][0]
		dipoles[i][0] = dipoles[i][0] / mag
		dipoles[i][1] = dipoles[i][1] / mag
		dipoles[i][0] = dipoles[i][0] * dipoleStrength
		dipoles[i][1] = dipoles[i][1] * dipoleStrength

	print('positions\n', circle2)
	print('dipoles\n', dipoles)

	return circle2, dipoles