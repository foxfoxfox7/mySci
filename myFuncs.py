import math
import numpy as np

import quantarhei as qr


def circularAgg(numMol, dipoleStrength):
	proteinDis = 8.7
	difference = 0.6
	r = 5
	while difference > 0.1:
		t = np.linspace(0, np.pi * 2, numMol+1)
		x = r * np.cos(t)
		y = r * np.sin(t)
		circle = np.c_[x, y]

		artificialDis = math.sqrt(((circle[0][0] - circle[1][0])**2)\
			+ ((circle[0][1] - circle[1][1])**2))
		difference = abs(artificialDis-proteinDis)

		if artificialDis > proteinDis:
			r = r - 0.1
		elif artificialDis < proteinDis:
			r = r + 0.1
	circle2 = np.delete(circle, numMol, 0)

	dipoles = np.empty([numMol,2])
	mag = math.sqrt((circle[0][0]**2) + (circle[0][1]**2))
	for i in range(numMol):
		dipoles[i][0] = -circle2[i][1]
		dipoles[i][1] = circle2[i][0]
		dipoles[i][0] = dipoles[i][0] / mag
		dipoles[i][1] = dipoles[i][1] / mag
		dipoles[i][0] = dipoles[i][0] * dipoleStrength
		dipoles[i][1] = dipoles[i][1] * dipoleStrength

	print('positions\n', circle2)
	print('dipoles\n', dipoles)

	forAggregate = []
	for i in range(numMol):
	    molName = qr.Molecule()
	    molName.position = [circle2[i][0], circle2[i][1], 0.0]
	    molName.set_dipole(0,1,[dipoles[i][0], dipoles[i][1], 0.0])
	    forAggregate.append(molName)

	#return circle2, dipoles
	return forAggregate

def bacteriochl_agg(name):
	pdb_name = name
	file = PDBFile(pdb_name + ".pdb")

	bcl_model = BacterioChlorophyll(model_type="PDB")
	molecules = file.get_Molecules(model=bcl_model)
	# The PDB names for the pigments
	bcl_names = []
	for m in molecules:
	    bcl_names.append(m.name)
	bcl_names.sort()

	# Creates the input file if one is not there to be read
	if not os.path.exists(pdb_name + "_input.txt"):
	    file_input = open(pdb_name + "_input.txt", "w")
	    for c, n in enumerate(bcl_names):
	        file_input.write(n + "\tBChl" + str(c + 1) + "\t12500.0\n")
	    file_input.close()

	pigment_name = []
	pigment_type = []
	with open(pdb_name + "_input.txt") as file_names:
	    for line in file_names:
	        pigment_name.append(line.split('\t')[0])
	        pigment_type.append(line.split('\t')[1])
	naming_map = dict(zip(pigment_name, pigment_type))

	# Make a list of the molecules and set their molecule type
	forAggregate = []
	for name in pigment_name:
	    for m in molecules:
	        if m.name == name:
	            m.set_name(naming_map[name])
	            forAggregate.append(m)

	return forAggregate
