import math
import numpy as np
import re

import quantarhei as qr
from quantarhei.models.spectdens import SpectralDensityDB
from quantarhei.models.bacteriochlorophylls import BacterioChlorophyll




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

	print('\nA list of molecules was generated. Positions are in a '
		 'ring with ', proteinDis, ' Angstrom spacings. Dipoles are '
		 'added running along the tangent of the ring. All in the '
		 'same direction with ', dipoleStrength, ' dipoles (D)\n')

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

	print('\n A list of molecules has been extracted from a PDB file. '
		 'molecules are of BCl type and have the appropriate position '
		 'and dipole.\n')

	return forAggregate

def save_eigen_data(agg, file):
	nM = agg.nmono
	print('nM', nM)

	H = agg.get_Hamiltonian()
	SS = H.diagonalize()
	trans1 = SS[1:nM+1,1:nM+1]
	H.undiagonalize()
	hamil = agg.HH[1:nM+1,1:nM+1]
	
	with open(file, 'a') as f:
		f.write('Hamiltonian\n')
		np.savetxt(f, hamil)

		f.write('Transformation Matrix\n')
		np.savetxt(f, trans1)

	agg.diagonalize()
	diag = agg.HH[1:nM+1,1:nM+1]

	with open(file, 'a') as f:
		f.write('Diagonalized\n')
		np.savetxt(f, agg.HH[1:nM+1,1:nM+1])

		f.write('Dipoles\n')
		np.savetxt(f, agg.D2[0][1:nM+1].reshape(1, nM))

def extracting_eigen_data(file):
	startLine = _get_line(file, 'Hamiltonian')
	endLine = _get_line(file, 'Transformation')
	nM = endLine[0] - startLine[0] - 1
	numIter = len(startLine)
	print('\nnumber of iterations - ' + str(numIter))
	print('number of molecules/sites - ' + str(nM) + '\n')

	hamilStartLine = _get_line(file, 'Hamiltonian', plus = 1)
	hamilEndLine = _get_line(file, 'Hamiltonian', plus = nM+1)
	diagStartLine = _get_line(file, 'Diagonalized', plus = 1)
	diagEndLine = _get_line(file, 'Diagonalized', plus = nM+1)
	eigenStartLine = _get_line(file, 'Transformation', plus = 1)
	eigenEndLine = _get_line(file, 'Transformation', plus = nM+1)
	dipoleStartLine = _get_line(file, 'Dipoles', plus = 1)
	dipoleEndLine = _get_line(file, 'Dipoles', plus = 2)

	hamilList = []
	hamil = []
	pig_en = []

	diagList = []
	diag = []
	state_en = []

	eigenList = []
	eig_vecs =[]

	state_dipsStr = []
	state_dips = []

	for i in range(numIter):
		hamilList.append(_get_data_between(file, hamilStartLine[i], hamilEndLine[i]))
		hamil.append(np.array(hamilList[i]).reshape(int(len(hamilList[i])/nM),nM))
		pig_en.append(np.diagonal(hamil[i]).astype(np.float))

		diagList.append(_get_data_between(file, diagStartLine[i], diagEndLine[i]))
		diag.append(np.array(diagList[i]).reshape(int(len(diagList[i])/nM),nM))
		state_en.append(np.diagonal(diag[i]).astype(np.float))

		eigenList.append(_get_data_between(file, eigenStartLine[i], eigenEndLine[i]))
		eig_vecs.append(np.transpose(np.array(eigenList[i]).reshape(int(len(eigenList[i])/nM),nM)))

		state_dipsStr.append(_get_data_between(file, dipoleStartLine[i], dipoleEndLine[i]))
		state_dips.append((np.array(state_dipsStr[i]).astype(np.float)))

	dip_order = np.flip(np.argsort(state_dips[0]))
	en_order = np.argsort(pig_en[0])

	return pig_en, state_en, eig_vecs, state_dips, dip_order, en_order

def extracting_eigen_data_dict(file):
    startLine = _get_line(file, 'Hamiltonian')
    endLine = _get_line(file, 'Transformation')
    nM = endLine[0] - startLine[0] - 1
    numIter = len(startLine)
    print('\nnumber of iterations - ' + str(numIter))
    print('number of molecules/sites - ' + str(nM) + '\n')

    hamilStartLine = _get_line(file, 'Hamiltonian', plus = 1)
    hamilEndLine = _get_line(file, 'Hamiltonian', plus = nM+1)
    diagStartLine = _get_line(file, 'Diagonalized', plus = 1)
    diagEndLine = _get_line(file, 'Diagonalized', plus = nM+1)
    eigenStartLine = _get_line(file, 'Transformation', plus = 1)
    eigenEndLine = _get_line(file, 'Transformation', plus = nM+1)
    dipoleStartLine = _get_line(file, 'Dipoles', plus = 1)
    dipoleEndLine = _get_line(file, 'Dipoles', plus = 2)

    hamilList = []
    hamil = []
    pig_en = []

    diagList = []
    diag = []
    state_en = []

    eigenList = []
    eig_vecs =[]

    state_dipsStr = []
    state_dips = []

    for i in range(numIter):
        hamilList.append(_get_data_between(file, hamilStartLine[i], hamilEndLine[i]))
        hamil.append(np.array(hamilList[i]).reshape(int(len(hamilList[i])/nM),nM))
        pig_en.append(np.diagonal(hamil[i]).astype(np.float))

        diagList.append(_get_data_between(file, diagStartLine[i], diagEndLine[i]))
        diag.append(np.array(diagList[i]).reshape(int(len(diagList[i])/nM),nM))
        state_en.append(np.diagonal(diag[i]).astype(np.float))

        eigenList.append(_get_data_between(file, eigenStartLine[i], eigenEndLine[i]))
        eig_vecs.append(np.transpose(np.array(eigenList[i]).reshape(int(len(eigenList[i])/nM),nM)))

        state_dipsStr.append(_get_data_between(file, dipoleStartLine[i], dipoleEndLine[i]))
        state_dips.append((np.array(state_dipsStr[i]).astype(np.float)))

    dip_order = np.flip(np.argsort(state_dips[0]))
    en_order = np.argsort(pig_en[0])

    eigen_data = {
        'pigment energies': pig_en,
        'state energies': state_en,
        'eigenvectors': eig_vecs,
        'state dipoles': state_dips,
        'state energy order': en_order,
        'dipole order': dip_order
    	}

    return eigen_data

def test_dynamics(agg_list, energies):
    # Adding the energies to the molecules. Neeed to be done before agg
    with qr.energy_units("1/cm"):
        for i, mol in enumerate(agg_list):
            mol.set_energy(1, energies[i])

    # Creation of the aggregate for dynamics. multiplicity can be 1
    agg = qr.Aggregate(molecules=agg_list)
    agg.set_coupling_by_dipole_dipole(epsr=1.21)
    agg.build(mult=1)
    agg.diagonalize()
    with qr.energy_units('1/cm'):
        print(agg.get_Hamiltonian())

    # Creating a propagation axis length t13_ax plus padding with intervals 1
    t2_prop_axis = qr.TimeAxis(0.0, 1000, 1)

    # Generates the propagator to describe motion in the aggregate
    prop_Redfield = agg.get_ReducedDensityMatrixPropagator(
        t2_prop_axis,
        relaxation_theory="stR",
        time_dependent=False,
        secular_relaxation=True
        )

    # Obtaining the density matrix
    shp = agg.get_Hamiltonian().dim
    rho_i1 = qr.ReducedDensityMatrix(dim=shp, name="Initial DM")
    # Setting initial conditions
    rho_i1.data[shp-1,shp-1] = 1.0
    # Propagating the system along the t13_ax_ax time axis
    rho_t1 = prop_Redfield.propagate(rho_i1, name="Redfield evo from agg")
    rho_t1.plot(coherences=False, axis=[0,t2_prop_axis.length,0,1.0], show=True)


def _get_line(fileName, thisLine, plus = 0):

    lineList = []
    with open(fileName, 'r') as f:
        for lineNum, line in enumerate(f):
            line=line.strip()
            if re.search('^' + thisLine, line):
                lineList.append(lineNum + plus)

    return lineList

def _get_data_between(fileName, lineStart, lineFin):

    dataInput=[]
    with open(fileName, 'r') as f:
        for line_num, line in enumerate(f):
            line=line.strip()
            if (line_num >= lineStart) and (line_num < lineFin):
                data=line.split()
                dataInput+=data

    return dataInput