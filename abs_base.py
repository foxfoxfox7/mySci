# -*- coding: utf-8 -*-
import os
import time
import random
import matplotlib.pyplot as plt
import numpy as np

from quantarhei import Aggregate
from quantarhei import energy_units
from quantarhei import CorrelationFunction
from quantarhei import SpectralDensity
from quantarhei import time_axxis
from quantarhei import DFunction
from quantarhei import Molecule
from quantarhei.spectroscopy.absbase import AbsSpectrumBase
from quantarhei.builders.pdb import PDBFile
from quantarhei.core.units import kB_int 
from quantarhei.core.units import convert
from quantarhei.models.bacteriochlorophylls import BacterioChlorophyll
from quantarhei.models.spectdens import SpectralDensityDB

import quantarhei as qr


time_ax = time_axxis(0.0, 200, 5.0)
temperature = 300.0

#***********************************************************************
#*                                                                     *
#*                       Creating the Aggregate                        *
#*                                                                     *
#***********************************************************************
'''
params = dict(ftype="OverdampedBrownian", T=temperature, reorg=50.0, cortime=100.0)
with energy_units('1/cm'):
  cf = CorrelationFunction(time_ax, params)

forAggregate = []
for i in range(30):
    molName = 'molecule' + str(i)
    with energy_units("1/cm"):
        molName = Molecule(elenergies=[0.0, 13000.0+(i*200)])
    molName.set_transition_environment((0,1),cf)
    molName.position = [0.0, 0.0, i*20.0]
    molName.set_dipole(0,1,[0.0, 10.0, 0.0])
    forAggregate.append(molName)
'''
#***********************************************************************
#*                                                                     *
#*                            The PDB system                           *
#*                                                                     *
#***********************************************************************

pdb_name = 'LH1'
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
energies = []
with open(pdb_name + "_input.txt") as file_names:
    for line in file_names:
        pigment_name.append(line.split('\t')[0])
        pigment_type.append(line.split('\t')[1])
        energies.append(float(line.split('\t')[2]))
naming_map = dict(zip(pigment_name, pigment_type))

# Make a list of the molecules and set their molecule type
forAggregate = []
for name in pigment_name:
    for m in molecules:
        if m.name == name:
            m.set_name(naming_map[name])
            forAggregate.append(m)

params = {"cortime":          100.0,
          "T":                temperature,
          "matsubara":        20}
db = SpectralDensityDB(verbose=False)
sd_renger = db.get_SpectralDensity(time_ax, "Renger_JCP_2002")
sd_wendling = db.get_SpectralDensity(time_ax, "Wendling_JPCB_104_2000_5825")

# Combining them to make our spectral density
ax = sd_renger.axis
sd_wendling.axis = ax
sd_tot = sd_renger + sd_wendling

# Obtaining the correlation function (and plotting)
with energy_units("1/cm"):
    corr_fun = sd_tot.get_CorrelationFunction(temperature)

for j, mol in enumerate(forAggregate):
    mol.set_transition_environment((0,1), corr_fun)
    # Initially sets the energies without static disorder
    with energy_units("1/cm"):
        #mol.set_energy(1, energies[j])
        mol.set_energy(1, random.gauss(energies[j], 400))

#***********************************************************************
#*                                                                     *
#*                      Hamiltonian and propagator                     *
#*                                                                     *
#***********************************************************************

agg = Aggregate(molecules=forAggregate)
agg.set_coupling_by_dipole_dipole(epsr=1.21)
agg.build()
rwa = agg.get_RWA_suggestion()

with energy_units("1/cm"):
    print(agg.get_Hamiltonian())
    for i in range(2):
        print(forAggregate[i])
print("Calculating the propagator...\n")

# Returning the propagator (tensor is included). stR=standard Redfield
t1 = time.time()
prop_Redfield = agg.get_ReducedDensityMatrixPropagator(
	time_ax,
	relaxation_theory="stR",
	time_dependent=False, 
	secular_relaxation=True
	)
t2 = time.time()
print("Redfield propagator done in ", t2-t1, " s\n")

#***********************************************************************
#*                                                                     *
#*                      Dynamics of the system                         *
#*                                                                     *
#***********************************************************************
'''
print("Calculating and printing the dynamics of the system...\n")
# Initial density matrix
shp = H.dim
rho_i1 = qr.ReducedDensityMatrix(dim=shp, name="Initial DM")
rho_i1.data[shp-1,shp-1] = 1.0   
   
# Propagation of the density matrix
t1 = time.time() 
rho_t1 = prop_Redfield.propagate(rho_i1,
                                 name="Redfield evolution from aggregate")
t2 = time.time()
print("Propagation of density matrix done in ", t2-t1, " s\n")

if _show_plots_: 
    rho_t1.plot(coherences=True, axis=[0,Nt*dt,0,1.0], show=True)
'''
#***********************************************************************
#*                                                                     *
#*                Calculating the absorption spectrum                  *
#*                                                                     *
#***********************************************************************

print("Calculating the absorption spectrum...\n")
calc = qr.AbsSpectrumCalculator(time_ax, system=agg, 
                              relaxation_tensor=prop_Redfield.RelaxationTensor)
calc.bootstrap(rwa=rwa)
abs_calc = calc.calculate()
abs_calc.normalize2()

#***********************************************************************
#*                                                                     *
#*                   Reading the measured abs spec                     *
#*                                                                     *
#***********************************************************************
'''
data_load = np.loadtxt('all_data.txt')
energy, seventy, room = np.hsplit(data_load, 3)
abs_load = AbsSpectrumBase()
# gives the correct interval step after conversion from nm to int
abs_load.set_by_interpolation(x=energy, y=room, xaxis="wavelength")
abs_load.normalize2()
'''
#***********************************************************************
#*                                                                     *
#*                            Plotting                                 *
#*                                                                     *
#***********************************************************************
'''
abs_show = AbsSpectrumBase()
abs_show.load_data('all_data.txt')
abs_show.set_axis(energy)
#abs_show.set_axis(energy)
#abs_show.set_data(room)
abs_show.plot()
'''
# Showing the response function
data_load = np.loadtxt('response2d.txt')
resR, resI, time = np.hsplit(data_load, 3)
plt.plot(time, resR, 'r--')
plt.plot(time, resI, 'b--')
plt.show()

# Whole spectrum
with qr.energy_units("1/cm"):
    abs_calc.plot()

