import re
import random
import math
import os
import copy
import numpy as np
import matplotlib.pyplot as plt

import quantarhei as qr
from quantarhei.models.spectdens import SpectralDensityDB
from quantarhei.models.bacteriochlorophylls import BacterioChlorophyll
import aceto
from aceto.lab_settings import lab_settings


class AggregateInstance():

    def __init__(self):

        self.num_mol = 0
        self.mol_list = []
        self.agg = None

        self.dipole_strength = 0
        self.static_dis = 300
        self.energy = 12500
        self.reorg = 102
        self.cor_time = 100
        self.temp = 300

        # Only needed for mock calculator
        self.width = 0

        self.pig_ens = None
        self.state_ens = None
        self.eig_vecs = None
        self.dipoles = None
        self.en_order = None
        self.dip_order = None

        self.resp_calc = None
        self.resp_cont_dict = {}
        self.spec_cont_dict = {}
        self.response = {}

        self.point_list = {}
        self.anisotropy = {}

    def get_molecules_circular(self, nM, dip = 5, dist = 8.7, dif = 0.6, verbose = True):

        self.dipole_strength = dip

        r = 5
        while dif > 0.1:
            t = np.linspace(0, np.pi * 2, nM+1)
            x = r * np.cos(t)
            y = r * np.sin(t)
            circle = np.c_[x, y]

            artificialDis = math.sqrt(((circle[0][0] - circle[1][0])**2)\
                + ((circle[0][1] - circle[1][1])**2))
            dif = abs(artificialDis-dist)

            if artificialDis > dist:
                r = r - 0.1
            elif artificialDis < dist:
                r = r + 0.1
        circle2 = np.delete(circle, nM, 0)

        dipoles = np.empty([nM,2])
        mag = math.sqrt((circle[0][0]**2) + (circle[0][1]**2))
        for i in range(nM):
            dipoles[i][0] = -circle2[i][1]
            dipoles[i][1] = circle2[i][0]
            dipoles[i][0] = dipoles[i][0] / mag
            dipoles[i][1] = dipoles[i][1] / mag
            dipoles[i][0] = dipoles[i][0] * self.dipole_strength
            dipoles[i][1] = dipoles[i][1] * self.dipole_strength

        forAggregate = []
        for i in range(nM):
            molName = qr.Molecule()
            molName.position = [circle2[i][0], circle2[i][1], 0.0]
            molName.set_dipole(0,1,[dipoles[i][0], dipoles[i][1], 0.0])
            forAggregate.append(molName)

        if verbose:
            print('\nA list of molecules was generated. Positions are in a '
                'ring with ', dist, ' Angstrom spacings. Dipoles are '
                'added running along the tangent of the ring. All in the '
                'same direction with ', self.dipole_strength, ' dipoles (D)\n')

        #self.positions = circle2
        #self.dipoles = dipoles
        self.mol_list = forAggregate

    def get_molecules_pdb(self, name, verbose = True):

        pdb_name = name
        file = qr.PDBFile(pdb_name + ".pdb")

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

        if verbose:
            print('\n A list of molecules has been extracted from a PDB file. '
                'molecules are of BCl type and have the appropriate position '
                'and dipole.\n')

        self.mol_list = forAggregate

    def assign_spec_dens(self, sd_type = 'OverdampedBrownian', high_freq = True):

        t_ax_sd = qr.TimeAxis(0.0, 10000, 1)
        db = SpectralDensityDB()

        # Parameters for spectral density. ODBO, Renger or Silbey
        params = {
            "ftype": sd_type,
            "alternative_form": True,
            "reorg": self.reorg,
            "T": self.temp,
            "cortime": self.cor_time
            }
        with qr.energy_units('1/cm'):
            sd_low_freq = qr.SpectralDensity(t_ax_sd, params)

        if high_freq:
            # Adding the high freq modes
            sd_high_freq = db.get_SpectralDensity(t_ax_sd, "Wendling_JPCB_104_2000_5825")
            ax = sd_low_freq.axis
            sd_high_freq.axis = ax
            sd_tot = sd_low_freq + sd_high_freq

            cf = sd_tot.get_CorrelationFunction(temperature=self.temp, ta=t_ax_sd)
            # Assigning the correlation function to the list of molecules
        else:
            cf = sd_low_freq.get_CorrelationFunction(temperature=self.temp, ta=t_ax_sd)

        for mol in self.mol_list:
            mol.set_transition_environment((0,1),cf)

    def assign_energies(self, method = None):

        nM = len(self.mol_list)

        if method == 'same':
            energies = [self.energy] * nM
        elif method == 'different':
            energies = [self.energy - (100 * nM / 2) + i * 100\
             for i in range(nM)]
        else:
            energies = [random.gauss(self.energy, self.static_dis)\
             for i in range(nM)]

        with qr.energy_units("1/cm"):
            for i, mol in enumerate(self.mol_list):
                mol.set_energy(1, energies[i])

    def build_agg(self, agg_list = None, mult = 1, diagonalize = True):
        
        if not agg_list:
            agg_list = self.mol_list

        agg = qr.Aggregate(molecules=agg_list)
        agg.set_coupling_by_dipole_dipole(epsr=1.21)
        agg.build(mult=mult)
        if diagonalize:
           agg.diagonalize()

        self.agg = agg
        self.num_mol = agg.nmono
        self.rwa = agg.get_RWA_suggestion()

    def test_dynamics(self, en_method = None):
        
        # Adding the energies to the molecules. Neeed to be done before agg
        mol_list_temp = self._temp_assign_energies(method = en_method)
        aggregate = self._temp_build_agg(agg_list = mol_list_temp)

        # Propagation axis length t13_ax plus padding with intervals of 1
        #t1_len = int(((t13_ax.length+padding-1)*t13_ax.step)+1)
        t2_prop_axis = qr.TimeAxis(0.0, 1000, 1)

        # Generates the propagator to describe motion in the aggregate
        prop_Redfield = aggregate.get_ReducedDensityMatrixPropagator(
            t2_prop_axis,
            relaxation_theory="stR",
            time_dependent=False,
            secular_relaxation=True
            )

        # Obtaining the density matrix
        shp = aggregate.get_Hamiltonian().dim
        rho_i1 = qr.ReducedDensityMatrix(dim=shp, name="Initial DM")
        # Setting initial conditions
        rho_i1.data[shp-1,shp-1] = 1.0
        # Propagating the system along the t13_ax_ax time axis
        rho_t1 = prop_Redfield.propagate(rho_i1, name="Redfield evo from agg")
        rho_t1.plot(coherences=False, axis=[0,t2_prop_axis.length,0,1.0], show=True)

    def get_state_data(self):

        H = self.agg.get_Hamiltonian()
        SS = H.diagonalize()
        trans1 = SS[1:self.num_mol+1,1:self.num_mol+1]
        self.trans_mat = copy.copy(trans1)
        self.eig_vecs = np.transpose(trans1)

        H.undiagonalize()
        hamil = self.agg.HH[1:self.num_mol+1,1:self.num_mol+1]
        self.hamiltonian = copy.copy(hamil)
        self.pig_ens = np.diagonal(hamil)
        self.en_order = np.argsort(self.pig_ens)


        self.agg.diagonalize()
        diag = self.agg.HH[1:self.num_mol+1,1:self.num_mol+1]
        self.diag_ham = copy.copy(diag)
        self.state_ens = np.diagonal(diag)

        dips = self.agg.D2[0][1:self.num_mol+1].reshape(1, self.num_mol)
        self.dipoles = copy.copy(dips)
        self.dip_order = np.flip(np.argsort(self.dipoles))

        eigen_data = {
            'pigment energies': self.pig_ens,
            'state energies': self.state_ens,
            'eigenvectors': self.eig_vecs,
            'state dipoles': self.dipoles,
            'state energy order': self.en_order,
            'dipole order': self.dip_order
            }

        return eigen_data 
        
    def save_state_data(self, file = 'eigen_dat.txt'):
 
        with open(file, 'a') as f:
            f.write('Hamiltonian\n')
            np.savetxt(f, self.hamiltonian)

            f.write('Transformation Matrix\n')
            np.savetxt(f, self.trans_mat)

            f.write('Diagonalized\n')
            np.savetxt(f, self.diag_ham)

            f.write('Dipoles\n')
            np.savetxt(f, self.dipoles)

    def twod_setup(self, ax2, ax13):

        self.resp_calc = qr.TwoDResponseCalculator(
            t1axis=ax13,
            t2axis=ax2,
            t3axis=ax13,
            system=self.agg
            )

        self.t1axis = ax13
        self.t2axis = ax2

    def twod_calculate(self, lab, pad = 0, name = 'place_holder'):#, resp = False

        print('Calculating twod')
        calc = copy.deepcopy(self.resp_calc)
        calc.bootstrap(self.rwa, lab = lab, pad = pad)#, printResp = resp
        resp_cont = calc.calculate()
        self.resp_cont_dict.update({name: resp_cont})
        spec_cont = resp_cont.get_TwoDSpectrumContainer()
        self.spec_cont_dict.update({name: spec_cont})

        self.pad = pad
        
        #if resp:

    def print_spectra(self, name = 'place_holder'):#, spread = 700

        for i, tt2 in enumerate(self.resp_calc.t2axis.data):
            twod = self.spec_cont_dict[name].get_spectrum(self.resp_calc.t2axis.data.data[i])
            #max_val = twod.get_max_value()
            with qr.energy_units('1/cm'):
                twod.plot()
                plt.xlim(11400, 13100)
                plt.ylim(11400, 13100)
                plt.title(name + str(int(tt2)))
                plt.show()

    def save_spectra(self, location = '.', name = 'place_holder'):#, spread = 700

        location = self._make_data_dir(location)

        for i, tt2 in enumerate(self.resp_calc.t2axis.data):
            twod = self.spec_cont_dict[name].get_spectrum(self.resp_calc.t2axis.data.data[i])
            #max_val = twod.get_max_value()
            with qr.energy_units('1/cm'):
                twod.plot()
                plt.xlim(11400, 13100)
                plt.ylim(11400, 13100)
                plt.title(name + str(int(tt2)))
                plt.savefig(location + name + str(int(tt2)) + '.png')

    def get_max_list(self, name = 'place_holder'):

        points = np.empty(len(self.resp_calc.t2axis.data))

        for i, tt2 in enumerate(self.resp_calc.t2axis.data):
            twod = self.spec_cont_dict[name].get_spectrum(self.resp_calc.t2axis.data.data[i])
            with qr.energy_units('1/cm'):
                points[i] = twod.get_max_value()
        
        if not name in self.point_list.keys():
            self.point_list.update({name: {}})
        self.point_list[name].update({'max': points})

    def get_point_list(self, point, name = 'place_holder'):

        points = np.empty(len(self.resp_calc.t2axis.data))

        for i, tt2 in enumerate(self.resp_calc.t2axis.data):
            twod = self.spec_cont_dict[name].get_spectrum(self.resp_calc.t2axis.data.data[i])
            with qr.energy_units('1/cm'):
                points[i] = twod.get_value_at(point, point).real
        
        if not name in self.point_list.keys():
            self.point_list.update({name: {}})
        self.point_list[name].update({str(point): points})

    def get_anisotropy(self, name1, name2):

        dictionary1 = self.point_list[name1]
        dictionary2 = self.point_list[name2]

        for key in dictionary1:
            anis = (dictionary1[key] - dictionary2[key]) / (dictionary1[key] + (2 * dictionary2[key]))
            self.anisotropy.update({key: anis})

    def print_data(self):

        print(self.agg)
        print('pigment energies - \n', self.pig_ens)
        print('state energies - \n', self.state_ens)
        print('eigenvectors - \n', self.eig_vecs)
        print('energy order - \n', self.en_order)
        print('dipole order - \n', self.dip_order)
        print('anisotropy - \n', self.anisotropy)

    def _make_data_dir(self, location):

        if not os.path.exists(location):
            os.mkdir(location)
        location = location + '/'

        return location

    def _temp_assign_energies(self, method = None):

        nM = len(self.mol_list)
        new_list = copy.deepcopy(self.mol_list)

        if method == 'same':
            energies = [self.energy] * nM
        elif method == 'different':
            energies = [self.energy - (100 * nM / 2) + i * 100\
             for i in range(nM)]
        else:
            energies = [random.gauss(self.energy, self.static_dis)\
             for i in range(nM)]

        with qr.energy_units("1/cm"):
            for i, mol in enumerate(new_list):
                mol.set_energy(1, energies[i])

        return new_list

    def _temp_build_agg(self, agg_list = None , mult = 1, diagonalize = True):

        if not agg_list:
            agg_list = self.mol_list
        agg = qr.Aggregate(molecules=agg_list)
        agg.set_coupling_by_dipole_dipole(epsr=1.21)
        agg.build(mult=mult)
        if diagonalize:
           agg.diagonalize()

        return agg


class AggregateAverage():

    def __init__(self):

        self.t2_axis = None
        self.spectra_data = {}

        self.point_list = {}
        self.anisotropy = {}
        self.ipr = {}
        
        
    def twod_setup(self, ax2, ax13, pad = 0, name = 'place_holder'):

        ax13 = ax13
        self.t2_axis = ax2
        #self.centre = qr.convert(instance.rwa, 'int', '1/cm')
        self.centre = 12500
        init = AggregateInstance()
        init.get_molecules_circular(nM = 2, verbose = False)
        init.assign_spec_dens()
        init.assign_energies()
        init.build_agg(mult = 2)
        
        a_0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        lab = lab_settings(lab_settings.FOUR_WAVE_MIXING)
        lab.set_laser_polarizations(a_0,a_0,a_0,a_0)

        init.twod_setup(ax2 = self.t2_axis, ax13 = ax13)
        init.twod_calculate(lab = lab, pad = pad, name = name)
        
        self.spectra_data.update({name: []})
        for i, time in enumerate(self.t2_axis.data):
            
            spec = init.spec_cont_dict[name].get_spectrum(time)
            self.spectra_data[name].append(spec)
            self.spectra_data[name][i].add_data(-spec.data)

    def add_data(self, container, name = 'place_holder'):

        for i, time in enumerate(self.t2_axis.data):
            spec = container[name].get_spectrum(time)
            self.spectra_data[name][i].add_data(spec.data)

    def normalise(self, num, name = 'place_holder'):

        for i, time in enumerate(self.t2_axis.data):
            self.spectra_data[name][i].devide_by(num)

    def plot_spec(self, disp_range = 1000, name = 'place_holder'):

        en1 = 11000
        en2 = 13500
        window = [en1, en2, en1, en2]

        #window = [self.centre - disp_range, self.centre + disp_range,\
        # self.centre - disp_range, self.centre + disp_range]


        for i, time in enumerate(self.t2_axis.data):
            with qr.energy_units('1/cm'):
                self.spectra_data[name][i].plot(window = window)
                plt.title(name + str(int(time)))
                plt.show()

    def save_spec(self, disp_range = 1000, location = '.', name = 'place_holder'):

        window = [self.centre - disp_range, self.centre + disp_range,\
         self.centre - disp_range, self.centre + disp_range]

        location = self._make_data_dir(location)

        for i, time in enumerate(self.t2_axis.data):
            with qr.energy_units('1/cm'):
                self.spectra_data[name][i].plot(window = window)#
                plt.title(name + str(int(time)))
                plt.savefig(location + name + str(int(time)) + '.png')

    def get_max_list(self, name = 'place_holder'):

        points = np.empty(len(self.spectra_data[name]))

        for i, time in enumerate(self.spectra_data[name]):
            with qr.energy_units('1/cm'):
                points[i] = time.get_max_value().real

        if not name in self.point_list.keys():
            self.point_list.update({name: {}})
        self.point_list[name].update({'max': points})

    def get_point_list(self, point, name = 'place_holder'):

        points = np.empty(len(self.spectra_data[name]))

        for i, time in enumerate(self.spectra_data[name]):
            with qr.energy_units('1/cm'):
                points[i] = time.get_value_at(point, point).real

        if not name in self.point_list.keys():
            self.point_list.update({name: {}})
        self.point_list[name].update({'max': points})

    def get_anisotropy(self, name1, name2, _print_=False):

        dictionary1 = self.point_list[name1]
        dictionary2 = self.point_list[name2]

        for key in dictionary1:
            anis = (dictionary1[key] - dictionary2[key]) / (dictionary1[key] + (2 * dictionary2[key]))
            self.anisotropy.update({key: anis})

        if _print_:
            print(self.anisotropy)

    def _make_data_dir(self, location):

        if not os.path.exists(location):
            os.mkdir(location)
        location = location + '/'

        return location
