from ase.calculators.calculator import Calculator, all_changes
from ase import atoms
import numpy as np
from mpi4py import MPI
import sirius_ase.sirius_interface as sirius_interface
from ase import units

class SIRIUS(Calculator):
    
    implemented_properties = ['energy', 'forces', 'stress', 'bandgap', 'fermienergy']
    default_parameters = {}
    nolabel = True
    siriusInterface = None

    def __init__(self, atom: atoms.Atom, pp_files, functionals, kpoints: np.array
            , kshift: np.array, pw_cutoff: float, gk_cutoff: float
            , json_params :dict, communicator: MPI.Comm = MPI.COMM_WORLD):

        super().__init__()
        self.siriusInterface = sirius_interface.siriusInterface(atom.get_scaled_positions(wrap=False),
            atom.get_cell(True) / units.Bohr, atom.get_chemical_symbols(), pp_files, functionals, 
            kpoints, kshift, pw_cutoff, gk_cutoff, json_params)


    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties

        # print(system_changes, properties)
        super().calculate(atoms, properties, system_changes)

        if not system_changes == []:
            self.siriusInterface.findGroundState(atoms.get_scaled_positions(wrap = False), atoms.get_cell(True) / units.Bohr)

        if 'energy' in properties:
            self.results['energy'] = self.siriusInterface.getEnergy() * units.Hartree

        if 'forces' in properties:
            self.results['forces'] = self.siriusInterface.getForces() * (units.Hartree / units.Bohr)

        if 'stress' in properties:
            stress_sirius = self.siriusInterface.getStress() * ( units.Hartree / units.Bohr**3)
            stress_sirius = 0.5 * (stress_sirius + stress_sirius.T)
            self.results['stress'] = np.array([stress_sirius[0][0], stress_sirius[1][1], stress_sirius[2][2]
                , stress_sirius[1][2], stress_sirius[0][2], stress_sirius[0][1]])

        if 'bandgap' in properties:
            self.results['bandgap'] = self.siriusInterface.getBandGap() * units.Hartree

        if 'fermienergy' in properties:
            self.results['fermienergy'] = self.siriusInterface.getFermiEnergy() * units.Hartree


    def getBandGap(self):
        return self.get_property('bandgap')

    def getFermiEnergy(self):
        return self.get_property('fermienergy')


    def recalculateBasis(self, atoms):
        self.siriusInterface.resetSirius(atoms.get_scaled_positions(wrap = False), atoms.get_cell(True) / units.Bohr)

    def close(self):
        self.siriusInterface.exit()
        del(self.siriusInterface)
