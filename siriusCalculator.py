from ase.calculators.calculator import Calculator, all_changes
from ase import atoms
import numpy as np
from mpi4py import MPI
import sirius_interface

class LennardJones(Calculator):
    
    implemented_properties = ['energy', 'forces', 'stress']
    default_parameters = {}
    nolabel = True
    siriusInterface = None

    def __init__(self, atom: atoms.Atom, pp_files, functionals, kpoints: np.array
            , kshift: np.array, pw_cutoff: float, gk_cutoff: float
            , json_params :str, communicator: MPI.Comm = MPI.COMM_WORLD):

        Calculator.__init__(self)
        self.siriusInterface = sirius_interface.siriusInterface(communicator, atom.get_postions(),
            atom.get_cell(True), atom.get_chemical_symbols(), pp_files, functionals, 
            kpoints, kshift, pw_cutoff, gk_cutoff, json_params)


    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)

        if 'energy' in properties:
            self.results['energy'] = self.siriusInterface.getEnergy(atoms.get_positions(), atoms.get_cell(True))

        if 'forces' in properties:
            self.results['forces'] = self.siriusInterface.getForces()

        if 'stress' in properties:
            self.results['stress'] = self.siriusInterface.getStress()