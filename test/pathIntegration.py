from pathIntegrationForceTest.PathIntegrationTest import PathIntegrationTest
from ase.lattice.cubic import Diamond
import sirius_ase.siriusCalculator as siriusCalculator
import numpy as np
from ase.lattice.cubic import Diamond
import ase
from ase.build import bulk

atoms = bulk('Si', crystalstructure='diamond')
np.random.seed(7340453)
atoms.positions = atoms.positions + 0.2 * (np.random.random(atoms.positions.shape) - 0.5)
atoms.cell = atoms.cell + 0.2 * (np.random.random((3,3)) - 0.5)

# atoms = Diamond('Si')
print(atoms.get_global_number_of_atoms())

pp_files = {'Si' : 'Si.json'}
pw_cutoff = 450 # in a.u.^-1
gk_cutoff = 100 # in a.u.^-1
functionals = ["XC_GGA_X_PBE", "XC_GGA_C_PBE"]
kpoints = np.array([2, 2, 2])
kshift = np.array([1, 1, 1])

jsonparams = {
    'mixer': {
        'beta': 0.5,
        'max_history': 8,
        'use_hartree': True,
    },
    "control" : {
        "verbosity" : 0,
        # "processing_unit" : 'gpu',
        # 'spglib_tolerance' : 1e-10
    },
    'parameters': {
        'use_symmetry': True,
        # 'smearing_width' : 0.0001
        'energy_tol' : 1e-7,
        'density_tol' : 1e-8
    }
}
atoms.calc = siriusCalculator.SIRIUS(atoms, pp_files, functionals, kpoints, kshift, pw_cutoff, gk_cutoff, jsonparams)

print(atoms.get_potential_energy(), np.linalg.norm(atoms.get_forces()), atoms.get_stress())

  # Setup: The pentagram is most fun
pathIntTest = PathIntegrationTest(atoms, 0.1, 100, shape='circle', check_stress=True, startingPointIsOnCircle = True, randomTrajectory = True)
# run the integration
pathIntTest.integrate()
# plot the energy and error
pathIntTest.plot_energy_along_path()
pathIntTest.plot_error_along_path()
# with the pentagram we can make these nice star shaped plots if we circle a local minimum
# pathIntTest.plot_pentagram_energy()
# pathIntTest.plot_pentagram_error()
pathIntTest.write_to_file('toto')
pathIntTest.write_trajectory('toto.extxyz')
atoms.calc.close()



