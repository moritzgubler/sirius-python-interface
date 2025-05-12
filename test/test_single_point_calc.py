"""Demonstrates molecular dynamics with constant energy."""

from ase.lattice.cubic import Diamond
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
import sirius_ase.siriusCalculator as siriusCalculator
import numpy as np
from ase.io import read


import sqnm.vcsqnm_for_ase

# Use Asap for a huge performance increase if it is installed
use_asap = False



# Set up a crystal
# atoms = FaceCenteredCubic(directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
#                           symbol='Si',
#                           size=(1, 1, 1),
#                           pbc=True)
atoms = read('toto/2.extxyz')
atoms.rattle(0.01)


pp_files = {'Si' : 'Si.json'}
pw_cutoff = 400 # in a.u.^-1
gk_cutoff = 100 # in a.u.^-1
functionals = ["XC_GGA_X_PBE", "XC_GGA_C_PBE"]
kpoints = np.array([2, 2, 2])
kshift = np.array([1, 1, 1])


jsonparams = {
    'mixer': {
        'beta': 0.5,
        'max_history': 8,
        'use_hartree': True
        
    },
    "control" : {
        "verbosity" : 0,
        "processing_unit" : 'auto',
    },
    'parameters': {
        'use_symmetry': True,
        'smearing_width' : 0.001,
        'energy_tol' : 1e-7,
        'density_tol' : 1e-7
    }
}



atoms.calc = siriusCalculator.SIRIUS(atoms, pp_files, functionals, kpoints, kshift, pw_cutoff, gk_cutoff, jsonparams)

opt = sqnm.vcsqnm_for_ase.aseOptimizer(atoms, True)
for i in range(20):
    opt.step(atoms)
    print(atoms.get_potential_energy(), np.max(np.abs(atoms.get_forces())), np.max(np.abs((atoms.get_stress())) ) )
    print('etot', atoms.get_potential_energy())
    print('bandgap', atoms.calc.getBandGap())
    print('fermienergy', atoms.calc.getFermiEnergy())

print('etot', atoms.get_potential_energy())
print('bandgap', atoms.calc.getBandGap())
print('fermienergy', atoms.calc.getFermiEnergy())
atoms.calc.close()


