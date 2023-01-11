"""Demonstrates molecular dynamics with constant energy."""

from ase.lattice.cubic import Diamond
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
import siriusCalculator
import numpy as np

# Use Asap for a huge performance increase if it is installed
use_asap = False



# Set up a crystal
# atoms = FaceCenteredCubic(directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
#                           symbol='Si',
#                           size=(1, 1, 1),
#                           pbc=True)
atoms = Diamond('Si')


pp_files = {'Si' : 'Si.json'}
pw_cutoff = 400 # in a.u.^-1
gk_cutoff = 80 # in a.u.^-1
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
        "processing_unit" : 'auto'
    }
}
atoms.calc = siriusCalculator.SIRIUS(atoms, pp_files, functionals, kpoints, kshift, pw_cutoff, gk_cutoff, jsonparams)

# Set the momenta corresponding to T=300K
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# We want to run MD with constant energy using the VelocityVerlet algorithm.
dyn = VelocityVerlet(atoms, 5 * units.fs)  # 5 fs time step.


def printenergy(a):
    """Function to print the potential, kinetic and total energy"""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))


# Now run the dynamics
printenergy(atoms)
for i in range(30):
    dyn.run(1)
    printenergy(atoms)