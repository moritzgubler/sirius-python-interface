# ASE calculator for SIRIUS
This repository contains a convenient python interface for the [SIRIUS electronic structure library](https://github.com/electronic-structure/SIRIUS.git). It creates an ASE calculator that uses SIRIUS. 
### Description and example
Implemented properties of the calculator are energies, forces, stress, Fermi energy and the band gap. The calculator is mpi compatible. When mpi is used, the calculator keeps all mpi processes after initialization. Only rank zero is returned. When the calculator is closed, the mpi processes that have been kept back will exit. Rank zero is once again the only rank that will return from the close function.

```python
from ase.lattice.cubic import Diamond
import sirius_ase.siriusCalculator as siriusCalculator
atoms = Diamond('Si')

# set up Parameters for SIRIUS
pp_files = {
    # Define the pseudopotential for every atom present is the system.
    # Convert pseudopotentials with the upf_to_json utility from SIRIUS or 
    # get them from here:
    # https://github.com/electronic-structure/species
    'Si': 'Si.json'
}
# Plane wave and density cutoff in Ry.
pw_cutoff = 450 # in a.u.^-1
gk_cutoff = 100 # in a.u.^-1
functionals = ["XC_GGA_X_PBE", "XC_GGA_C_PBE"]
kpoints = np.array([2, 2, 2])
kshift = np.array([1, 1, 1])

# Additional parameters can be passed with the json params dictionary. 
jsonparams = {
    'mixer': {
        'beta': 0.5,
        'max_history': 8,
        'use_hartree': True,
    },
    "control" : {
        "verbosity" : 0,
    },
    'parameters': {
        'use_symmetry': False,
        "use_scf_correction": False,
        'energy_tol' : 1e-7,
        'density_tol' : 1e-8
    }
}

# Two additional paramaters are:
# * pressure_giga_pascale: Pressure in gigapascale. Adds p*V to energy and the pressure to the stress tensor.
# * communicator: MPI comminicator (from mpi4py) that SIRIUS should use. Can be a group communicator.

# Set up the ase communicator
atoms.calc = siriusCalculator.SIRIUS(atoms, pp_files, functionals, 
    kpoints, kshift, pw_cutoff, gk_cutoff, jsonparams)
    
# When mpi is used, code before the initializing the calculator is executed by every mpi rank. 
# This is be problematic when files are written.
# After the calculator is initialized, only the master process is returned
# and the simulation should behave as if it was a serial calculation.
    
# Do simuation...
atoms.get_potential_energy()
# ...

# Close sirius Calculator
atoms.calc.close()
```

### Installation
Install the SIRIUS library with python bindings enabled:
```
spack install sirius@develop +python
```
At the time of writing, the sirius version of the develop branch must be used. The necessary features will probably be in the next sirius release.
Install this repository with pip:
```
pip install .
```

### Command line interface
After the installation with pip the executable ```SiriusSinglePoint``` is added to the path in the pip directory. Check the installation and the documentation with ```SiriusSinglePoint -h```. The program reads a json file that contains the sirius parameters and a list of ase compatible periodic structure files. The energies, forces and stress tensor will be written into a new extxyz file.
