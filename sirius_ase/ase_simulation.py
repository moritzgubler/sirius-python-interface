from ase import atoms
import ase.io
import os
import sys
import traceback
import numpy as np
import json
import sirius_ase.siriusCalculator
import argparse

def entry():

    parser = argparse.ArgumentParser(description ="""
    Reads sirius paramater and an ase list of structures (extxyz format is recommended)
    and sirius parameter json file and performs a DFT calculation of all structures.
    """)

    parser.add_argument('-s', '--sirius_parameters', dest ='sirius_parameters',
        action ='store', help ='Filename of json file that contains sirius parameters.', required=True)
    
    parser.add_argument('-g', '--geometry', dest ='filename',
        action ='store', help ='Filename of ASE compatible structure file.', required=True)
    
    parser.add_argument('-o', '--output', dest ='outfile',
        action ='store', help ='Filename of output file. Default is output.extxyz', 
        default="output.extxyz", required=False)
    
    parser.add_argument('-i', '--index', dest ='index',
        action ='store', help ='Index of structures, will be passed directly to ase.io.read()', 
        default=":", required=False)
    
    parser.add_argument('-p', '--pressure', dest ='pressure_gpa',
        action ='store', help ='Pressure that will be added to the system E->H=E+p*V, and pressure will be added to diagonal stress elements.',
        type=float, default=0.0, required=False)
    
    parser.add_argument('-n', '--no_recalculate', dest ='recalculateBasis',
        action ='store_false', help ='If present, basis set will not be recalculated before every structure. Can be set if the lattice vectors do not change between calculations.',
        default = True, required=False)

    args = parser.parse_args()

    if not os.path.exists(args.sirius_parameters):
        print('json file does not exist')
        quit()
    if not os.path.exists(args.filename):
        print('Input filename.')
        quit()
    

    atoms = ase.io.read(filename=args.filename, parallel=False, index = args.index)

    if not isinstance(atoms, list):
        atoms = [atoms]

    f = open(args.sirius_parameters)
    jsonparams = json.load(f)
    f.close()
    try:
        pp_files = jsonparams["unit_cell"]["atom_files"]
        pw_cutoff = jsonparams['parameters']["pw_cutoff"]
        gk_cutoff = jsonparams['parameters']["gk_cutoff"]
        functionals = jsonparams['parameters']['xc_functionals']
        kpoints = jsonparams['parameters']['ngridk']
        kshift = jsonparams['parameters']["shiftk"]
        if "atom_types" in jsonparams["unit_cell"]:
            jsonparams["unit_cell"].pop("atom_types")
        jsonparams["unit_cell"].pop("atom_files")
    except KeyError:
        print("required parameter was missing")
        traceback.print_exc()
        quit()

    calc = sirius_ase.siriusCalculator.SIRIUS(atoms[0], pp_files, functionals, kpoints,
                                              kshift, pw_cutoff, gk_cutoff, jsonparams,
                                              pressure_giga_pascale=args.pressure_gpa)
    
    i = 0
    for at in atoms:
        if i > 0 and args.recalculateBasis:
            calc.recalculateBasis(at)
        results = {}
        at.calc = calc
        results["positions"] = at.get_positions().tolist()
        cell = at.get_cell()
        results["cell_vector_a"] = cell[0,:].tolist()
        results["cell_vector_b"] = cell[1,:].tolist()
        results["cell_vector_c"] = cell[2,:].tolist()
        results["energy"] = at.get_potential_energy()
        results["forces"] = at.get_forces().tolist()
        results["stress"] = at.get_stress(voigt=False).tolist()
        print("Results of iteration " + str(i))
        print(json.dumps(results, indent=4))
        i += 1


    ase.io.write(args.outfile, atoms, append=False, parallel=False)

    calc.close()
