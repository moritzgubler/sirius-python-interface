from ase import atoms
import ase.io
import os
import sys
import traceback
import numpy as np
import json
import sirius_ase.siriusCalculator
import argparse


def aseSimulation(atoms: atoms.Atom, structufileName: str, outputFilename: str,
                  startIndex:int =0, endIndex:int=-1,calc_energy = True, 
                  calc_forces = True, calc_stress = True):

    atom_list = ase.io.read(filename=structufileName, parallel=False, index = ':')

    n_atoms = len(atom_list)

    calc = atoms.calc

    if os.path.exists(outputFilename):
        os.remove(outputFilename)

    if endIndex < 0:
        endIndex = endIndex + n_atoms + 1
    if endIndex > n_atoms:
        endIndex = n_atoms
    if startIndex >= endIndex:
        startIndex = endIndex

    if structufileName == outputFilename:
        print('input and outputfile are identical. aborting')
        return

    firstIteration = True

    for i in range(startIndex, endIndex):
        atom_list[i].calc = calc

        # recalculate basis when a new structure is loaded for most precise results
        if firstIteration:
            firstIteration = False
        else:
            atom_list[i].calc.recalculateBasis(atom_list[i])

        print('start calculating properties of structure ' + str(i))
        sys.stdout.flush()
        if calc_energy:
            atom_list[i].get_potential_energy()
            print('Potential energy: ', atom_list[i].get_potential_energy())
        if calc_forces:
            atom_list[i].get_forces()
        if calc_stress:
            atom_list[i].get_stress()

        sys.stdout.flush()

        ase.io.write(outputFilename, atom_list[i], append=True, parallel=False)
        # write file also to standard out in order to monitor progress
        # ase.io.write('-', atom_list[i], parallel=False)
        sys.stdout.flush()


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
        action ='store', help ='Pressure that will be added to the system',
        type=float, default=0.0, required=False)

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
    
    for at in atoms:
        at.calc = calc
        at.get_potential_energy()
        at.get_forces()
        at.get_stress

    ase.io.write(args.outfile, atoms, append=False, parallel=False)

    calc.close()
