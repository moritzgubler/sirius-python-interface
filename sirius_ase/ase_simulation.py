from ase import atoms
import ase.io
import os
import sys

def aseSimulation(atoms: atoms.Atom, structufileName: str, outputFilename: str, startIndex:int =0, endIndex:int=-1
        , calc_energy = True, calc_forces = True, calc_stress = True):

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
