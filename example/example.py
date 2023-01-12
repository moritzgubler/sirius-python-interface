import sirius_ase.ase_simulation
import sirius_ase.siriusCalculator
import numpy as np
import json
import traceback
import sys
import ase.io
import os

siriusJsonFileName = sys.argv[1]
structufileName = sys.argv[2]
outputFilename = sys.argv[3]
if not os.path.exists(siriusJsonFileName):
    print('json file does not exist')
    quit()

atoms = ase.io.read(filename=structufileName, parallel=False, index = 0)

f = open(siriusJsonFileName)
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

atoms.calc = sirius_ase.siriusCalculator.SIRIUS(atoms, pp_files, functionals, kpoints, kshift, pw_cutoff, gk_cutoff, jsonparams)

sirius_ase.ase_simulation.aseSimulation(atoms, structufileName, outputFilename)

atoms.calc.close()