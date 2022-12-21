import sirius
import json
import numpy as np
from mpi4py import MPI


def createChapters(json_params):
    chapters = ['mixer', 'settings', 'unit_cell', 'iterative_solver', 'control', 'parameters', 'nlcg', 'hubbard']
    for chap in chapters:
        if chap not in json_params:
            json_params[chap] = {}


class siriusInterface:

    communicator = None
    isMaster = False
    initPos = None
    initLat = None
    context = None
    paramDict = None
    kgrid = None
    dft = None

    def __init__(self, communicator: MPI.Comm, pos: np.array, lat: np.array, atomNames, pp_files, functionals, kpoints: np.array, kshift: np.array, pw_cutoff: float, gk_cutoff: float, json_params :str):
        self.communicator = communicator
        self.paramDict = json.loads(json_params)
        createChapters(self.paramDict)
        # self.paramDict['parameters']['ngridk'] = kpoints
        # self.paramDict['parameters']['kshift'] = kshift
        self.paramDict["parameters"]['pw_cutoff'] = np.sqrt(pw_cutoff)
        self.paramDict["parameters"]['gk_cutoff'] = np.sqrt(gk_cutoff)
        self.paramDict["parameters"]['xc_functionals'] = functionals

        jsonstring = json.dumps(self.paramDict)

        # sc = sirius.Communicator(self.communicator)

        self.context = sirius.Simulation_context(jsonstring)
        self.context.unit_cell().set_lattice_vectors(lat[0, :], lat[1,:], lat[2, :])

        for element in pp_files:
            self.context.unit_cell().add_atom_type(element, pp_files[element])
        for element in atomNames:
            if element not in pp_files:
                print('Element has no corresponding pseudopotential file', element)
                quit()
        
        for i in range(pos.shape[0]):
            self.context.unit_cell().add_atom(atomNames[i], pos[i, :])

        self.context.initialize()
        self. kgrid = sirius.K_point_set(self.context, kpoints, kshift, True)
        self.dft = sirius.DFT_ground_state(self.kgrid)
        self.dft.initial_state()

    def calculate(self, pos, lat, atomNames):
        self.context.unit_cell().set_lattice_vectors(lat[0, :], lat[1,:], lat[2, :])
        for i, p in enumerate(pos):
            print(i, pos)
            self.context.set_atom_positions(i, p)
        self.dft.update()
        result = self.dft.find(1e-6, 1e-6, 1e-2, 100, False)
        print(json.dumps(result, indent=2))

if __name__ == '__main__':

    comm = MPI.COMM_WORLD

    a0 = 5.31148326730872
    lat = np.array(
        [[0.0, a0, a0],
         [a0, 0.0, a0],
         [a0, a0, 0.0]
    ])

    pos = np.array([[0, 0, 0],
            [0.25, 0.25, 0.25]])
    atomNames = ['Si', 'Si']
    pp_files = {'Si' : 'Si.json'}

    pw_cutoff = 400 # in a.u.^-1
    gk_cutoff = 80 # in a.u.^-1
    
    funtionals = ["XC_LDA_X", "XC_LDA_C_PZ"]
    kpoints = np.array([3, 3, 3])
    kshift = np.array([0, 0, 0])

    jsonparams = "{}"
    s = siriusInterface(comm, pos, lat, atomNames, pp_files, funtionals, kpoints, kshift, pw_cutoff, gk_cutoff, jsonparams)

    s.calculate(pos, lat, atomNames)
    del(s)

    

        