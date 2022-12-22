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
    dftRresult = {}

    energy_tol = 1e-6
    density_tol = 1e-6
    initial_tol = 1e-2
    num_dft_iter = 100
    write_dft_ground_state = False

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

    def findGroundState(self, pos, lat):
        self.context.unit_cell().set_lattice_vectors(lat[0, :], lat[1,:], lat[2, :])
        for i, p in enumerate(pos):
            print(i, p)
            self.context.unit_cell().atom(i).set_position(p)
        self.dft.update()
        self.dftRresult = self.dft.find(self.density_tol, self.energy_tol, self.initial_tol, self.num_dft_iter, self.write_dft_ground_state)
        if not self.dftRresult['converged']:
            print("dft calculation did not converge. Don't trust the results and increase num_dft_iter!")
        if self.dftRresult['rho_min'] < 0:
            print("Converged charge density has negative values. Don't trust the result")

    def getEnergy(self, pos, lat):
       self.findGroundState(pos, lat)
       return self.dftRresult['energy']['total'] + self.dftRresult['energy']['scf_correction']

    def getForces(self):
        return self.dft.forces()

    def getStress(self):
        return self.dft.stress().calc_stress_total()

    def getEnergyForcesStress(self, pos, lat):
        return self.getEnergy(pos, lat), self.getForces() , self.getStress() 




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

    e, f, stress = s.getEnergyForcesStress(pos, lat)

    print(e, f.total)
    # pos[0,:] = pos[0,:] + 0.05
    # e, f, stress = s.getEnergyForcesStress(pos, lat)

    # print(e, f.total, stress.total)
    del(s)

    

        