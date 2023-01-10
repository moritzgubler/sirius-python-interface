import sirius
import json
import numpy as np
from mpi4py import MPI


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

    mpiSize = None
    mpiRank = None
    isMaster = False
    isWorker = False

    energy_tol = 1e-6
    density_tol = 1e-6
    initial_tol = 1e-2
    num_dft_iter = 100
    write_dft_ground_state = False

    def __init__(self, pos: np.array, lat: np.array, atomNames, pp_files, functionals, kpoints: np.array
            , kshift: np.array, pw_cutoff: float, gk_cutoff: float, json_params :str, communicator: MPI.Comm = MPI.COMM_WORLD):
        self.communicator = communicator
        self.paramDict = json.loads(json_params)
        createChapters(self.paramDict)
        # self.paramDict['parameters']['ngridk'] = kpoints
        # self.paramDict['parameters']['kshift'] = kshift
        self.paramDict["parameters"]['pw_cutoff'] = np.sqrt(pw_cutoff)
        self.paramDict["parameters"]['gk_cutoff'] = np.sqrt(gk_cutoff)
        self.paramDict["parameters"]['xc_functionals'] = functionals

        if 'num_dft_iter' in self.paramDict['parameters']:
            self.num_dft_iter = self.paramDict['parameters']['num_dft_iter']
        # else:
        #     self.paramDict['parameters']['num_dft_iter'] = self.num_dft_iter
        if 'density_tol' in self.paramDict['parameters']:
            self.density_tol = self.paramDict['parameters']['density_tol']
        # else:
        #     self.paramDict['parameters']['density_tol'] = self.density_tol
        if 'energy_tol' in self.paramDict['parameters']:
            self.energy_tol = self.paramDict['parameters']['energy_tol']
        # else:
        #     self.paramDict['parameters']['energy_tol'] = self.energy_tol
        if 'energy_tolerance' in self.paramDict['iterative_solver']:
            self.initial_tol = self.paramDict['iterative_solver']['energy_tolerance']

        self.mpiSize = communicator.Get_size()
        self.mpiRank = communicator.Get_rank()
        if self.mpiSize > 0 and self.mpiRank == 0:
            self.isMaster = True
        if self.mpiSize > 0 and self.mpiRank > 0:
            self.isWorker = True

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
        if self.isWorker:
            self.worker_loop()

    def worker_loop(self):
        while True:
            print('worker receiving message', self.mpiRank)
            messageTag, data = self.communicator.bcast(None)
            print('message received ', messageTag)
            if messageTag == 'findGroundState':
                print('startcalonslave')
                self.findGroundState(*data)
            elif messageTag == 'energy':
                self.getEnergy()
            elif messageTag == 'forces':
                self.getForces()
            elif messageTag == 'stress':
                self.getStress()
            elif messageTag == 'exit':
                break
        quit()

    def exit(self):
        self.communicator.bcast(('exit', 0))

    def findGroundState(self, pos, lat):
        print('start find ',self.mpiRank)

        if self.isMaster:
            print('send starting signal from master')
            self.communicator.bcast(('findGroundState', [pos, lat]))
            # self.communicator.bsend(('findGroundState', [pos, lat]))
            print('sending done')

        self.context.unit_cell().set_lattice_vectors(lat[0, :], lat[1,:], lat[2, :])
        for i, p in enumerate(pos):
            self.context.unit_cell().atom(i).set_position(p)
        self.dft.update()
        self.dftRresult = self.dft.find(self.density_tol, self.energy_tol, self.initial_tol, self.num_dft_iter, self.write_dft_ground_state)
        if not self.dftRresult['converged']:
            print("dft calculation did not converge. Don't trust the results and increase num_dft_iter!")
        if self.dftRresult['rho_min'] < 0:
            print("Converged charge density has negative values. Don't trust the result")
        print('find done', self.mpiRank)

    def getEnergy(self):
        if self.isMaster:
            self.communicator.bcast(('energy', 0))
        return self.dftRresult['energy']['total'] + self.dftRresult['energy']['scf_correction']

    def getForces(self):
        if self.isMaster:
            self.communicator.bcast(('forces', 0))
        return np.array(self.dft.forces().calc_forces_total())

    def getStress(self):
        if self.isMaster:
            self.communicator.bcast(('stress', 0))
        return np.array(self.dft.stress().calc_stress_total())

    def getEnergyForcesStress(self, pos, lat):
        self.findGroundState(pos, lat)
        return self.getEnergy(), self.getForces() , self.getStress() 


def createChapters(json_params):
    chapters = ['mixer', 'settings', 'unit_cell', 'iterative_solver', 'control', 'parameters', 'nlcg', 'hubbard']
    for chap in chapters:
        if chap not in json_params:
            json_params[chap] = {}


if __name__ == '__main__':

    # comm = MPI.COMM_WORLD

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
    print('before')
    s = siriusInterface(pos, lat, atomNames, pp_files, funtionals, kpoints, kshift, pw_cutoff, gk_cutoff, jsonparams)
    print('after')
    e, f, stress = s.getEnergyForcesStress(pos, lat)

    print(e, np.linalg.norm(f))
    pos[0,:] = pos[0,:] + 0.1
    e, f, stress = s.getEnergyForcesStress(pos, lat)

    print(e, np.linalg.norm(f))

    s.exit()

    del(s)

    

        