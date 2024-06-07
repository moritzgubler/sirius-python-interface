import sirius
import json
import numpy as np
from mpi4py import MPI
import sirius_ase.k_grid
import logging
import time


class siriusInterface:

    communicator = None
    isMaster = False
    initPos = None
    initLat = None
    context = None
    paramDict = None
    k_point_set = None
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
    use_k_sym = True
    write_dft_ground_state = False
    jsonString = ''
    pp_files = {}
    atomNames = []
    initialPositions = None
    initialLattice = None
    kpoints = np.ones(3)
    kshift = np.zeros(3)

    first_eval = True
    sirius_communicator = None

    useCustomeMeshGenerator = True

    def __init__(self, pos: np.array, lat: np.array, atomNames: list, pp_files: dict, functionals, kpoints: np.array
            , kshift: np.array, pw_cutoff: float, gk_cutoff: float, json_params :dict, communicator: MPI.Comm = MPI.COMM_WORLD, returnWorkers: bool = False):

        self.pp_files = pp_files
        self.communicator = communicator
        self.sirius_communicator = sirius.make_sirius_comm(self.communicator)
        # self.paramDict = json.loads(json_params)
        self.paramDict = json_params
        self.atomNames = atomNames
        self.initialPositions = pos.copy()
        self.initialLattice = lat.copy()
        self.kpoints = kpoints
        self.kshift = kshift

        createChapters(self.paramDict)
        self.paramDict["parameters"]['pw_cutoff'] = np.sqrt(pw_cutoff)
        self.paramDict["parameters"]['gk_cutoff'] = np.sqrt(gk_cutoff)
        self.paramDict["parameters"]['xc_functionals'] = functionals

        self.mpiSize = communicator.Get_size()
        self.mpiRank = communicator.Get_rank()
        if self.mpiSize > 1 and self.mpiRank == 0:
            self.isMaster = True
        if self.mpiSize > 1 and self.mpiRank > 0:
            self.isWorker = True
        if returnWorkers:
            self.isMaster = False
            self.isWorker = False

        self.setDefaultParameters()
        self.jsonString = json.dumps(self.paramDict)

        self.createSiriusObjects(atomNames, pos, lat)
        if self.isWorker:
            self.worker_loop()

    def createSiriusObjects(self, atomNames: list, pos, lat):
        self.context = sirius.Simulation_context(self.jsonString, self.sirius_communicator)
        self.context.unit_cell().set_lattice_vectors(lat[0, :], lat[1,:], lat[2, :])

        for element in self.pp_files:
            if element in atomNames:
                self.context.unit_cell().add_atom_type(element, self.pp_files[element])
        for element in atomNames:
            if element not in self.pp_files:
                logging.error('Element has no corresponding pseudopotential file', element)
                quit()
        
        for i in range(pos.shape[0]):
            self.context.unit_cell().add_atom(atomNames[i], pos[i, :])

        self.context.initialize()

        kpointlist = sirius_ase.k_grid.createGridAndWeights(self.kpoints, self.kshift)

        if self.useCustomeMeshGenerator: # use own mesh
            self.k_point_set = sirius.K_point_set(self.context)
            for k, w in kpointlist:
                self.k_point_set.add_kpoint(np.array(k), w)
            self.k_point_set.initialize()
            if self.mpiRank == 0:
                logging.info('Number of k points: %d'%len(kpointlist))
                logging.debug("Kpoint, weight:")
                for k, w in kpointlist:
                    logging.debug(k, w)
        else: # create kgrid using constructor from sirius. 
            self.k_point_set = sirius.K_point_set(self.context, self.kpoints, self.kshift, self.use_k_sym)
        self.dft = sirius.DFT_ground_state(self.k_point_set)
        self.dft.initial_state()

    def setDefaultParameters(self):
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

        if not 'electronic_structure_method' in self.paramDict['parameters']:
            self.paramDict['parameters']['electronic_structure_method'] = 'pseudopotential'
        
        if 'use_symmetry' in self.paramDict['parameters']:
            self.use_k_sym = self.paramDict['parameters']['use_symmetry']


    def worker_loop(self):
        messageTag = 'asdf'
        data = 0
        message = (messageTag, data)
        while True:
            message = self.communicator.bcast((message, data))
            messageTag = message[0]
            if str(messageTag) == 'findGroundState':
                data = message[1]
                self.findGroundState(*data)
            elif str(messageTag) == 'energy':
                self.getEnergy()
            elif str(messageTag) == 'forces':
                self.getForces()
            elif str(messageTag) == 'stress':
                self.getStress()
            elif str(messageTag) == 'bandgap':
                self.getBandGap()
            elif str(messageTag) == 'fermienergy':
                self.getFermiEnergy()
            elif str(messageTag) == 'resetSirius':
                data = message[1]
                self.resetSirius(*data)
            elif str(messageTag) == 'exit':
                del(self)
                quit()

    def exit(self):
        if self.isMaster:
            self.communicator.bcast(('exit', 0))

    def findGroundState(self, pos, lat):
        # print(pos, lat)
        if self.isMaster:
            self.communicator.bcast(('findGroundState', [pos, lat]))

        tester = np.linalg.norm(self.initialLattice - lat, axis=1) < 0.01 * np.linalg.norm(self.initialLattice, axis=1)

        if self.first_eval:
            self.first_eval = False
        else:
            self.updateSirius(pos, lat)

        self.dftRresult = self.dft.find(self.density_tol, self.energy_tol, self.initial_tol, self.num_dft_iter, self.write_dft_ground_state)
        if not self.dftRresult['converged']:
            print("dft calculation did not converge. Don't trust the results and increase num_dft_iter!")
        if self.dftRresult['rho_min'] < 0:
            print("Converged charge density has negative values. Don't trust the result")


    def resetSirius(self, atomNames: list, pos, lat, kpoints: np.array = None, 
                    kshift: np.array = None, pw_cutoff: float = None, gk_cutoff: float= None):
        if kpoints is not None:
            self.kpoints = kpoints
        elif not self.isMaster:
            print("kpoints argument cannot be empty if not master process.")
        if kshift is not None:
            self.kshift = kshift
        elif not self.isMaster:
            print("kshift argument cannot be empty if not master process.")
        if pw_cutoff is not None:
            self.paramDict["parameters"]['pw_cutoff'] = np.sqrt(pw_cutoff)
        elif not self.isMaster:
            print("pw_cutoff argument cannot be empty if not master process.")
        if gk_cutoff is not None:
            self.paramDict["parameters"]['gk_cutoff'] = np.sqrt(gk_cutoff)
        elif not self.isMaster:
            print("gk_cutoff argument cannot be empty if not master process.")

        if self.isMaster:
            self.communicator.bcast(('resetSirius', [atomNames, pos, lat, self.kpoints, self.kshift,
                                                     self.paramDict["parameters"]['pw_cutoff'] ** 2,
                                                     self.paramDict["parameters"]['gk_cutoff'] ** 2]))
            print("Master sent resetSirius")
            print("current paramter dict (master) (pw and gk cutoff are square root of input value): ")
            print(self.paramDict)
            print("pos", pos)
            print("lat", lat)
            print('atomNames', atomNames)
        else:
            time.sleep(1)
            print("Worker received resetSirius")
            print("current paramter dict (worker) (pw and gk cutoff are square root of input value): ")
            print(self.paramDict)
            print("pos", pos)
            print("lat", lat)
            print('atomNames', atomNames)

        del(self.context)
        del(self.k_point_set)
        del(self.dft)
        self.atomNames = atomNames
        self.initialPositions = pos
        self.initialLattice = lat
        self.createSiriusObjects(atomNames, pos, lat)
        self.first_eval = True


    def updateSirius(self, pos, lat):
        self.context.unit_cell().set_lattice_vectors(lat[0, :], lat[1,:], lat[2, :])
        for i, p in enumerate(pos):
            self.context.unit_cell().atom(i).set_position(p)
        self.dft.update()


    def getEnergy(self):
        if self.isMaster:
            self.communicator.bcast(('energy', 0))
        return self.dftRresult['energy']['total'] + self.dftRresult['energy']['scf_correction']

    
    def getBandGap(self):
        if self.isMaster:
            self.communicator.bcast(('bandgap', 0))
        return self.k_point_set.band_gap()


    def getFermiEnergy(self):
        if self.isMaster:
            self.communicator.bcast(('fermienergy', 0))
        return self.k_point_set.energy_fermi()


    def getForces(self):
        if self.isMaster:
            self.communicator.bcast(('forces', 0))
        return np.array(self.dft.forces().calc_forces_total()).T

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
    s = siriusInterface(pos, lat, atomNames, pp_files, funtionals, kpoints, kshift, pw_cutoff, gk_cutoff, jsonparams)
    e, f, stress = s.getEnergyForcesStress(pos, lat)

    print(e, np.linalg.norm(f), f.shape)
    pos[0,:] = pos[0,:] + 0.1
    e, f, stress = s.getEnergyForcesStress(pos, lat)

    print(e, np.linalg.norm(f), f.shape)

    s.exit()

    del(s)

    

        