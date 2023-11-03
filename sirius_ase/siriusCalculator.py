from ase.calculators.calculator import Calculator, all_changes
from ase import atoms
import numpy as np
from mpi4py import MPI
import sirius_ase.sirius_interface as sirius_interface
from ase import units
import chargePartitioning.hirshfeldWeightFunction as hirshfeld
import time
from ase.neighborlist import NeighborList, NewPrimitiveNeighborList
from ase.neighborlist import natural_cutoffs

class SIRIUS(Calculator):
    
    implemented_properties = ['energy', 'forces', 'stress', 'bandgap', 'fermienergy', 'chargedensity', 'chargedensityandgrid', 'charges']
    default_parameters = {}
    nolabel = True
    siriusInterface = None

    def __init__(self, atom: atoms.Atom, pp_files, functionals, kpoints: np.array
            , kshift: np.array, pw_cutoff: float, gk_cutoff: float
            , json_params :dict, pressure_giga_pascale: float = 0.0, communicator: MPI.Comm = MPI.COMM_WORLD):

        super().__init__()
        self.siriusInterface = sirius_interface.siriusInterface(atom.get_scaled_positions(wrap=False),
            atom.get_cell(True) / units.Bohr, atom.get_chemical_symbols(), pp_files, functionals, 
            kpoints, kshift, pw_cutoff, gk_cutoff, json_params, communicator=communicator)

        self.pressure = pressure_giga_pascale * units.GPa


    def calculate(
        self,
        atoms: atoms.Atoms = None,
        properties = None,
        system_changes = all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties

        # print(system_changes, properties)
        super().calculate(atoms, properties, system_changes)

        if not system_changes == []:
            self.siriusInterface.findGroundState(atoms.get_scaled_positions(wrap = False), atoms.get_cell(True) / units.Bohr)

        if 'energy' in properties:
            self.results['energy'] = self.siriusInterface.getEnergy() * units.Hartree + self.pressure * atoms.get_volume()

        if 'forces' in properties:
            self.results['forces'] = self.siriusInterface.getForces() * (units.Hartree / units.Bohr)

        if 'stress' in properties:
            stress_sirius = self.siriusInterface.getStress() * ( units.Hartree / units.Bohr**3)
            stress_sirius = 0.5 * (stress_sirius + stress_sirius.T)
            self.results['stress'] = np.array([stress_sirius[0][0] + self.pressure, stress_sirius[1][1] + self.pressure
                , stress_sirius[2][2] + self.pressure, stress_sirius[1][2], stress_sirius[0][2], stress_sirius[0][1]])

        if 'bandgap' in properties:
            self.results['bandgap'] = self.siriusInterface.getBandGap() * units.Hartree

        if 'fermienergy' in properties:
            self.results['fermienergy'] = self.siriusInterface.getFermiEnergy() * units.Hartree

        if 'chargedensity' in properties:
            self.results['chargedensity'] = self.siriusInterface.getChargeDensity()

        if 'chargedensityandgrid' in properties:
            self.results['chargedensityandgrid'] = self.siriusInterface.getRealGrid(atoms.get_cell(True) / units.Bohr)

        if 'charges' in properties:

            def getNearestGridPoint(p, lattice, nx, ny, nz):
                p_frac = p @ np.linalg.inv(lattice)
                nx = round(p_frac[0])
                ny = round(p_frac[1])
                nz = round(p_frac[2])
                return np.array([nx, ny, nz])


            t1 = time.time()
            self.results['charges'] = []
            pos = atoms.get_positions() / units.Bohr
            lat = atoms.get_cell() / units.Bohr
            elements = atoms.get_chemical_symbols()

            ts = time.time()
            grid, rho, indices = self.siriusInterface.getRealGrid(lat)
            nx, ny, nz = indices
            te = time.time()
            print('grid time', te - ts)


            cutoff = 4
            lattice = atoms.get_cell()


            functionDict = hirshfeld.createDensityInterpolationDictionary(elements)
            ts = time.time()
            normalizer = hirshfeld.getNormalizer(grid.reshape((nx * ny * nz, 3)), pos, elements, functionDict, lat, 6.0)
            te = time.time()
            print('normalizer time', te - ts)

            dv = np.abs(np.linalg.det(lat)) / (nx * ny * nz)

            # build neighbourlist
            nl = NeighborList(natural_cutoffs(atoms, mult = 1.5), bothways=True)
            # nl.update([True, True, True], atoms.get_cell(), atoms.get_positions())
            nl.update(atoms)


            import matplotlib.pyplot as plt

            # plt.plot(normalizer[:nx])
            plt.plot(rho[np.mod(np.arange(2 * nz), nz), 0, 0])
            plt.show()
            # rho = np.ones(rho.shape)

            t2 = time.time()
            da = atoms.cell[0, :] / nx
            db = atoms.cell[1, :] / ny
            dc = atoms.cell[2, :] / nz

            g_cutoffs = hirshfeld.countGostCellsFractional(atoms.get_cell(), cutoff)
            pnx = round(2 * (np.ceil(g_cutoffs[0] * nx)) + 1)
            pny = round(2 * (np.ceil(g_cutoffs[1] * nx)) + 1)
            pnz = round(2 * (np.ceil(g_cutoffs[2] * nx)) + 1)


            print('pnx', pnx, pny, pnz)

            origin_grid = np.empty((pnz, pnx, pny, 3))

            for iz in range(nz):
                for iy in range(ny):
                    temp = iz * lattice[2, :] + iy * lattice[1, :]
                    origin_grid[iz, iy, :, 0] = temp[0] + np.linspace(0, lattice[0, 0], pnx, endpoint=False)
                    origin_grid[iz, iy, :, 1] = temp[1] + np.linspace(0, lattice[0, 1], pnx, endpoint=False)
                    origin_grid[iz, iy, :, 2] = temp[2] + np.linspace(0, lattice[0, 2], pnx, endpoint=False)

            

            for i in range(len(atoms)):
                indices, offsets = nl.get_neighbors(i)
                neighbour_positions = np.empty((len(indices), 3))
                ineighbour = 0
                for j, offset in zip(indices, offsets):
                    neighbour_positions[ineighbour, :] = atoms.positions[j, :] @ atoms.get_cell()
                    ineighbour += 1
                
                # print('neighbours', i, len(indices))
                # print(indices)
                # print(offsets)
                # print(neighbour_positions)

                weights = hirshfeld.partitioningWeights(grid.reshape((nz * ny * nx, 3)), pos[i, :], functionDict[elements[i]], normalizer, lat, 6.0)
                # plt.plot(weights[:nx])
                # plt.plot(normalizer[:nx])
                # plt.show()
                self.results['charges'].append(np.sum(rho.reshape(nx * ny * nz) * weights) * dv)
            
            t3 = time.time()
            print('hf timings setup, hf integrals, ratio %f %f %f'%(t2 - t1, t3 - t2, (t3 - t2) / (t3 - t1)))

    def getBandGap(self):
        return self.get_property('bandgap')

    def getFermiEnergy(self):
        return self.get_property('fermienergy')
    
    def getChargeDensity(self):
        return self.get_property('chargedensity')

    def getChargeDensityAndGrid(self):
        return self.get_property('chargedensityandgrid')

    def recalculateBasis(self, atoms: atoms.Atoms, kpoints: np.array = None, 
                    kshift: np.array = None, pw_cutoff: float = None, gk_cutoff: float= None):
        paramdict= {
            'atomNames': atoms.get_chemical_symbols(),
            'pos': atoms.get_scaled_positions(wrap = True),
            'lat': atoms.get_cell(True) / units.Bohr
        }
        if kpoints is not None:
            paramdict['kpoints'] = kpoints
        if kshift is not None:
            paramdict['kshift'] = kshift
        if pw_cutoff is not None:
            paramdict['pw_cutoff'] = pw_cutoff
        if gk_cutoff is not None:
            paramdict['gk_cutoff'] = gk_cutoff
        self.siriusInterface.resetSirius(**paramdict)

    def close(self):
        self.siriusInterface.exit()
        del(self.siriusInterface)
