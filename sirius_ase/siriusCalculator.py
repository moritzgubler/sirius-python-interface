from ase.calculators.calculator import Calculator, all_changes
from ase import atoms
import numpy as np
from mpi4py import MPI
import sirius_ase.sirius_interface as sirius_interface
from ase import units

class SIRIUS(Calculator):
    """ASE Calculator interface to the SIRIUS electronic structure library.

    Wraps SIRIUS DFT calculations and exposes them through the standard ASE
    Calculator API.  Supports MPI parallelization transparently: after
    initialization only rank 0 returns; worker ranks block inside a message
    loop until :meth:`close` is called.

    Implemented properties (all in ASE units):

    - ``energy``      : total DFT energy in eV (includes p*V if pressure is set)
    - ``forces``      : atomic forces in eV/Angstrom, shape (N, 3)
    - ``stress``      : Voigt stress vector in eV/Angstrom^3, shape (6,)
    - ``bandgap``     : electronic band gap in eV
    - ``fermienergy`` : Fermi energy in eV

    Parameters
    ----------
    atom : ase.Atoms
        Initial atomic structure.
    pp_files : dict[str, str]
        Mapping from element symbol to pseudopotential file path (JSON format).
        Convert pseudopotentials with the SIRIUS ``upf_to_json`` utility or
        download them from https://github.com/electronic-structure/species.
    functionals : list[str]
        Exchange-correlation functional codes using libxc naming convention,
        e.g. ``["XC_GGA_X_PBE", "XC_GGA_C_PBE"]``.
    kpoints : numpy.ndarray, shape (3,)
        Number of k-points along each reciprocal lattice direction.
    kshift : numpy.ndarray, shape (3,)
        K-point grid shift: 0 for no shift, 1 for a shift of 0.5/N_k in that
        direction.
    pw_cutoff : float
        Density/potential plane-wave cutoff in Ry (SIRIUS: ``pw_cutoff``;
        QE equivalent: ``ecutrho``).
    gk_cutoff : float
        Wavefunction |G+k| cutoff in Ry (SIRIUS: ``gk_cutoff``;
        QE equivalent: ``ecutwfc``).
    json_params : dict
        Additional SIRIUS parameters.  At minimum the ``"mixer"``,
        ``"control"``, and ``"parameters"`` sections should be provided.
        For the full parameter schema see
        https://github.com/electronic-structure/SIRIUS/blob/develop/src/context/input_schema.json
    pressure_giga_pascale : float, optional
        External hydrostatic pressure in GPa.  Adds p*V to the energy and
        the pressure to the diagonal stress components.  Default is 0.0.
    communicator : mpi4py.MPI.Comm, optional
        MPI communicator to use.  Defaults to ``MPI.COMM_WORLD``.  Can be a
        group communicator for embedding in a larger parallel job.
    """

    implemented_properties = ['energy', 'forces', 'stress', 'bandgap', 'fermienergy']
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
        """Run a SIRIUS DFT calculation.

        Called automatically by the ASE machinery when a property is requested.
        Runs the SCF ground-state search and caches energy, forces, stress,
        band gap, and Fermi energy in ``self.results``.
        """
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


    def getBandGap(self):
        """Return the electronic band gap.

        Returns
        -------
        float
            Band gap in eV.
        """
        return self.get_property('bandgap')

    def getFermiEnergy(self):
        """Return the Fermi energy.

        Returns
        -------
        float
            Fermi energy in eV.
        """
        return self.get_property('fermienergy')


    def recalculateBasis(self, atoms: atoms.Atoms, kpoints: np.array = None,
                    kshift: np.array = None, pw_cutoff: float = None, gk_cutoff: float= None):
        """Reinitialize SIRIUS with a new structure or updated basis parameters.

        Use this when the lattice vectors change between structures in a batch
        calculation (e.g. when iterating over a trajectory with variable cell).
        If only atomic positions change, SIRIUS updates them in place without a
        full reinitialization.

        Parameters
        ----------
        atoms : ase.Atoms
            New atomic structure.
        kpoints : numpy.ndarray, shape (3,), optional
            New k-point grid.  If None, the current grid is kept.
        kshift : numpy.ndarray, shape (3,), optional
            New k-point shift.  If None, the current shift is kept.
        pw_cutoff : float, optional
            New density/potential cutoff in Ry (QE: ``ecutrho``).
            If ``None``, the current value is kept.
        gk_cutoff : float, optional
            New wavefunction |G+k| cutoff in Ry (QE: ``ecutwfc``).
            If ``None``, the current value is kept.
        """
        paramdict= {
            'atomNames': atoms.get_chemical_symbols(),
            'pos': atoms.get_scaled_positions(wrap = False),
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
        self.reset()

    def close(self):
        """Shut down SIRIUS and release MPI worker processes.

        Must be called when the calculator is no longer needed.  Signals all
        worker ranks to exit their message loop and then frees the underlying
        SIRIUS objects.  After calling this method the calculator cannot be
        used again.
        """
        self.siriusInterface.exit()
        del(self.siriusInterface)
