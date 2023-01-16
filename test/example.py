import sirius
import json
import numpy

def make_new_ctx(pw_cutoff, gk_cutoff):
# lattice vectors
    a0 = 5.31148326730872
    lat = numpy.array(
        [[0.0, a0, a0],
         [a0, 0.0, a0],
         [a0, a0, 0.0]
    ])
# basic input parameters
    inp={
        "parameters" : {
            "xc_functionals" : ["XC_LDA_X", "XC_LDA_C_PZ"],
            "electronic_structure_method" : "pseudopotential",
            "pw_cutoff" : pw_cutoff,
            "gk_cutoff" : gk_cutoff
        },
        "control" : {
            "verbosity" : 0
        }
    }
# create simulation context
    ctx = sirius.Simulation_context(json.dumps(inp))
# set lattice vectors
    ctx.unit_cell().set_lattice_vectors(*lat)
# add atom type
    ctx.unit_cell().add_atom_type('Si','Si.json')
# add atoms
    ctx.unit_cell().add_atom('Si', [0.0,0.0,0.0])
    ctx.unit_cell().add_atom('Si', [0.25,0.25,0.25])
# intialize and return simulation context
    ctx.initialize()
    return ctx

def main():
    pw_cutoff = 20 # in a.u.^-1
    gk_cutoff = 8 # in a.u.^-1
    ctx = make_new_ctx(pw_cutoff, gk_cutoff)
    k = 2
    kgrid = sirius.K_point_set(ctx, [k,k,k], [1,1,1], True)
    dft = sirius.DFT_ground_state(kgrid)
    dft.initial_state()
    result = dft.find(1e-6, 1e-6, 1e-2, 100, False)
    # print(json.dumps(result, indent=2))


    # Extracting stress is working: 
    stressSirius = dft.stress()
    print('type of siriusForces', type(stressSirius))
    stress = stressSirius.calc_stress_total()
    print('stress', stress)

    # Now I try to extract forces
    siriusForces = dft.forces()
    print('type of siriusForces', type(siriusForces))

    # Now I try to get the forces from ther sirius.Force object:
    # none of those work, all return a sddk::mdarray<double, 2> object which does not seem to be python compatible
    # forces = siriusForces.total
    # forces = siriusForces.total()
    forces = numpy.array(siriusForces.calc_forces_total())
    print(forces)

    print('ef', kgrid.energy_fermi(), 'bandgap', kgrid.band_gap())

    # density = dft.density()
    # print(type(density))
    # print(density)

    # dm = density.density_matrix
    # d1 = density.generate(kgrid)

    # print('d1', type(d1), d1)

    # # print(dm.shape)

    # # for i in range(dm.shape[0]):
    # #     print(dm[i, i, :, :])

if __name__ == "__main__":
    main()