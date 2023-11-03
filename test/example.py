import sirius
import json
import numpy

def make_new_ctx(pw_cutoff, gk_cutoff):
    a0 = 5.31148326730872
    lat = numpy.array(
        [[0.0, a0, a0],
         [a0, 0.0, a0],
         [a0, a0, 0.0]
    ])
    inp={
        "parameters" : {
            "xc_functionals" : ["XC_GGA_X_PBE", "XC_GGA_C_PBE"],
            "electronic_structure_method" : "pseudopotential",
            "pw_cutoff" : pw_cutoff,
            "gk_cutoff" : gk_cutoff
        },
        "control" : {
            "verbosity" : 0
        }
    }
    ctx = sirius.Simulation_context(json.dumps(inp))
    ctx.unit_cell().set_lattice_vectors(*lat)
    ctx.unit_cell().add_atom_type('Si','Si.json')
    ctx.unit_cell().add_atom('Si', [0.0,0.0,0.0])
    ctx.unit_cell().add_atom('Si', [0.25,0.25,0.25])
    ctx.initialize()
    return ctx

def main():
    pw_cutoff = 20 # in a.u.^-1
    gk_cutoff = 8 # in a.u.^-1
    ctx = make_new_ctx(pw_cutoff, gk_cutoff)
    k = 3
    kgrid = sirius.K_point_set(ctx, [k,k,k], [0,0,0], True)
    dft = sirius.DFT_ground_state(kgrid)
    dft.initial_state()
    dft.find(1e-6, 1e-6, 1e-2, 100, False)

    # now I can get the charge density in fourier space
    density = dft.density()


    print('reciprocal space density: density.rho')   
    print(density.rho)

    print('density.f_rg(0). I think this is the real space density.')
    print(density.f_rg(0))

    import matplotlib.pyplot as plt

    plt.title('density.f_rg')
    plt.plot(density.f_rg(0))
    plt.show()

    # transform density to real space:
    density.fft_transform(1)
    fft_grid = ctx.fft_grid()
    print(fft_grid)
    print(fft_grid.shape)



if __name__ == "__main__":
    main()