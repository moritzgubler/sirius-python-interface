from setuptools import setup

setup(
    name='sirius_ase',
    version='1.0.0',    
    description='A example Python package',
    url='https://github.com/moritzgubler/sirius-python-interface',
    author='Moritz Guber',
    author_email='moritz.gubler@e.email',
    license='BSD 2-clause',
    packages=['sirius_ase'],
    install_requires=['mpi4py',
                      'numpy',
                      'ase',         
                      'sqnm @ git+https://github.com/moritzgubler/vc-sqnm#egg=sqnm&subdirectory=src/python/',
                      ],
    entry_points={
      'console_scripts': [
        'SiriusSinglePoint=sirius_ase.ase_simulation:entry',
        'SiriusGeometryOpt=sirius_ase.ase_simulation:geopt'
      ]
    }
)
