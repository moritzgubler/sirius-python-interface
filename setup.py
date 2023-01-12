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
                      ],

    # classifiers=[
    #     'Development Status :: 1 - Planning',
    #     'Intended Audience :: Science/Research',
    #     'License :: OSI Approved :: BSD License',  
    #     'Operating System :: POSIX :: Linux',        
    #     'Programming Language :: Python :: 2',
    #     'Programming Language :: Python :: 2.7',
    #     'Programming Language :: Python :: 3',
    #     'Programming Language :: Python :: 3.4',
    #     'Programming Language :: Python :: 3.5',
    # ],
)
