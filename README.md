
[![DOI](https://zenodo.org/badge/791746137.svg)](https://zenodo.org/doi/10.5281/zenodo.11072984)
# Sources for: Path integral spin dynamics for quantum paramagnets

This repository contains source code which implements a path integral approach to calculate the thermodynamics of quantum spin for a quantum paramagnet with easy axis along the applied field direction and with compressive/tensile stress along the same direction. It contains functions for analytic solutions and a numerical approach based on atomistic spin dynamics. 

## Authors

Thomas Nussle, *School of Physics and Astronomy, University of Leeds, Leeds, LS2 9JT, United Kingdom*, http://orcid.org/0000-0001-9783-4287

Stam Nicolis, *Institut Denis Poisson, Université de Tours, Université d'Orléans, CNRS (UMR7013), Parc de Grandmont, F-37200, Tours, France*, http://orcid.org/0000-0002-4142-5043

Joseph Barker, *School of Physics and Astronomy, University of Leeds, Leeds, LS2 9JT, United Kingdom*, http://orcid.org/0000-0003-4843-5516

Pascal Thibaudeau, *CEA/DAM le Ripault, F-37260, Monts, France*, https://orcid.org/0000-0002-0374-5038

## Description

This is a simple code for atomistic simulations for a single spin for a quantum paramagnet with easy axis along the applied field direction (arbitrary choice of z-axis here) and with compressive/tensile stress along the same direction. It computes both the classical limit and quantum corrections of the thermal expectation value of the z-component of the spin. 

There are two approximation schemes included: The first one is a "high" temperature approximation, and the second one is a exact method which directly takes a log function as the Hamiltonian.

Also included are exact results computed from the partition function for the quantum case.

## File descriptions

### ./

**environment.yml**
Conda environment file to reproduce the python environment for executing the calculations.

**LICENSE**
MIT License file.

**Makefile**
Makefile to run all calculation and produce the data and figures in the figures/ folder. 

**README.md**
This readme file.

### ./python/

This folder contains python code and scripts to generate the figures.

**python/analytic.py** 
Defines python functions for the analytic equations which appear in the figures. The mathematical expression for each function are written as docstrings in the python functions. 

**python/asd.py** 
Defines python functions for atomistic spin dynamics calculations including numerical integration methods, effective fields and stochastic fields. 

**python/pisd.py**
An executable python program for running general path integral spin dynamics calculations using the functions in python/asd.py.

**python/figure_{a,b,c,d}.py**
Calculates and plots the quantum analytic results and approximate results for the classical limit and first correction using the high temperature approximation of the atomistic approximation method. Expectation value of Sz for s={1/2, 1, 3/2, 2} as a function of temperature. 

**python/figure2_{a,b,c,d}.py**
Calculates and plots the quantum analytic results and approximate results for the "exact" quantum correction using the logarithm form of the Hamiltonian of the atomistic approximation method. Expectation value of Sz for s={1/2, 1, 3/2, 2} as a function of temperature. 

### ./figures/

Output generated from the python/figure*X*.py scripts. 

**figures/figure*X*.pdf**
PDF figure used in the manuscript.

**figures/figure*X*.log**
Output logged from executing the figure script.

**figures/figure*X*_data_**
Folder containing the raw data generated by the script with the filenames representing `<method>_<approximation>_<spin>.tsv`

### ./resources/

**aps-paper.mplstyle**
matplotlib style file for plotting figures with fonts and font sizes similar to the American Physical Society typesetting. 

## Computational Environment

All calculations were performed on a Mac Studio (Apple M2 Ultra, 64GB RAM, Model Identifier: Mac14,14) running macOS version 14.4.1 (Sonoma). Python and associated packages were installed using conda. The installed package versions were:

 - python=3.10.12
 - matplotlib=3.6.2
 - numba=0.59.0
 - numpy=1.23.5
 - scipy=1.9.3
 - sympy=1.12.1rc1
  
The conda environment can be recreated using the `environment.yml` file on .

<details>
  <summary>Click here for the complete list of package installed by conda including all dependencies and hashes</summary>
```text
# Name                    Version                   Build  Channel
brotli                    1.1.0                hb547adb_0    conda-forge
brotli-bin                1.1.0                hb547adb_0    conda-forge
bzip2                     1.0.8                h3422bc3_4    conda-forge
ca-certificates           2023.7.22            hf0a4a13_0    conda-forge
certifi                   2023.7.22          pyhd8ed1ab_0    conda-forge
contourpy                 1.1.1           py310h38f39d4_0    conda-forge
cycler                    0.11.0             pyhd8ed1ab_0    conda-forge
fonttools                 4.42.1          py310h2aa6e3c_0    conda-forge
freetype                  2.12.1               hadb7bae_2    conda-forge
gmp                       6.2.1                h9f76cd9_0    conda-forge
gmpy2                     2.1.2           py310h2e6cad2_1    conda-forge
kiwisolver                1.4.5           py310h38f39d4_1    conda-forge
lcms2                     2.15                 h40e5a24_2    conda-forge
lerc                      4.0.0                h9a09cb3_0    conda-forge
libblas                   3.9.0           18_osxarm64_openblas    conda-forge
libbrotlicommon           1.1.0                hb547adb_0    conda-forge
libbrotlidec              1.1.0                hb547adb_0    conda-forge
libbrotlienc              1.1.0                hb547adb_0    conda-forge
libcblas                  3.9.0           18_osxarm64_openblas    conda-forge
libcxx                    16.0.6               h4653b0c_0    conda-forge
libdeflate                1.19                 hb547adb_0    conda-forge
libffi                    3.4.2                h3422bc3_5    conda-forge
libgfortran               5.0.0           13_2_0_hd922786_1    conda-forge
libgfortran5              13.2.0               hf226fd6_1    conda-forge
libjpeg-turbo             2.1.5.1              hb547adb_1    conda-forge
liblapack                 3.9.0           18_osxarm64_openblas    conda-forge
libllvm11                 11.1.0               hfa12f05_5    conda-forge
libopenblas               0.3.24          openmp_hd76b1f2_0    conda-forge
libpng                    1.6.39               h76d750c_0    conda-forge
libsqlite                 3.43.0               hb31c410_0    conda-forge
libtiff                   4.6.0                h77c4dce_1    conda-forge
libwebp-base              1.3.2                hb547adb_0    conda-forge
libxcb                    1.15                 hf346824_0    conda-forge
libzlib                   1.2.13               h53f4e23_5    conda-forge
llvm-openmp               16.0.6               h1c12783_0    conda-forge
llvmlite                  0.42.0                   pypi_0    pypi
matplotlib                3.6.2           py310hb6292c7_0    conda-forge
matplotlib-base           3.6.2           py310h78c5c2f_0    conda-forge
mpc                       1.3.1                h91ba8db_0    conda-forge
mpfr                      4.2.0                he09a6ba_0    conda-forge
mpmath                    1.3.0              pyhd8ed1ab_0    conda-forge
munkres                   1.1.4              pyh9f0ad1d_0    conda-forge
ncurses                   6.4                  h7ea286d_0    conda-forge
numba                     0.59.0                   pypi_0    pypi
numpy                     1.23.5          py310h5d7c261_0    conda-forge
openjpeg                  2.5.0                h4c1507b_3    conda-forge
openssl                   3.1.3                h53f4e23_0    conda-forge
packaging                 23.1               pyhd8ed1ab_0    conda-forge
pillow                    10.0.1          py310hadb9e77_1    conda-forge
pip                       23.2.1             pyhd8ed1ab_0    conda-forge
pthread-stubs             0.4               h27ca646_1001    conda-forge
pyparsing                 3.1.1              pyhd8ed1ab_0    conda-forge
python                    3.10.12         h01493a6_0_cpython    conda-forge
python-dateutil           2.8.2              pyhd8ed1ab_0    conda-forge
python_abi                3.10                    4_cp310    conda-forge
readline                  8.2                  h92ec313_1    conda-forge
scipy                     1.9.3           py310ha0d8a01_2    conda-forge
setuptools                68.2.2             pyhd8ed1ab_0    conda-forge
six                       1.16.0             pyh6c4a22f_0    conda-forge
sympy                     1.12.1rc1                pypi_0    pypi
tk                        8.6.12               he1e0b03_0    conda-forge
tornado                   6.3.3           py310h2aa6e3c_1    conda-forge
tzdata                    2023c                h71feb2d_0    conda-forge
unicodedata2              15.0.0          py310h8e9501a_0    conda-forge
wheel                     0.41.2             pyhd8ed1ab_0    conda-forge
xorg-libxau               1.0.11               hb547adb_0    conda-forge
xorg-libxdmcp             1.1.3                h27ca646_0    conda-forge
xz                        5.2.6                h57fd34a_0    conda-forge
zstd                      1.5.5                h4f39d0f_0    conda-forge
```
</details>

## Reproduction

 The `make` build tool can be used to execute the Makefile and re-produce all of the figures. The steps to reproduce are:

```bash
conda env create -f environment.yml
conda activate quantum_spin_dynamics
make clean
make
```

Note that the the atomistic spin dynamics are stochastic so the results will differ slightly due to random seeding.

### Runtimes

- python/figure_a.py: 1529.213 (s)
- python/figure_b.py: 1401.299 (s)
- python/figure_c.py: 1511.157 (s)
- python/figure_d.py: 1520.961 (s)
- python/figure2_a.py: 702.361 (s)
- python/figure2_b.py: 783.914 (s)
- python/figure2_c.py: 750.803 (s)
- python/figure2_d.py: 793.125 (s)

## Code use for general calculations

The python/pisd.py script can be used to generate data with different integration methods, approximations and spin values.

Using the provided environment.yml file, create a conda environment in your terminal:

```bash
conda env create -f environment.yml
conda activate quantum_spin_dynamics
```

Then run the python/pisd.py script

```bash
python python/pisd.py <options>
```

where the options available are

```text
usage: pisd.py [-h] [--integrator {runge-kutta-4,symplectic}] --approximation {classical-limit, quantum-approximation, quantum-exact} --spin SPIN --field FIELD --stress STRESS --anisotropy ANISOTROPY

Simulation parameters from command line.

options:
  -h, --help            show this help message and exit
  --integrator {runge-kutta-4,symplectic}
                        Numerical integration method for solving the spin dynamics
  --approximation {classical-limit,quantum-approximation, quantum-exact}
                            Approximation scheme to use
  --order ORDER             Order of approximation. These are the quantum correction terms to be taken into account for the approximation (3 is the first quantum correction and this is used neither for the classical limit where the order is 2 nor for the "exact" quantum field method)
  --spin SPIN               Quantum spin value (should normally be an integer multiple of 1/2)
  --field FIELD             Z-component of magnetic field (in Tesla)
  --stress STRESS           Applied stress energy (ƛσ in Joules)
  --anisotropy ANISOTROPY   Uniaxial anisotropy energy (K in Joules)
```
  
### Additional variables

Depending on computational resources and specific system, one can change some parameters 
in the python/pisd.py script (or examples scripts as well):
- the value of the gilbert damping `alpha` (default is 0.5)
- the range and increments in temperature for atomistic simulations by 
changing `np.linspace(<starting-temperature>, <final-temperature>, 
<number-of-temperatures-in-range>)`
- the initial spin orientation `s0` (must be of unit norm and different 
from `np.array([0, 0, 1])`
- the equilibration period `equilibration_time`, the computation time `production_time`
and the integration time step `time_step` for each stochastic realisation
- the number of stochastic realisations `num_realisation` of the noise over
which to average the time average

## Notes

The numba package is used for just in time compilation which greatly reduces the calculation time. In principle the `@njit` statements can be removed from all code if numba is not supported on a given platform, but the calculation time will be extremely long.

In python/analytic.py, sympy is used to generate higher order terms for approximations symbolically before being compiled. Without this, each order of approximation would need to be hard coded.

This work is an extension of the method for a single spin in a constant magnetic field from: Thomas Nussle, Stam Nicolis and Joseph Barker, "Numerical simulations of a spin dynamics model based on a path integral approach", [Phys. Rev. Research 5, 043075 (2023)](https://doi.org/10.1103/PhysRevResearch.5.043075).

By default, the atomistic spin dynamics uses a symplectic integrator described in: Pascal Thibaudeau and David Beaujouan, "Thermostatting the atomic spin dynamics from controlled demons", [Phys. A: Stat. Mech. its Appl. 391, 1963–1971 (2012)](http://dx.doi.org/10.1016/j.physa.2011.11.030).

## Grant Acknowledgement

This software was produced with funding from the UKRI Engineering and Physical Sciences Research Council [grant number EP/V037935/1 - *Path Integral Quantum Spin Dynamics*] and support from the Royal Society through a University Research Fellowship.
