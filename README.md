
[![DOI](https://zenodo.org/badge/{github_id}.svg)](https://zenodo.org/badge/latestdoi/{github_id})

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

**python/figure3.py**
Calculates and plots the thermal fluctuations of the quantum system from the partition function as sqrt(<S_z^2>-<S_z>^2)/<S_z> to show which regime has classical thermal behaviour and which regime has quantum thermal behaviour. Expectation value of Sz for s=1 as a function of temperature. 

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

All calculations were performed on a MacBook Pro (Quad-Core Intel Core i5 2GHz, 16GB RAM, Model Identifier: MacBookPro16,2) running macOS version 11.7.4 (Big Sur). Python and associated packages were installed using conda. The installed package versions were:

 - python=3.10
 - matplotlib=3.6.2
 - numba=0.56.4
 - numpy=1.23.5
 - scipy=1.9.3
 - sympy=1.11.1
  
The conda environment can be recreated using the `environment.yml` file on .

<details>
  <summary>Click here for the complete list of package installed by conda including all dependencies and hashes</summary>
```text
# Name                    Version                   Build  Channel
blas                      1.0                         mkl  
bottleneck                1.3.5            py38h67323c0_0  
brotli                    1.0.9                hca72f7f_7  
brotli-bin                1.0.9                hca72f7f_7  
ca-certificates           2023.01.10           hecd8cb5_0  
certifi                   2022.12.7        py38hecd8cb5_0  
contourpy                 1.0.5            py38haf03e11_0  
cycler                    0.11.0             pyhd3eb1b0_0  
fftw                      3.3.9                h9ed2024_1  
fonttools                 4.25.0             pyhd3eb1b0_0  
freetype                  2.12.1               hd8bbffd_0  
giflib                    5.2.1                haf1e3a3_0  
gmp                       6.2.1                he9d5cce_3  
gmpy2                     2.1.2            py38hd5de756_0  
importlib-metadata        4.11.3           py38hecd8cb5_0  
importlib_metadata        4.11.3               hd3eb1b0_0  
intel-openmp              2021.4.0          hecd8cb5_3538  
jpeg                      9e                   hca72f7f_0  
kiwisolver                1.4.4            py38hcec6c5f_0  
lcms2                     2.12                 hf1fd2bf_0  
lerc                      3.0                  he9d5cce_0  
libbrotlicommon           1.0.9                hca72f7f_7  
libbrotlidec              1.0.9                hca72f7f_7  
libbrotlienc              1.0.9                hca72f7f_7  
libcxx                    14.0.6               h9765a3e_0  
libdeflate                1.8                  h9ed2024_5  
libffi                    3.4.2                hecd8cb5_6  
libgfortran               5.0.0           11_3_0_hecd8cb5_28  
libgfortran5              11.3.0              h9dfd629_28  
libllvm11                 11.1.0               h46f1229_6  
libpng                    1.6.37               ha441bb4_0  
libtiff                   4.5.0                h2cd0358_0  
libwebp                   1.2.4                h56c3ce4_0  
libwebp-base              1.2.4                hca72f7f_0  
llvm-openmp               14.0.6               h0dcd299_0  
llvmlite                  0.39.1           py38h8346a28_0  
lz4-c                     1.9.4                hcec6c5f_0  
matplotlib                3.6.2            py38hecd8cb5_0  
matplotlib-base           3.6.2            py38h220de94_0  
mkl                       2021.4.0           hecd8cb5_637  
mkl-service               2.4.0            py38h9ed2024_0  
mkl_fft                   1.3.1            py38h4ab4a9b_0  
mkl_random                1.2.2            py38hb2f4e1b_0  
mpc                       1.1.0                h6ef4df4_1  
mpfr                      4.0.2                h9066e36_1  
mpmath                    1.2.1            py38hecd8cb5_0  
munkres                   1.1.4                      py_0  
ncurses                   6.3                  hca72f7f_3  
numba                     0.56.4           py38h07fba90_0  
numexpr                   2.8.4            py38he696674_0  
numpy                     1.23.5           py38he696674_0  
numpy-base                1.23.5           py38h9cd3388_0  
openssl                   1.1.1t               hca72f7f_0  
packaging                 22.0             py38hecd8cb5_0  
pandas                    1.5.2            py38h07fba90_0  
pillow                    9.3.0            py38h81888ad_1  
pip                       22.3.1           py38hecd8cb5_0  
pyparsing                 3.0.9            py38hecd8cb5_0  
python                    3.8.16               h218abb5_2  
python-dateutil           2.8.2              pyhd3eb1b0_0  
pytz                      2022.7           py38hecd8cb5_0  
readline                  8.2                  hca72f7f_0  
scipy                     1.9.3            py38h3d31255_0  
setuptools                65.6.3           py38hecd8cb5_0  
six                       1.16.0             pyhd3eb1b0_1  
sqlite                    3.40.1               h880c91c_0  
sympy                     1.11.1           py38hecd8cb5_0  
tbb                       2021.6.0             ha357a0b_1  
tk                        8.6.12               h5d9f67b_0  
tornado                   6.2              py38hca72f7f_0  
wheel                     0.37.1             pyhd3eb1b0_0  
xz                        5.2.10               h6c40b1e_0  
zipp                      3.11.0           py38hecd8cb5_0  
zlib                      1.2.13               h4dc903c_0  
zstd                      1.5.2                hcb37349_0  
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

- python/figure_a.py: 0.939 (s)
- python/figure_b.py: 0.939 (s)
- python/figure_c.py: 0.939 (s)
- python/figure_d.py: 0.939 (s)
- python/figure2_a.py: 2923.927 (s)
- python/figure2_b.py: 2923.927 (s)
- python/figure2_c.py: 2923.927 (s)
- python/figure2_d.py: 2923.927 (s)
- python/figure3.py: 1502.845 (s)

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