# AI Nuclear Fusion 2022
## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Usage](#usage)
3. [Todo](#Todo)

## Introduction
This project aims to develop a Particle in Cell plasma code.
Original code in 2021 by Hilary,Yin Yue and Chloe, extensive improvements by Samuel,Ananth and Vishwa.

## Getting Started
### Prerequisites
- MSYS2
- to avoid confusion, use either of "mingw64" or "ucrt" and do not mix the two. The following examples make use pf ucrt  
- Tools 
    pacman -S base-devel cmake git mingw-w64-ucrt-x86_64-gcc paraview  
- Libs (Opencl, OpenMP, gsl, fftw, vtk, nfftw)
    pacman -S mingw-w64-ucrt-x86_64-opencl-headers mingw-w64-ucrt-x86_64-opencl-clhpp mingw-w64-ucrt-x86_64-opencl-icd mingw-w64-ucrt-x86_64-openmp mingw-w64-ucrt-x86_64-gsl mingw-w64-ucrt-x86_64-fftw mingw-w64-ucrt-x86_64-vtk 
using float version of nfftw3f. should install from source using ./configure --enable-float
https://www-user.tu-chemnitz.de/~potts/nfft/installation.php

- GCC added to PATH in MSYS
    - In the root directory, run `export PATH=$PATH:/ucrt64/bin`
    - this can also be added in windows "edit system environment variables" , "Path". Add C:\msys64\usr\bin and C:\msys64\ucrt64\bin



## Usage

## Todo
- Setup particles for hot rod project
- Setup particles for hot plate project 
- Setup particles for MagLIF
    - Two sets of plasmas similar to hot rod project
        - 1. Low density plasma cylinder
        - 2. High density thin cylindrical shell around plasma cylinder.
    - Setup external fields. 
        - Electric field along the cylinder.
- add in "artificial viscosity" to simulate energy loss/gain
    - F=q(E+vxB)+rv w
    - here viscosity r is negative when there is energy loss and positive when there is energy gain
- add temperature field Te[x][y][z]
    - approximate Te as average KE of particles
- viscosity_field[p][x][y][z] 
    - P=F*v=r * v * v, So r=P_perparticle/(v*v)
    - Bremstrahlung radiation loss from NRL Plasma formulary, https://tanimislam.github.io/research/NRL_Formulary_2019.pdf
     - Pbr=1.69e-32 * Ne * pow(Te,0.5) Sum(Z * Z * Ni(Z)) in Watts per cm^3
    - cyclotron radiation loss from NRL Plasma formulary
     - Pc=6.21-28 * B * B * Ne * Te in Watts per cm^3
    - power density from fusion from NRL Plasma formulary (Assume to be absorbed within the cell for the time being not realistic as fusion products may have very long range. if cell sizes are small, most of energy from fusion products will be lost from the cell) all in in Watts per cm^3
     - P_DD=3.3e-13 * N_D * N_D * sigma_v_dd
     - P_DT=5.6e-13 * N_D * N_T * sigma_v_DT
     - P_DHe3=2.9e-12 * N_D * N_He3 * sigma_v_DHe3
    - power transfer between electrons and ion



- to get more performance, you might want to recompile the libraries used. for example to install FFTW3 recompiled with OMP enabled (if not on Windows):
 > wget https://www.fftw.org/fftw-3.3.10.tar.gz
 > tar xvzf fftw-3.3.10.tar.gz
 > cd fftw-3.3.10/
 > ./configure --enable-threads --enable-openmp --enable-avx --enable-avx2 --enable-avx512 --enable-avx-128-fma --enable-float --with-our-malloc --enable-sse2
 > make
 > make install

    For Windows, we have to compile FFTW into one single shared library:
 > ./configure --enable-threads --enable-sse2 --enable-avx --enable-avx2 --enable-avx-128-fma --with-our-malloc --enable-float --enable-shared --disable-static --with-combined-threads

- Next, we can also recompile NFFT to make use of multithreading:
> cd ../nfft-3.5.3/
> ./configure --enable-float --enable-openmp
> make
> make install
- Resulting library can be linked using -lnfft3f_threads
