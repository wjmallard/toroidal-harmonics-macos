#!/bin/bash
set -e

# clean up
rm -f wrapDTORH64.dylib

# compile Fortran modules
gfortran -c -fPIC -O2 dtorh1_mod0.f
gfortran -c -fPIC -O2 dtorh1_mod1.f
gfortran -c -fPIC -O2 dtorh1_mod2.f

gfortran -c -fPIC -O2 dtorh2_mod0.f
gfortran -c -fPIC -O2 dtorh2_mod1.f
gfortran -c -fPIC -O2 dtorh2_mod2.f

gfortran -c -fPIC -O2 dtorh3_mod0.f
gfortran -c -fPIC -O2 dtorh3_mod1.f
gfortran -c -fPIC -O2 dtorh3_mod2.f

gfortran -c -fPIC -O2 rout_mod.f

# compile C wrappers
gcc -c -fPIC -O2 wrapDTORH1.c
gcc -c -fPIC -O2 wrapDTORH2.c
gcc -c -fPIC -O2 wrapDTORH3.c

# link shared library
gfortran -shared -o wrapDTORH64.dylib \
    dtorh1_mod0.o dtorh1_mod1.o dtorh1_mod2.o \
    dtorh2_mod0.o dtorh2_mod1.o dtorh2_mod2.o \
    dtorh3_mod0.o dtorh3_mod1.o dtorh3_mod2.o \
    rout_mod.o \
    wrapDTORH1.o wrapDTORH2.o wrapDTORH3.o \
    -framework Accelerate
