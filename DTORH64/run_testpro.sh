#!/bin/bash
set -e

rm -f testpro

gfortran -O2 dtorh1.f dtorh2.f dtorh3.f rout.f testpro.f -o testpro -framework Accelerate

./testpro
