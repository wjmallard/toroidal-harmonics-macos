#!/bin/bash
set -e

rm -f test

gcc test.c wrapDTORH64.dylib -o test

./test
