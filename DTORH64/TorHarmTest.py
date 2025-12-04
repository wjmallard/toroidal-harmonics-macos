import ctypes as ct
import numpy as np

# load dll
dllTorHarm=ct.CDLL('wrapDTORH64.dylib')

# prep data
nMax=300
mVal=2
c_newN=(ct.c_int)(0)# need a propper array here to pass as pointer
zVal=np.array(1.00001, dtype=np.double)
# arrays
# some weird 'fails to converge' comes on if the
# PLVec, QLVec is set to be what it is supposed to be
# increasing it by even 2 seems to fix the error. But i want to be on the safe side
# tested it all by running the author's program in gdb-debugger and
# mine, and ensuring the equality between going-out given the same going-in params
qlVec=np.zeros(nMax*2+10, dtype=np.double)
plVec=np.zeros(nMax*2+10, dtype=np.double)

c_err = (ct.c_int)(0)
mode = 0

dllTorHarm.wrapDTORH1(  (ct.c_double)(zVal),
                            (ct.c_int)(mVal),
                            (ct.c_int)(nMax),
                            plVec.ctypes.data_as(ct.POINTER(ct.c_double)),
                            qlVec.ctypes.data_as(ct.POINTER(ct.c_double)),
                            ct.pointer(c_newN),
                            ct.pointer(c_err),
                            (ct.c_int)(mode))

newN=c_newN.value

print('z=%.5f' % zVal)
print('PL[0]=%.5e,\tQL[0]=%.5e' % (plVec[0], qlVec[0]))
print('PL[1]=%.5e,\tQL[1]=%.5e' % (plVec[1], qlVec[1]))
print('PL[last]=%.5e,\tQL[last]=%.5e' % (plVec[newN+1], qlVec[newN+1]))

print('done')
