##################################################
# Written by Vassili Savinov on 14/09/2018
# redo the testpro.f in Python and get the same results
##################################################

import ctypes as ct
import numpy as np

############ load dll
dllTorHarm = ct.CDLL('wrapDTORH64.dylib')

############ prep data
nDim = 301 # fortran will access inidces 0...nMax, so nMax+1 elements
mDim = 51
nMax = 300
mVal = 120

ql_o_gammaVec = np.zeros(nDim+1, dtype=np.double) # scaled by 1/gamma(m+1/2)
pl_o_gammaVec = np.zeros(nDim+1, dtype=np.double)

c_newN=(ct.c_int)(0)
c_mode=(ct.c_int)(0)
c_errCode=ct.c_int(0)

########### dtorh1
# Note that DTORH1 is hard-coded (dtorh1.f:52) for mode 1
# so its output is scaled by 1/gamma(m+1/2)

c_mode.value=1

print('Using DTORH1')
for iCalc in range(0,6):
    zVal = 1.5+iCalc*1.6 # values of z

    dllTorHarm.wrapDTORH1((ct.c_double)(zVal),
                          (ct.c_int)(mVal),
                          (ct.c_int)(nMax),
                          pl_o_gammaVec.ctypes.data_as(ct.POINTER(ct.c_double)),
                          ql_o_gammaVec.ctypes.data_as(ct.POINTER(ct.c_double)),
                          ct.pointer(c_newN),
                          ct.pointer(c_errCode),
                          c_mode)
    if c_errCode.value != 0: print('dtorh1 error: code %d'% c_errCode.value)



    print('\tz=%.3f,\t nNew=%d,\t PL/Gamma[nNew]=%-20.8e QL/Gamma[nNew]=%-20.8e PL/Gamma[0]=%-20.8e' % \
          (zVal, c_newN.value, pl_o_gammaVec[c_newN.value], ql_o_gammaVec[c_newN.value], pl_o_gammaVec[0]))

    altN=10
    print('\tz=%.3f,\t nNew=%d,\t PL/Gamma[  10]=%-20.8e QL/Gamma[  10]=%-20.8e PL/Gamma[0]=%-20.8e' % \
          (zVal, altN, pl_o_gammaVec[altN], ql_o_gammaVec[altN], pl_o_gammaVec[0]))

#error handling
dllTorHarm.wrapDTORH1((ct.c_double)(0.5),
                           (ct.c_int)(mVal),
                           (ct.c_int)(nMax),
                          pl_o_gammaVec.ctypes.data_as(ct.POINTER(ct.c_double)),
                           ql_o_gammaVec.ctypes.data_as(ct.POINTER(ct.c_double)),
                           ct.pointer(c_newN),
                      ct.pointer(c_errCode),
                      c_mode)
if c_errCode.value != 0: print('dtorh1 error: code %d'% c_errCode.value)

########### dtorh2
# Note that DTORH2 is hard-coded (dtorh1.f:52) for mode 0

c_mode.value=0
print('\n\nUsing DTORH2')

#define FORT2D_TOC1D(M,N,MDIM) (N*(MDIM+1)+M)
#Fort2D_to_1D=lambda m,n,mdim: n*(mDim+1)+m

mDim = 51
nDim = 301
mMax = mDim-1
nMax = nDim-1

c_newM = ct.c_int(0)
newNM = np.zeros(mDim+1, dtype=np.int32)

ql1DVec = np.zeros((nDim+1)*(mDim+1), dtype=np.double)
pl1DVec = np.zeros((nDim+1)*(mDim+1), dtype=np.double)

for iCalc in range(0,6):
    zVal = 1.5+iCalc*1.6 # values of z

    dllTorHarm.wrapDTORH2(    (ct.c_double)(zVal),
                              (ct.c_int)(mDim),
                              (ct.c_int)(nDim),
                              (ct.c_int)(mMax),
                              (ct.c_int)(nMax),
                              pl1DVec.ctypes.data_as(ct.POINTER(ct.c_double)),
                              ql1DVec.ctypes.data_as(ct.POINTER(ct.c_double)),
                              ct.pointer(c_newM),
                              newNM.ctypes.data_as(ct.POINTER(ct.c_int)),
                              ct.pointer(c_errCode), c_mode)
    if c_errCode.value != 0: print('dtorh2 error: code %d' % c_errCode.value)

    NM=newNM[c_newM.value]

    plMat=pl1DVec.reshape((nDim+1, mDim+1))# now it is [N,M] array
    qlMat=ql1DVec.reshape((nDim+1, mDim+1))# now it is [N,M] array

    print('\tz=%.3f,\t newM=%d,\t NM=%d,\t PL[N,M]=%-20.8e QL[N,M]=%-20.8e PL[0,0]=%-20.8e' % \
              (zVal, c_newM.value, NM, \
               plMat[NM, c_newM.value], \
               qlMat[NM, c_newM.value], \
               plMat[0,0]))

    M=10
    NM=newNM[M]

    print('\tz=%.3f,\t newM=%d,\t NM=%d,\t PL[N,M]=%-20.8e QL[N,M]=%-20.8e PL[0,0]=%-20.8e' % \
          (zVal, M, NM, \
           plMat[NM, M], \
           qlMat[NM, M], \
           plMat[0, 0]))

########### dtorh3

print('\n\nUsing DTORH3')

mDim = 51
nDim = 301
mMax = mDim - 1
nMax = nDim - 1

c_newM = ct.c_int(0)
c_newN = ct.c_int(0)

ql1DVec = np.zeros((nDim + 1) * (mDim + 1), dtype=np.double)
pl1DVec = np.zeros((nDim + 1) * (mDim + 1), dtype=np.double)

for iCalc in range(0, 6):
    zVal = 1.5 + iCalc * 1.6  # values of z

    dllTorHarm.wrapDTORH3((ct.c_double)(zVal),
                          (ct.c_int)(mDim),
                          (ct.c_int)(nDim),
                          (ct.c_int)(mMax),
                          (ct.c_int)(nMax),
                          pl1DVec.ctypes.data_as(ct.POINTER(ct.c_double)),
                          ql1DVec.ctypes.data_as(ct.POINTER(ct.c_double)),
                          ct.pointer(c_newM),
                          ct.pointer(c_newN),
                          ct.pointer(c_errCode), c_mode)
    if c_errCode.value != 0: print('dtorh3 error: code %d' % c_errCode.value)

    plMat = pl1DVec.reshape((nDim + 1, mDim + 1))  # now it is [N,M] array
    qlMat = ql1DVec.reshape((nDim + 1, mDim + 1))  # now it is [N,M] array

    print('\tz=%.3f,\t newM=%d,\t NM=%d,\t PL[N,M]=%-20.8e QL[N,M]=%-20.8e PL[0,0]=%-20.8e' % \
          (zVal, c_newM.value, c_newN.value, \
           plMat[c_newN.value, c_newM.value], \
           qlMat[c_newN.value, c_newM.value], \
           plMat[0, 0]))

    M = 10
    NM = newNM[M]

    print('\tz=%.3f,\t newM=%d,\t NM=%d,\t PL[N,M]=%-20.8e QL[N,M]=%-20.8e PL[0,0]=%-20.8e' % \
          (zVal, M, c_newN.value, \
           plMat[c_newN.value, M], \
           qlMat[c_newN.value, M], \
           plMat[0, 0]))


# error handling
c_mode.value=5
dllTorHarm.wrapDTORH3((ct.c_double)(-zVal),
                          (ct.c_int)(mDim),
                          (ct.c_int)(nDim),
                          (ct.c_int)(mMax),
                          (ct.c_int)(nMax),
                          pl1DVec.ctypes.data_as(ct.POINTER(ct.c_double)),
                          ql1DVec.ctypes.data_as(ct.POINTER(ct.c_double)),
                          ct.pointer(c_newM),
                          ct.pointer(c_newN),
                          ct.pointer(c_errCode), c_mode)
if c_errCode.value != 0: print('dtorh3 error: code %d' % c_errCode.value)