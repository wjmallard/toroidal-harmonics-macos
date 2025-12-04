# Written by Vassili Savinov on 05/11/2018
# getting myself familiarised with FFT in python

import numpy as np
import pylab as pl
import numpy.fft as npfft

################### ld
# basic params
choiceFqs=np.array((1.5, 5))
periodCount=20

# prepare the time axis
timePeriods=1.0/np.abs(choiceFqs)
timeLims=timePeriods*periodCount
timeRange1=np.linspace(-timeLims[0]/2.0, timeLims[0]/2.0, periodCount*10)
timeRange2=np.linspace(-timeLims[1]/2.0, timeLims[1]/2.0, periodCount*10)
stepCounts=np.zeros(2, dtype=np.int64)
stepCounts[0]=len(timeRange1)
stepCounts[1]=len(timeRange2)

T1, T2=np.meshgrid(timeRange1, timeRange2)

timeSteps=timeLims/stepCounts

valMat=np.exp(1j*2*np.pi*(choiceFqs[0]*T1+choiceFqs[1]*T2))+np.exp(1j*2*np.pi*(choiceFqs[0]*T1-choiceFqs[1]*T2))

pl.figure(1)
pl.contour(T1, T2, np.real(valMat))

fftMat=npfft.fft2(valMat)

# build axis
fqSteps=1.0/timeLims
maxFqs=1.0/(2.0*timeSteps) # max possible frequency is 1 period per two time-steps
fqRange1=np.arange(0, stepCounts[0])*(fqSteps[0])# correct for frequencies that are too large
fqRange2=np.arange(0, stepCounts[1])*(fqSteps[1])
iNeedToChange1= np.where(fqRange1>(maxFqs[0]+fqSteps[0]/2.0))# find those components which are definitely too large (in abs)
iNeedToChange2= np.where(fqRange2>(maxFqs[1]+fqSteps[1]/2.0))
# i.e. if the frequencies do actually reach the maximum allowable fq, let this one be positive
fqRange1[iNeedToChange1] = fqRange1[iNeedToChange1]-2.0*maxFqs[0]
fqRange2[iNeedToChange2] = fqRange2[iNeedToChange2]-2.0*maxFqs[1]

FQ1, FQ2=np.meshgrid(fqRange1, fqRange2)

fftMat=npfft.fftshift(fftMat)
FQ1=npfft.fftshift(FQ1)
FQ2=npfft.fftshift(FQ2)

pl.figure(2)
pl.contour(FQ1, FQ2, np.abs(fftMat))
pl.axis('equal')

pl.show()