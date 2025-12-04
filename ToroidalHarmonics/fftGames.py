# Written by Vassili Savinov on 05/11/2018
# getting myself familiarised with FFT in python

import numpy as np
import pylab as pl
import numpy.fft as npfft

################### lets start in 1d
# basic params
choiceFq=1.5
periodCount=30

# prepare the time axis
timePeriod=1.0/np.abs(choiceFq)
timeLim=timePeriod*periodCount
timeRange=np.linspace(-timeLim/2.0, timeLim/2.0, periodCount*100)
stepCount=len(timeRange)
timeStep=timeLim/stepCount

valRange=np.exp(1j*2*np.pi*choiceFq*timeRange)

pl.figure(1)
pl.plot(timeRange, valRange)

# now the fft
fftRange=npfft.fft(valRange)

# get the frequency axis
fqStep=1.0/timeLim
maxFq=1.0/(2.0*timeStep) # max possible frequency is 1 period per two time-steps
fqRange=np.arange(0, stepCount)*(fqStep)# correct for frequencies that are too large
iNeedToChange= np.where(fqRange>(maxFq+fqStep/2.0))# find those components which are definitely too large (in abs)
# i.e. if the frequencies do actually reach the maximum allowable fq, let this one be positive
fqRange[iNeedToChange] = fqRange[iNeedToChange]-2.0*maxFq

# apply fftshift
fftRange=npfft.fftshift(fftRange)
fqRange=npfft.fftshift(fqRange)

# expected trace
tgtFFTRange=np.zeros(fftRange.shape, dtype=np.complex128)
tgtFFTRange[np.abs(fqRange-choiceFq).argmin()]=stepCount*np.exp(-1j*np.pi*timeLim*choiceFq)

pl.figure(2)
pl.subplot(211)
pl.plot(fqRange, np.real(fftRange), label='real')
pl.plot(fqRange, np.imag(fftRange), label='imag')
pl.plot((-choiceFq, -choiceFq), (-stepCount, stepCount))
pl.plot((choiceFq, choiceFq), (-stepCount, stepCount))
pl.legend()
#
pl.subplot(212)
pl.plot(fqRange, np.real(tgtFFTRange), label='real')
pl.plot(fqRange, np.imag(tgtFFTRange), label='imag')
pl.legend()

pl.show()