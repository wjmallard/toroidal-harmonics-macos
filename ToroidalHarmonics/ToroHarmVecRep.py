# Written by Vassili Savinov on 05/02/19
# class for containing the representation of a 3d vector
# field in terms of toroidal harmonics
# see Toroidal_Harmonics_Def.pdf

import numpy as np
import numpy.linalg as npla
from DTORH import DTORH
from ToroCoords import ToroCoords

class ToroHarmVecRep:

    # find the decomposution of the vector field
    def __find_decomposition(self, Q_Tens, h_Tens):
        # contract the vector field terms to extract the coupling matrix
        # in my notes LNM is lnm with bar
        # in the notes this is the M - matrix
        div_coup_Tens = np.einsum('LNMs,lnms->LNMlnm', np.conj( self.jacTens * self.psiTens ), Q_Tens)
        div_sol_Tens = np.einsum('LNMs,s->LNM', np.conj( self.jacTens * self.psiTens ), h_Tens)

        # in the notes this is q
        return npla.tensorsolve(div_coup_Tens, div_sol_Tens)


    # build a rank 4 tensor lnms
    # l - type (P,Q)
    # n - legendre n
    # m - legendre m
    # s - all the spatial values flattened in an array
    def __prep_contraction_tens(self):
        # way that divergence, angular momentum and radius. modify the toroidal harmonics
        # div(r Psi)
        self.divContr_Tens = (3 * self.psiTens
                              - np.sinh(self.etaTens) * np.cos(self.thetaTens) * self.D_psi_D_eta_Tens
                              - np.cosh(self.etaTens) * np.sin(self.thetaTens) * self.D_psi_D_theta_Tens)

        # the second one needs to be done in terms
        # r.curl(L Psi)
        term_eta =  (-4.0*np.cosh(self.etaTens))/(np.sinh(self.etaTens)**3) + \
                    (np.sinh(4*self.etaTens)*np.cos(2*self.thetaTens))/(np.sinh(self.etaTens)**4)
        #
        term_theta = (8*np.sin(2*self.thetaTens))/(np.tanh(self.etaTens)**2)
        #
        term_eta_theta = (8*np.sin(2*self.thetaTens))/(np.tanh(self.etaTens))
        #
        term_eta2 = (-8 * np.sin(self.thetaTens)**2) / (np.tanh(self.etaTens)**2)
        #
        term_theta2 = (-8 * np.cos(self.thetaTens) ** 2)
        #
        term_phi2 = 4*(np.cos(2*self.thetaTens)-np.cosh(2*self.etaTens))/(np.sinh(self.etaTens)**4)

        self.rDotContr_Tens = (1j*np.sinh(self.etaTens)**2/8.0)*(
            term_eta * self.D_psi_D_eta_Tens + term_theta * self.D_psi_D_theta_Tens + \
            term_eta_theta * self.D2_psi_D_eta_theta_Tens + term_eta2 * self.D2_psi_D_eta2_Tens + \
            term_theta2 * self.D2_psi_D_theta2_Tens + term_phi2 * (1j*self.mTens)**2 * self.psiTens
        )

        # third term is not needed since the b-coeff can be extracted
        # using the rDotContr_Tens, i.e. r.curl(L Psi) tensor

    # get the legendre polynomials and build all the necessary psi and related tensors that will be needed for further
    # computation
    # the tensors here will have 4 dimensions: first dimension is either for first or second kind of Leg function
    # second dimension for n, third for m, fourth dimension is for different positions in space (flattened into a single column)
    def __prep_psi_tens(self):
        # get the necessary legendre functions
        with DTORH() as dtorh: (pCube, qCube) = \
            dtorh.GetCubeNMZ(self.nCount + 2, self.mCount, np.cosh(self.etaVec))

        # need to define the basic tensors
        self.etaTens = np.tile(self.etaVec, (2, self.nCount, self.mCount, 1))
        self.thetaTens = np.tile(self.thetaVec, (2, self.nCount, self.mCount, 1))
        self.phiTens = np.tile(self.phiVec, (2, self.nCount, self.mCount, 1))
        self.jacTens = np.tile(self.jacobian, (2, self.nCount, self.mCount, 1))#Jacobian tensor
        self.r2Tens = np.tile(self.r2Vec, (2, self.nCount, self.mCount, 1))

        # occurs all the time
        CoshCos = np.cosh(self.etaTens) - np.cos(self.thetaTens)

        # nm tensors
        nVec = np.arange(self.nCount)
        self.nTens = np.tile(nVec[None, :, None, None], (2, 1, self.mCount, len(self.etaVec)))
        #
        mVec = np.arange(self.mCount)
        self.mTens = np.tile(mVec[None, None, :, None], (2, self.nCount, 1, len(self.etaVec)))

        # extend cubes to full tensors
        # basic
        pqTens = np.zeros( (2, self.nCount, self.mCount, len(self.etaVec)), dtype=np.complex128)
        pqTens[0, :, :, :] = pCube[0:self.nCount, :, :]#  loose tail
        pqTens[1, :, :, :] = qCube[0:self.nCount, :, :]  # loose tail

        # based on this define the psi tensor
        self.psiTens = np.sqrt( CoshCos ) * pqTens * \
                       np.exp(1j*(self.nTens*self.thetaTens)) * np.exp(1j*(self.mTens*self.phiTens))

        # shifted n -> n+1
        pq_n_plus_1_Tens = np.zeros( (2, self.nCount, self.mCount, len(self.etaVec)), dtype=np.complex128)
        pq_n_plus_1_Tens[0, :, :, :] = pCube[1:(self.nCount+1), :, :]#  loose tail
        pq_n_plus_1_Tens[1, :, :, :] = qCube[1:(self.nCount+1), :, :]  # loose tail

        # based on this define shifted psi tensor
        self.psi_n_plus_1_Tens = np.sqrt( CoshCos ) * pq_n_plus_1_Tens * \
                       np.exp(1j * ( (self.nTens+1) * self.thetaTens)) * np.exp(1j * (self.mTens * self.phiTens))

        # shifted n -> n+2
        pq_n_plus_2_Tens = np.zeros((2, self.nCount, self.mCount, len(self.etaVec)), dtype=np.complex128)
        pq_n_plus_2_Tens[0, :, :, :] = pCube[2:(self.nCount + 2), :, :]  # loose tail
        pq_n_plus_2_Tens[1, :, :, :] = qCube[2:(self.nCount + 2), :, :]  # loose tail

        # based on this define twice shifted psi tensor
        self.psi_n_plus_2_Tens = np.sqrt( CoshCos ) * pq_n_plus_2_Tens * \
                                 np.exp(1j * ((self.nTens + 2) * self.thetaTens)) * np.exp(1j * (self.mTens * self.phiTens ))

        # now we can work, the basic psi is defined, but let as also define the derivatives
        # first order
        self.D_psi_D_eta_Tens = \
            ( np.cosh(self.etaTens)*((2*self.nTens+1)*np.cos(self.thetaTens)-2*self.nTens*np.cosh(self.etaTens)) - 1 )*\
            self.psiTens/\
            (2 * np.sinh(self.etaTens) * CoshCos ) + \
            (self.nTens+0.5-self.mTens)*np.exp(-1j*self.thetaTens)*self.psi_n_plus_1_Tens/np.sinh(self.etaTens)

        self.D_psi_D_theta_Tens = \
            (1j*self.nTens + (np.sin(self.thetaTens))/(2 * CoshCos )) * self.psiTens

        self.D_psi_D_phi_Tens = 1j * self.mTens * self.psiTens

        # second order tensor with respect to eta. needs to be done in bits to keep clarity
        preFac_0 = 1.0/(16 * np.sinh(self.etaTens) * CoshCos**2)
        #
        term1_0 = (-4*np.cos(self.thetaTens)/np.tanh(self.etaTens))*\
                ((4*self.nTens**2+2*self.nTens+1)*np.cosh(2*self.etaTens)+5+2*self.nTens*(7+2*self.nTens))
        #
        term2_0 = (1/np.sinh(self.etaTens))*\
                  (6*(3+4*self.nTens*(2+self.nTens)) +
                   (1+2*self.nTens)*(5+2*self.nTens+(1+2*self.nTens)*np.cosh(2*self.etaTens))*np.cos(2*self.thetaTens))
        #
        term34_0 = 18*np.sinh(self.etaTens) + \
                   8*self.nTens*(5+4*self.nTens+self.nTens*np.cosh(2*self.etaTens))*np.sinh(self.etaTens)
        term_0 = preFac_0 * (term1_0 + term2_0 + term34_0)
        ##
        preFac_1 = (2*self.mTens - 2*self.nTens - 1)*np.exp(-1j*self.thetaTens)/(2*(np.sinh(self.etaTens)**2)*CoshCos)
        #
        term123_1 = 2 + self.nTens - (3+2*self.nTens)*np.cos(self.thetaTens)*np.cosh(self.etaTens)+\
                    (1+self.nTens)*np.cosh(2*self.etaTens)
        #
        term_1 = preFac_1 * term123_1
        ##
        term_2 = \
            np.exp(-1j*2*self.thetaTens)*(2*self.mTens-3-2*self.nTens)*(2*self.mTens-2*self.nTens-1)\
            /(4*np.sinh(self.etaTens)**2)

        #now put it together
        self.D2_psi_D_eta2_Tens = term_0*self.psiTens + term_1*self.psi_n_plus_1_Tens + term_2*self.psi_n_plus_2_Tens

        #second order with respect  to theta
        preFac = 1.0/(4*CoshCos**2)
        term0 = 2*CoshCos*(np.cos(self.thetaTens)-2*self.nTens**2*CoshCos)
        term1 = 1j*4*self.nTens*CoshCos*np.sin(self.thetaTens)
        term2 = -np.sin(self.thetaTens)**2
        #
        self.D2_psi_D_theta2_Tens = preFac*(term0+term1+term2)*self.psiTens

        # cross-term
        self.D2_psi_D_eta_theta_Tens = (-np.sin(self.thetaTens)*np.sinh(self.etaTens)/(2*CoshCos**2))*self.psiTens + \
                                       (np.sin(self.thetaTens)/(2*CoshCos)+1j*self.nTens)*self.D_psi_D_eta_Tens


    # remove spatial points that are bad in toroidal coordinates and flatten arrays
    def __extract_clean_points(self, xTens, yTens, zTens, divFTens, rDotFTens, LDotFTens, tiny):
        # first of all get the toroidal coordinates
        torCrd = ToroCoords()
        etaTens, thetaTens, phiTens = torCrd.Cart_to_Toro(xTens, yTens, zTens)

        # prepare the legendre polynomials
        temp_coshEtaVec = (np.cosh(etaTens)).flatten()

        # some values are bad for numerical calculations, e.g. coshEtaVec=1.0 filter them
        # add all the filters here
        # there is no problem with dropping some values, that's why we have many!
        goodLogical = np.isfinite(temp_coshEtaVec)
        # further checks are impossible non-finite numbers, so remove them before any further logical checks
        temp_coshEtaVec[np.where(goodLogical != True)] = 0.0
        # good to contninue
        goodLogical = np.logical_and((temp_coshEtaVec - 1.0) >= tiny, goodLogical)
        self.goodInds = np.where(goodLogical)  # find indices , keep this in case any further cleanup is needed

        # store good flattened values
        self.etaVec = etaTens.flatten()[self.goodInds]
        self.thetaVec = thetaTens.flatten()[self.goodInds]
        self.phiVec = phiTens.flatten()[self.goodInds]
        self.divFVec = divFTens.flatten()[self.goodInds]
        self.rDotFVec = rDotFTens.flatten()[self.goodInds]
        self.LDotFVec = LDotFTens.flatten()[self.goodInds]

        # also store the sqrt_metric for further computations
        self.jacobian = torCrd.sqrt_metric(self.etaVec, self.thetaVec, self.phiVec)

        # need r2
        self.r2Vec = xTens.flatten()[self.goodInds]**2 + yTens.flatten()[self.goodInds]**2 + zTens.flatten()[self.goodInds]**2

    # to initialize itwe need is
    # three rank-3 tensors giving the XYZ coordinates of the points in 3d space
    # then we need 3 rank three tensors giving the div(F), r.F and L.F, where F is the field of interest
    # nCount and mCount is the maxiumum orders of toroidal harmonics to consider
    # tiny = a finite small number taken to be equivalent to zero
    def __init__(self, raw_xTens, raw_yTens, raw_zTens, raw_divFTens, raw_rDotFTens, raw_LDotFTens, nCount=19, mCount=19, tiny=1e-12):

        self.nCount = nCount
        self.mCount = mCount

        # convert tensors for spatial points into flattened vectors
        # and remove all the points that have bad cooridinates in toroidal coordinate system
        self.__extract_clean_points(raw_xTens, raw_yTens, raw_zTens, raw_divFTens, raw_rDotFTens, raw_LDotFTens, tiny)

        # prepare psi tensors
        self.__prep_psi_tens()

        # get the contraction tensors
        self.__prep_contraction_tens()

        # find the actual decomposition, project the scalar field onto toroidal harmonic tensor
        # and then solve the tensor equation to get the coupling coefficient
        self.aCoeff_Tens = self.__find_decomposition(self.divContr_Tens, self.divFVec)

        # the c-term is found using the a-term
        rDot_sol_Vec = (self.rDotFVec-np.einsum('lnm,lnms->s', self.aCoeff_Tens, self.r2Tens*self.psiTens))
        self.cCoeff_Tens = self.__find_decomposition(self.rDotContr_Tens, rDot_sol_Vec)

        # final trerm
        # note that it can actually be obtained using
        # the same contraction tensor as for the c-term
        self.bCoeff_Tens = self.__find_decomposition(self.rDotContr_Tens, 1j*self.LDotFVec)