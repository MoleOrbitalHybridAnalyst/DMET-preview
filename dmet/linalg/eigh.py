import scipy.linalg as la
import numpy as np
from pyscf import lib

class EIGH:
    def __init__(self, A):
        '''
        A:   input square matrix 
        u:   eigen column vectors 
        val: eigen values
        '''
        self.A = A
        self.u = None
        self.val = None

    def kernel(self):
        if self.u is None or self.val is None:
            self.val, self.u = la.eigh(self.A)
        return self.val, self.u

    def gradient(self, bR, contract='post'):
        '''
        Contract d u_ij / d A_kl with bR
        Assuming bR has the shape of (X, *A.shape) or A.shape
        pre-contract:
            return bR_xij d u_ij / d A_kl
        post-contract:
            return d u_ij / d A_kl bR_xkl
        '''
        if bR.ndim == 2:
            bR = bR[None]
        # sanity check
        assert bR.ndim == 3
        assert contract == "post" or contract == "pre"
        assert self.A.shape == bR.shape[1:]
        dim = self.A.shape[0]

        einsum = lib.einsum

        val, u = self.kernel()
        W = val[:,None] - val[None,:]
        W[range(dim), range(dim)] = np.inf
        W = 1 / W

        if contract == "post":
            temp = einsum('xkl,lj->xkj', bR, u)
            temp = einsum('km,xkj->xjm', u, temp)
            temp = einsum('jm,xjm->xjm', W, temp)
            return einsum('im,xjm->xij', u, temp)
        else:
            temp = einsum('xij,im->xjm', bR, u)
            temp = einsum('xjm,jm->xjm', temp, W)
            temp = einsum('xjm,km->xjk', temp, u)
            return einsum('xjk,lj->xkl', temp, u)

    def gradient_predecomp(self, aR):
        '''
        d u_ij / d A_kl takes the form of u_im D_jmkl
        bR_ij d u_ij / d A_kl = bR_ij u_im D_jmkl
        define bR_ij u_im = aR_jm
        this function returns aR_jm D_jmkl
        '''
        # sanity check
        assert aR.ndim == 2
        assert self.A.shape == aR.shape
        dim = self.A.shape[0]

        einsum = lib.einsum

        val, u = self.kernel()
        W = val[:,None] - val[None,:]
        W[range(dim), range(dim)] = np.inf

        temp = aR / W
        temp = einsum('jm,km->jk', temp, u)
        return einsum('jk,lj->kl', temp, u)
