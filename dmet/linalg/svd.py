import scipy.linalg as la
import numpy as np
from pyscf import lib

class SVD:
    def __init__(self, A):
        '''
        A:   input rectangular matrix 
        u:   left  SVD column vectors (full)
        v:   right SVD column vectors 
        val: singular values
        '''
        self.A = A
        self.u = None
        self.v = None
        self.val = None

    def kernel(self):
        if self.u is None or self.v is None or self.val is None:
            self.u, self.val, vt = la.svd(self.A)
            self.v = vt.T
        return self.u, self.val, self.v

    def gradient(self, bR, contract="post", mask=None):
        '''
        Contract d u_ij / d A_kl with bR
        Assuming bR has the shape of (X, *A.shape) or A.shape
        pre-contract:
            return bR_xij d u_ij / d A_kl
        post-contract:
            return d u_ij / d A_kl bR_xkl
        mask:
            allows computing gradient of partial SVD vectors
        '''
        dim1, dim2 = self.A.shape
        if dim1 < dim1:
            raise NotImplemented('SVD matrix has to have dim1 >= dim2')
        if bR.ndim == 2:
            bR = bR[None]
        # sanity check
        assert bR.ndim == 3
        assert contract == "post" or contract == "pre"
        if mask is None:
            assert self.A.shape == bR.shape[1:]
        else:
            assert contract == "pre"
            assert sum(mask) == bR.shape[-1]
            assert self.A.shape[0] == bR.shape[1]
            assert len(mask) == self.A.shape[1]

        einsum = lib.einsum

        if mask is None:
            mask = [True] * dim2

        uf, val, v = self.kernel()
        u = uf[:,:dim2]
        uw = uf[:,dim2:]
        W = val[:,None]**2 - val[None,:]**2
        W[range(dim2), range(dim2)] = np.inf
        W = 1 / W

        if contract == "post":
            # sj ukm vlj bkl
            vbR = einsum('xkl,lj->xkj', bR, v[:,mask])
            temp = einsum('km,xkj->xmj', u, vbR)
            temp1 = einsum('j,xmj->xmj', val[mask], temp)
            # sm ukj vlm bkl
            temp = einsum('xkl,lm->xkm', bR, v)
            temp = einsum('kj,xkm->xmj', u[:,mask], temp)
            temp2 = einsum('m,xmj->xmj', val, temp)
            # uim (sj ukm vlj + sm ukj vlm) bkl / (sj^2 - sm^2)
            res = einsum('jm,xmj->xmj', W[mask], temp1+temp2)
            res = einsum('im,xmj->xij', u, res)
            if dim1 > dim2:
                # uim ukm vlj bkl / sj
                temp = einsum('km,xkj->xmj', uw, vbR)
                temp = einsum('j,xmj->xmj', 1/val[mask], temp)
                res += einsum('im,xmj->xij', uw, temp)
            return res
        else:
            # bij uim sj ukm vlj / (sj^2 - sm^2)
            temp = einsum('xij,im->xmj', bR, u)
            bRuW = einsum('xmj,jm->xmj', temp, W[mask])
            temp = einsum('xmj,j->xmj', bRuW, val[mask])
            temp = einsum('xmj,km->xjk', temp, u)
            res  = einsum('xjk,lj->xkl', temp, v[:,mask])
            # bij uim sm ukj vlm / (sj^2 - sm^2)
            temp = einsum('xmj,m->xmj', bRuW, val)
            temp = einsum('xmj,kj->xmk', temp, u[:,mask])
            res += einsum('xmk,lm->xkl', temp, v)
            if dim1 > dim2:
                # bij uim ukm vlj / sj
                temp = einsum('xij,im->xmj', bR, uw)
                temp = einsum('xmj,km->xkj', temp, uw)
                temp = einsum('xkj,j->xkj', temp, 1/val[mask])
                res += einsum('xkj,lj->xkl', temp, v[:,mask])
            return res
