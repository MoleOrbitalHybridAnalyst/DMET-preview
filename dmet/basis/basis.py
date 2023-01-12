from dmet import linalg

from pyscf.lib import einsum, dot
from pyscf import ao2mo

import numpy as np
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
class Basis:
    '''
    Class for basis transformation
    '''
    def __init__(self):
        self.Cao2mo = None
        self.Cmo2ao = None
        self.Cao2lo = None
        self.Clo2ao = None
        self.Clo2eo = None
        self.Cao2eo = None
        self.Ceo2ao = None
        self.bath_mask = None
        self.ovlp = None
        self.svd = None

    def make_ao_lo(self, ovlp, Cao2lo=None):
        '''
        make basis transformation between AO and LO
        '''
        if Cao2lo is None:
            raise Exception("virtual function called")
        else:
            self.Cao2lo = Cao2lo
            self.Clo2ao = Cao2lo.T @ ovlp
            self.ovlp = ovlp

    def make_ao_mo(self, Cao2mo, ovlp=None):
        '''
        make basis transformation between AO and MO
        '''
        if ovlp is None:
            ovlp = self.ovlp
        else:
            self.ovlp = ovlp
        self.Cao2mo = Cao2mo
        self.Cmo2ao = Cao2mo.T @ ovlp

    def make_lo_eo(self, glob_r1, frag, bath_tol=-1):
        '''
        make basis from LO to EO given 
        then env-imp block of global rdm1 in LO basis
        '''
        self.svd = linalg.SVD(glob_r1[frag.env_imp])
        u, val, _ = self.svd.kernel()

        # find the least bath orbs such that 
        # sum(bath) >= (1-bath_tol) * total
        numrank = sum(np.abs(val))
        cum = np.cumsum(np.abs(val)) / numrank
        if len(cum) == 1:
            i = -1
        for i in range(len(cum)-1):
            if cum[i] < 1 - bath_tol and cum[i+1] >= 1 - bath_tol:
                break
        mask = np.zeros(len(cum), dtype=bool)
        mask[:i+2] = True
        if cum[0] >= 1 - bath_tol and len(cum) > 1:
            mask[1:] = False

        nlo, nimp = self.svd.A.shape
        nlo += nimp
        neo = nimp + sum(mask)
        self.Clo2eo = np.zeros((nlo, neo))
        self.Clo2eo[frag.imp, :nimp] = np.eye(nimp)
        self.Clo2eo[frag.env, nimp:] = u[:, :nimp][:, mask]
        self.bath_mask = mask

    def make_ao_eo(self, ovlp=None, glob_r1=None, frag=None, bath_tol=-1):
        if self.Cao2lo is None:
            self.make_ao_lo(ovlp)
        if self.Clo2eo is None:
            self.make_lo_eo(glob_r1, frag, bath_tol=bath_tol)
        self.Cao2eo = self.Cao2lo @ self.Clo2eo
        self.Ceo2ao = self.Cao2eo.T @ self.ovlp
        self.Ceo2lo = self.Clo2eo.T

    def transform_h(self, h, string):
        h_dict = \
            {'AE': self.Cao2eo, 'AL': self.Cao2lo, 'EA': self.Ceo2ao, \
            'LE': self.Clo2eo, 'AM': self.Cao2mo, 'MA': self.Cmo2ao, \
            'AA': Literal[1], 'LL': Literal[1], 'EE': Literal[1]}
        rep1, rep2 = string.upper().split(',')
        try:
            b1 = h_dict[rep1[0]+rep2[0]]
            b2 = h_dict[rep1[1]+rep2[1]]
        except KeyError:
            raise NotImplementedError
        if b1 is not Literal[1] and b2 is not Literal[1]:
            return b1.T @ h @ b2
        elif b1 is not Literal[1]:
            return b1.T @ h
        elif b2 is not Literal[1]:
            return h @ b2
        else:
            return h
    
    def transform_rdm1(self, r1, string):
        dm_dict = \
            {'AE': self.Ceo2ao, 'AL': self.Clo2ao, \
            'EA': self.Cao2eo, 'EL': self.Clo2eo, \
            'AM': self.Cmo2ao, 'MA': self.Cao2mo, \
            'LA': self.Cao2lo, \
            'AA': Literal[1], 'LL': Literal[1], 'EE': Literal[1]}
        rep1, rep2 = string.upper().split(',')
        try:
            b1 = dm_dict[rep1[0]+rep2[0]]
            b2 = dm_dict[rep1[1]+rep2[1]]
        except KeyError:
            raise NotImplementedError
        if b1 is not Literal[1] and b2 is not Literal[1]:
            return b1 @ r1 @ b2.T
        elif b1 is not Literal[1]:
            return b1 @ r1
        elif b2 is not Literal[1]:
            return r1 @ b2.T
        else:
            return r1

    transform_dm = transform_rdm1

    def transform_rdm2(self, rdm2, string):
        dm_dict = \
            {'AE': self.Ceo2ao, 'AL': self.Clo2ao, \
            'EA': self.Cao2eo, 'EL': self.Clo2eo, \
            'AM': self.Cmo2ao, 'MA': self.Cao2mo, \
            'AA': Literal[1], 'LL': Literal[1], 'EE': Literal[1]}
        rep1, rep2 = string.upper().split(',')
        for ib, (r1, r2) in enumerate(zip(rep1,rep2)):
            try:
                b_ =dm_dict[r1+r2]
            except KeyError:
                raise NotImplementedError
            if b_ is Literal[1]:
                pass
            else:
                if ib == 0:
                    rdm2 = einsum('ijkl,ai->ajkl', rdm2, b_)
                elif ib == 1:
                    rdm2 = einsum('ijkl,aj->iakl', rdm2, b_)
                elif ib == 2:
                    rdm2 = einsum('ijkl,ak->ijal', rdm2, b_)
                else:
                    rdm2 = einsum('ijkl,al->ijka', rdm2, b_)
        return rdm2

    def transform_eri(self, eri, string, norbs=None, contract_order=None):
        '''
        eri:   an eri or Mole object
        norbs: number of orbs of input eri
        '''
        eri_dict = \
            {'AE': self.Cao2eo, 'AL': self.Cao2lo, \
            'LE': self.Clo2eo, 'AM': self.Cao2mo, 'MA': self.Cmo2ao, \
            'AA': Literal[1], 'LL': Literal[1], 'EE': Literal[1]}
        rep1, rep2 = string.upper().split(',')
        try:
            len(norbs)
        except TypeError:
            norbs = [norbs]*4

        # for no-symmetry eri, ao2mo will just call einsum
        # so we better handle this case ourselves
        if hasattr(eri, 'ndim'):
            if eri.ndim == 4:
                if contract_order is  None:
                    contract_order = range(4)
                for ib in contract_order:
                    r1 = rep1[ib]
                    r2 = rep2[ib]
                    try:
                        b_ = eri_dict[r1+r2]
                    except KeyError:
                        raise NotImplementedError
                    shape = eri.shape
                    if b_ is not Literal[1]:
                        if ib == 0:
                            eri = dot(b_.T, eri.reshape(shape[0],-1)).reshape(-1, *shape[1:])
                        elif ib == 1:
                            eri = dot(
                                    b_.T, 
                                    eri.transpose(1,0,2,3).reshape(shape[1],-1)
                                    ).reshape(-1, shape[0], shape[2], shape[3])\
                                     .transpose(1,0,2,3)
                        elif ib == 2:
                            eri = dot(
                                    b_.T, 
                                    eri.transpose(2,1,0,3).reshape(shape[2],-1)
                                    ).reshape(-1, shape[1], shape[0], shape[3])\
                                     .transpose(2,1,0,3)
                        else:
                            eri = dot(eri.reshape(-1,shape[-1]), b_).reshape(*shape[:-1], -1)
                return eri

        # for eri with symmetry, go with ao2mo
        b = list()
        for norb, r1, r2 in zip(norbs, rep1,rep2):
            try:
                b_ = eri_dict[r1+r2]
            except KeyError:
                raise NotImplementedError
            if b_ is Literal[1]:
                b_ = np.eye(norb)
            b.append(b_)
        return ao2mo.general(eri, b, compact=False)

if __name__ == "__main__":
    basis = Basis()
    ovlp   = np.random.random((10,10))
    Cao2lo = np.random.random((10,10))
    rdm1   = np.random.random((10,10))
    class F:
        def __init__(self):
            self.env = [0,1,5,6,7,8,9]
            self.imp = [2,3,4]
            self.env_imp = np.ix_(self.env, self.imp)
    basis.make_ao_lo(ovlp, Cao2lo=Cao2lo)
    basis.make_lo_eo(rdm1, F())
    basis.make_ao_eo()
    
    h1 = np.random.random((10,10))

    norm = np.linalg.norm
    einsum = np.einsum
    assert norm(basis.transform_h(h1, 'aa,ee') \
              - einsum('uv,ui,vj->ij', h1, basis.Cao2eo, basis.Cao2eo)) < 1e-10
    assert norm(basis.transform_dm(rdm1, 'aa,ee') \
              - einsum('uv,iu,jv->ij', rdm1, basis.Ceo2ao, basis.Ceo2ao)) < 1e-10
    assert norm(basis.transform_dm(rdm1[:6,:6], 'ee,aa') \
            - einsum('ij,ui,vj->uv', rdm1[:6,:6], basis.Cao2eo, basis.Cao2eo)) < 1e-10
    assert norm(basis.transform_dm(rdm1[:6,:6], 'ee,ll') \
            - einsum('ts,it,js->ij', rdm1[:6,:6], basis.Clo2eo, basis.Clo2eo)) < 1e-10

    h2 = np.random.random((10,10,10,10))
    assert norm(basis.transform_eri(h2, 'aaaa,eaaa') \
            - einsum('ijkl,ia->ajkl', h2, basis.Cao2eo)) < 1e-10
    assert norm(basis.transform_eri(h2, 'aaaa,aeaa') \
            - einsum('ijkl,ja->iakl', h2, basis.Cao2eo)) < 1e-10
    assert norm(basis.transform_eri(h2, 'aaaa,aaea') \
            - einsum('ijkl,ka->ijal', h2, basis.Cao2eo)) < 1e-10
    assert norm(basis.transform_eri(h2, 'aaaa,aaae') \
            - einsum('ijkl,la->ijka', h2, basis.Cao2eo)) < 1e-10
