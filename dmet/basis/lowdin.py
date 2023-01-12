from dmet.basis import Basis
from dmet import linalg

import numpy as np

class LowdinBasis(Basis):
    def make_ao_lo(self, ovlp):
        self.eigh = linalg.EIGH(ovlp)
        val, u = self.eigh.kernel()
        Cao2lo = u @ np.diag(val**(-0.5)) @ u.T
        Basis.make_ao_lo(self, ovlp, Cao2lo)

    def grad_Cao2lo_ovlp(self, b_al):
        '''
        b_al contracted with dCao2lo/dovlp
        '''
        from pyscf.lib import einsum

        val, u = self.eigh.kernel()
        s1 = val**(-0.5)
        s3 = -0.5*val**(-1.5)

        temp = einsum('ui,ij->uj', b_al, u)
        temp1 = einsum('uj,j->uj', temp, s1)
        temp = einsum('ui,uj->ji', b_al, u)
        temp2 = einsum('ji,j->ij', temp, s1)
        res = self.eigh.gradient(temp1+temp2, contract='pre')[0]
        temp = einsum('ji,j->ji', temp, s3) 
        temp = einsum('ji,ij->j', temp, u) 
        res += einsum('j,wj,xj->wx', temp, u, u)
        
        return res

if __name__ == "__main__":
    norm = np.linalg.norm
    einsum = np.einsum

    ovlp = np.random.random((5,5))
    ovlp = ovlp + ovlp.T
    val, u = np.linalg.eigh(ovlp)
    ovlp = u @ np.diag(np.abs(val)) @ u.T
    dovlp = np.random.random((5,5))
    dovlp = dovlp + dovlp.T
    dovlp = 1e-3 * dovlp / norm(dovlp)

    b = LowdinBasis(); b.make_ao_lo(ovlp)
    b2 = LowdinBasis(); b2.make_ao_lo(ovlp + dovlp) 
    b1 = LowdinBasis(); b1.make_ao_lo(ovlp - dovlp)
    B_al = np.random.random((5,5))
    assert norm( einsum('ui,ui->', B_al, (b2.Cao2lo - b1.Cao2lo) / 2) \
            - einsum('uv,uv->', b.grad_Cao2lo_ovlp(B_al), dovlp) ) \
         < 1e-5 * norm(einsum('ui,ui->', B_al, (b2.Cao2lo - b1.Cao2lo) / 2))
