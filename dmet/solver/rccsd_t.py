from dmet.solver import post_hf
from dmet.basis import Basis
from dmet.exception import SolverNotConvergedError

from pyscf.cc import rccsd

class RCCSD_T(post_hf.POST_HF):
    '''
    restricted CCSD(T)
    '''
    def make_rdm12(self):
        from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
        from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm
        mycc = self._solver
        eris = mycc.ao2mo()
        conv, self.l1, self.l2 = ccsd_t_lambda.kernel(mycc, eris, mycc.t1, mycc.t2)
        if not conv:
            raise SolverNotConvergedError()
        self.r1 = ccsd_t_rdm.make_rdm1(
            mycc, self.t1, self.t2, self.l1, self.l2, eris=eris)
        self.r2 = ccsd_t_rdm.make_rdm2(
            mycc, self.t1, self.t2, self.l1, self.l2, eris=eris)
        return self.r1, self.r2

    def kernel(self, h1e, eri, **kwargs):
        # HF
        post_hf.POST_HF.kernel(self, h1e, eri, **kwargs)

        # CCSD
        self._solver = rccsd.RCCSD(self.mf)
        if self.conv_tol is not None:
            self._solver.conv_tol = self.conv_tol
        if self.max_cycle is not None:
            self._solver.max_cycle = self.max_cycle
        self.e_corr, self.t1, self.t2 = self._solver.kernel()
        if not self._solver.converged:
            raise SolverNotConvergedError()
        self.e_corr += self._solver.ccsd_t()
        self.E = self.e_corr + self.mf.e_tot

        self.make_rdm12()
        self.r1 = self.basis.transform_rdm1(self.r1, 'mm,aa')
        self.r2 = self.basis.transform_rdm2(self.r2, 'mmmm,aaaa')

        return self.E, self.r1, self.r2
