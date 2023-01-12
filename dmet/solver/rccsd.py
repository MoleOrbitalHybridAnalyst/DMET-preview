from dmet.solver import post_hf
from dmet.basis import Basis
from dmet.exception import SolverNotConvergedError

from pyscf.cc import rccsd

class RCCSD(post_hf.POST_HF):
    '''
    restricted CCSD
    '''
    def make_rdm12(self):
        self.l1, self.l2 = self._solver.solve_lambda(self.t1, self.t2)
        if not self._solver.converged_lambda:
            raise SolverNotConvergedError()
        self.r1 = self._solver.make_rdm1(self.t1, self.t2, self.l1, self.l2)
        self.r2 = self._solver.make_rdm2(self.t1, self.t2, self.l1, self.l2)
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
        self.E = self.e_corr + self.mf.e_tot

        self.make_rdm12()
        self.r1 = self.basis.transform_rdm1(self.r1, 'mm,aa')
        self.r2 = self.basis.transform_rdm2(self.r2, 'mmmm,aaaa')

        return self.E, self.r1, self.r2


class RCCSD_APPROX(RCCSD):
    '''
    restricted CCSD but not solving lambda
    '''
    def make_rdm12(self):
        self.r1 = self._solver.make_rdm1(self.t1, self.t2, self.t1, self.t2)
        self.r2 = self._solver.make_rdm2(self.t1, self.t2, self.t1, self.t2)
        return self.r1, self.r2
