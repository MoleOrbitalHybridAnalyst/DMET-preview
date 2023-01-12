from dmet.solver import post_hf
from dmet.basis import Basis
from dmet.exception import SolverNotConvergedError

from pyscf.ci import cisd

class RCISD(post_hf.POST_HF):
    '''
    restricted CISD
    '''
    def kernel(self, h1e, eri, **kwargs):
        # HF
        post_hf.POST_HF.kernel(self, h1e, eri, **kwargs)

        # CISD
        self._solver = cisd.RCISD(self.mf)
        if self.conv_tol is not None:
            self._solver.conv_tol = self.conv_tol
        if self.max_cycle is not None:
            self._solver.max_cycle = self.max_cycle
        self.E, self.c = self._solver.kernel()
        if not self._solver.converged:
            raise SolverNotConvergedError()
        self.E += self.mf.e_tot

        self.r1 = self._solver.make_rdm1(self.c)
        self.r2 = self._solver.make_rdm2(self.c)

        self.r1 = self.basis.transform_rdm1(self.r1, 'mm,aa')
        self.r2 = self.basis.transform_rdm2(self.r2, 'mmmm,aaaa')

        return self.E, self.r1, self.r2
