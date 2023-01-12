from dmet.solver.fci import FCI
from dmet.solver import post_hf
from dmet.exception import *

class FCIMO(FCI, post_hf.POST_HF):
    '''
    restricted Full CI after HF
    '''
    def kernel(self, h1e, eri, dm0=None, **kwargs):
        # HF
        post_hf.POST_HF.kernel(self, h1e, eri, dm0=dm0, **kwargs)

        norb = h1e.shape[0]

        # FCI
        h1e = self.basis.transform_h(h1e, 'aa,mm')
        eri = self.basis.transform_eri(eri, 'aaaa,mmmm')

        if hasattr(self, "c"):
            c0 = self.c[0][0]
        else:
            c0 = 0

        self.E, self.c = \
            self._solver.kernel(h1e, eri, norb, self.nelec, **kwargs)
        if not self._solver.converged:
            raise SolverNotConvergedError()

        if c0 * self.c[0][0] > 0:
            self.change_sign = False
        else:
            self.change_sign = True

        self.r1, self.r2 = self._solver.make_rdm12(self.c, norb, self.nelec)

        self.r1 = self.basis.transform_rdm1(self.r1, 'mm,aa')
        self.r2 = self.basis.transform_rdm2(self.r2, 'mmmm,aaaa')

        return self.E, self.r1, self.r2

    def gradient(self, **kwargs):
        from dmet.grad.solver.fci_mo import FCIMOGradients
        return FCIMOGradients(self, **kwargs)

    def mu_response(self, nimp, **kwargs):
        from dmet.mu_fitting.solver.fci_mo import FCIMOMuResponse
        return FCIMOMuResponse(self, nimp, **kwargs)