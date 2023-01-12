from dmet.solver import Solver
from dmet.exception import SolverNotConvergedError
from pyscf.fci import direct_spin1

class FCI(Solver):
    '''
    restricted Full CI solver
    '''
    def __init__(self, **kwargs):
        Solver.__init__(self, **kwargs)
        self._solver = direct_spin1.FCI()
        if self.verbose is not None:
            self._solver.verbose = self.verbose
        if self.conv_tol is not None:
            self._solver.conv_tol = self.conv_tol
        if self.max_cycle is not None:
            self._solver.max_cycle = self.max_cycle

    @property
    def solution(self):
        return self.c

    def make_rdm12(self):
        return self._solver.make_rdm12(self.c, self.norb, self.nelec)

    def kernel(self, h1e, eri, dm0=None, **kwargs):
        if self.nelec is None:
            nelec = h1e.shape[0] # restricted and half-filling
        else:
            nelec = self.nelec
        norb  = h1e.shape[0]
        if hasattr(self, 'c'):
            c0 = self.c[0][0]
        else:
            c0 = 0
        self.E, self.c = \
            self._solver.kernel(h1e, eri, norb, nelec, **kwargs)
        if not self._solver.converged:
            raise SolverNotConvergedError()

        # TODO TBH this is kind of ugly
        if c0 * self.c[0][0] > 0:
            self.change_sign = False
        else:
            self.change_sign = True

        self.nelec = nelec
        self.norb = norb
        self.r1, self.r2 = self.make_rdm12()

        return self.E, self.r1, self.r2

    def gradient(self, **kwargs):
        from dmet.grad.solver.fci import FCIGradients
        return FCIGradients(self, **kwargs)

    def mu_response(self, nimp, **kwargs):
        from dmet.mu_fitting.solver.fci import FCIMuResponse
        return FCIMuResponse(self, nimp, **kwargs)
