from dmet.solver import Solver
from dmet.exception import SolverNotConvergedError

from pyscf import scf
from pyscf.lib import einsum
from pyscf.gto import Mole

import numpy as np

class RHF(Solver):
    '''
    restricted HF solver
    '''

    def kernel(self, h1e, eri, dm0=None, **kwargs):
        if self.nelec is None:
            nelec = h1e.shape[0] # restricted and half-filling
        else:
            nelec = self.nelec
        norb = h1e.shape[0]

        mol = Mole()
        mol.nelectron = self.nelec
        if self.verbose is not None:
            mol.verbose = self.verbose
        mol.incore_anyway = True
        mol.build()

        # use a derived class to avoid overwritten warnings
        class _RHF(scf.rhf.RHF):
             get_hcore = lambda *args: h1e
             get_ovlp = lambda *args: np.eye(norb)
        mf = _RHF(mol)
        mf._eri = eri
        if self.conv_tol is not None:
            mf.conv_tol = self.conv_tol
        if dm0 is None:
            mf.init_guess = '1e'
            self.E = mf.kernel(dump_chk=False) # not dump check to avoid warn
        else:
            self.E = mf.kernel(dm0=dm0, dump_chk=False)
        self._solver = mf
        if not mf.converged:
            raise SolverNotConvergedError()

        self.r1 = mf.make_rdm1()
        self.r2 =  einsum('uv,wx->uvwx', self.r1, self.r1)
        self.r2 -= 0.5 * einsum('ux,vw->uvwx', self.r1, self.r1)

        return self.E, self.r1, self.r2

    def gradient(self, **kwargs):
        from dmet.grad.solver.rhf import RHFGradients
        return RHFGradients(self, **kwargs)

    def mu_response(self, nimp, **kwargs):
        from dmet.mu_fitting.solver.rhf import RHFMuResponse
        return RHFMuResponse(self, nimp, **kwargs)
