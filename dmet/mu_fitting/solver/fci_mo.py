from dmet.mu_fitting.solver import SolverMuResponse

from pyscf.fci import cpci

import numpy as np

class FCIMOMuResponse(SolverMuResponse):
    def __init__(self, base, nimp, conv_tol=0.0001, max_cycle=20, verbose=0):
        super().__init__(base, nimp, conv_tol, max_cycle, verbose)

    @property
    def fci(self):
        return self.base._solver

    def check_consistency(self):
        from dmet.solver.fci_mo import FCIMO
        assert type(self.base) is FCIMO

    def dN_dc(self):
        fake_h1 = self.basis.Cao2mo[:self.nimp].T @ self.basis.Cao2mo[:self.nimp]
        fake_h1 += fake_h1.T
        return self.fci.contract_1e(fake_h1, self.base.c, self.norb, self.nelec)

    def dot_lambda(self):
        '''
        dot d M / d mu with lmbda
            for CI, d M / d mu = - dH c
        '''
        c = self.base.c
        norb = self.norb
        nelec = self.nelec

        dHc = -np.trace(self.base.r1[:self.nimp,:self.nimp]) * c    #  dE/dmu * c
        fake_h1 = self.basis.Cao2mo[:self.nimp].T @ self.basis.Cao2mo[:self.nimp]
        dHc += self.fci.contract_1e(fake_h1, c, norb, nelec)        # -dH/dmu @ c
        self.dHc = dHc
        def ldMdmu(lmbda):
            return - (dHc.ravel() @ lmbda.ravel())
        return ldMdmu

    def kernel(self, h1, h2, **kwargs):
        base = self.base
        c = base.c
        E = base.E

        h1 = self.basis.transform_h(h1, 'aa,mm')
        h2 = self.basis.transform_eri(h2, 'aaaa,mmmm')

        if "lmbda0" in kwargs:
            dci0 = kwargs["lmbda0"]
        elif hasattr(self, "lP"):
            if not self.base.change_sign:
                dci0 = self.lP
            else:
                dci0 = -self.lP
        else:
            dci0 = None
        if dci0 is not None and dci0.ndim > 1:
            dci0 = dci0.flatten()
        lP, stat = \
            cpci.solve(h1, h2, E, c, self.dN_dc(), self.norb, self.nelec, 
                    dci0=dci0, tol=self.conv_tol, max_cycle=self.max_cycle)
        if stat != 0:
            raise Exception("CP-CI not converged")

        self.lP = lP

        self.dNdmu = -self.dot_lambda()(self.lP)
        return self.dNdmu