from dmet.mu_fitting.solver import SolverMuResponse

from pyscf.fci import cpci

import numpy as np

class FCIMuResponse(SolverMuResponse):
    def __init__(self, base, nimp, conv_tol=0.0001, max_cycle=20, verbose=0):
        super().__init__(base, nimp, conv_tol, max_cycle, verbose)
        self.fci = self.base._solver

    def check_consistency(self):
        from dmet.solver.fci import FCI
        assert type(self.base) is FCI

    def dN_dc(self):
        norb = self.base.norb
        nelec = self.base.nelec

        fake_h1 = np.zeros((norb,norb))
        fake_h1[(range(self.nimp),range(self.nimp))] = 2
        return self.fci.contract_1e(fake_h1, self.base.c, norb, nelec)

    def dot_lambda(self):
        '''
        dot d M / d mu with lmbda
            for CI, d M / d mu = - dH c
        '''
        nimp = self.nimp
        c = self.base.c
        norb = self.norb
        nelec = self.nelec
        r1 = self.base.r1

        dHc = -np.trace(r1[:nimp,:nimp]) * c        # dE * c
        dh1 = np.zeros((norb,norb))
        dh1[(range(nimp),range(nimp))] = -1
        dHc -= self.fci.contract_1e(dh1, c, norb, nelec) # dH * c
        def ldMdmu(lmbda):
            return - (dHc.ravel() @ lmbda.ravel())
        return ldMdmu

    def kernel(self, h1, h2, **kwargs):
        # TODO more clever way via not computing useless responses
        base = self.base
        c = base.c
        E = base.E

        if "lmbda0" in kwargs:
            dci0 = kwargs["lmbda0"]
        elif hasattr(self, "lP"):
            if self.base.change_sign:
                '''
                if the solver solution change sign
                the last saved lambda should be negative
                to form a good initial guess
                '''
                dci0 = -self.lP
            else:
                dci0 = self.lP
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