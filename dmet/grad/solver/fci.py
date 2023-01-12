from dmet.grad.solver.solver import SolverGradients

from pyscf.fci import cpci
from pyscf.fci import direct_spin1

class FCIGradients(SolverGradients):
    def lambda_size(self):
        return self.base.solution.size

    def check_consistency(self):
        from dmet.solver.fci import FCI
        assert type(self.base) is FCI

    def dE_dc(self, h1pp, h2w):
        '''
        compute d Edmet / d c
        '''
        fci = self.base._solver
        h2eE = fci.absorb_h1e(h1pp, h2w, self.norb, self.base.nelec, 0.5)
        return 2 * fci.contract_2e(h2eE, self.base.c, self.norb, self.base.nelec)

    def dot_lambda(self, h1, h2):
        '''
        dot d M / d c with lambda
        where c is the impurity solution
              M(c) = 0 is the solver equation
              lmbda is the multiplier of c

        NOTE this is hidden in pyscf.fci.cpci.solver
             so kind of replicated if this function is exactly wanted
        TODO reduce this duplication somehow
        '''
        fci = self.base._solver

        lidxa, lidxb = direct_spin1._unpack(self.norb, self.nelec, None)
        Na = lidxa.shape[0]; Nb = lidxb.shape[0]

        h2e = fci.absorb_h1e(h1, h2, self.norb, self.nelec, 0.5)

        def ldMdc(lmbda):
            dc = lmbda.ravel()
            cdc = self.base.c.reshape(-1) @ dc
            Mdc = 2 * (cdc * self.base.c)
            dc = dc.reshape((Na, Nb))
            Mdc += fci.contract_2e(h2e, dc, self.norb, self.nelec, (lidxa,lidxb))
            Mdc -= (self.base.E * dc)
            return Mdc

        return ldMdc

    def make_rdm12(self, lmbda):
        lPcP = lmbda.ravel() @ self.base.c.ravel()
        r1, r2 = self.base._solver.trans_rdm12(
            lmbda, self.base.c, self.norb, self.nelec)
        r1bar = lPcP * self.base.r1 - r1
        r2bar = lPcP * self.base.r2 - r2
        return r1bar, r2bar

    def kernel(self, h1pp, h2w, h1, h2, **kwargs):
        '''
        given h1'' and h2w
        such that dEDMET/dc = h1'' dr1/dc + h2w dr2/dc

        return r1bar and r2bar 
        such that dEDMET/dc * dc/dR = r1bar * dh1/dR + 0.5 * r2bar * dh2/dR
        where h1 and h2 are really system's Hamiltionian
        '''
        base = self.base
        fci = base._solver
        c = base.c
        E = base.E

        self.bP = self.dE_dc(h1pp, h2w)

        if "dci0" in kwargs:
            dci0 = kwargs["dci0"]
        else:
            dci0 = None

        self.lP = self.solve_lambda(h1, h2, dci0=dci0, **kwargs)

        r1bar, r2bar = self.make_rdm12(self.lP)

        return r1bar, r2bar

    def solve_lambda(self, h1, h2, dci0=None):
        lP, stat = \
            cpci.solve(h1, h2, self.base.E, self.base.c, self.bP, 
                    self.norb, self.nelec, dci0=dci0, 
                    tol=self.conv_tol, max_cycle=self.max_cycle)
        if stat != 0:
            raise Exception("CP-CI not converged")
        
        return lP
