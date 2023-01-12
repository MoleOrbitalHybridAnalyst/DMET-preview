from dmet.grad.solver.fci import FCIGradients

from pyscf.fci import cpci
from pyscf.fci import direct_spin1

class FCIMOGradients(FCIGradients):
    def check_consistency(self):
        from dmet.solver.fci_mo import FCIMO
        assert type(self.base) is FCIMO

    @property
    def fci(self):
        return self.base._solver

    def dE_dc(self, h1pp, h2w):
        h1pp = self.basis.transform_h(h1pp, 'aa,mm')
        h2w = self.basis.transform_eri(h2w, 'aaaa,mmmm')

        h2eE = self.fci.absorb_h1e(h1pp, h2w, self.norb, self.nelec, 0.5)
        return 2 * self.fci.contract_2e(h2eE, self.base.c, self.norb, self.nelec)

    def dot_lambda(self, h1, h2):
        lidxa, lidxb = direct_spin1._unpack(self.norb, self.nelec, None)
        Na = lidxa.shape[0]; Nb = lidxb.shape[0]

        h1 = self.basis.transform_h(h1, 'aa,mm')
        h2 = self.basis.transform_eri(h2, 'aaaa,mmmm')
        h2e = self.fci.absorb_h1e(h1, h2, self.norb, self.nelec, 0.5)

        def ldMdc(lmbda):
            dc = lmbda.ravel()
            cdc = self.base.c.reshape(-1) @ dc
            Mdc = 2 * (cdc * self.base.c)
            dc = dc.reshape((Na, Nb))
            Mdc += self.fci.contract_2e(h2e, dc, self.norb, 
                                        self.nelec, (lidxa,lidxb))
            Mdc -= (self.base.E * dc)
            return Mdc

        return ldMdc

    def make_rdm12(self, lmbda):
        lPcP = lmbda.ravel() @ self.base.c.ravel()
        r1, r2 = self.fci.trans_rdm12(
            lmbda, self.base.c, self.norb, self.nelec)
        r1 = self.basis.transform_rdm1(r1, 'mm,aa')
        r2 = self.basis.transform_rdm2(r2, 'mmmm,aaaa')
        r1bar = lPcP * self.base.r1 - r1
        r2bar = lPcP * self.base.r2 - r2
        return r1bar, r2bar

    def kernel(self, h1pp, h2w, h1, h2, **kwargs):
        base = self.base
        fci = base._solver
        c = base.c
        E = base.E
        basis = base.basis

        # rotate h'' and h2w into MO
        h1 = basis.transform_h(h1, 'aa,mm')
        h2 = basis.transform_eri(h2, 'aaaa,mmmm')

        if "dci0" in kwargs:
            dci0 = kwargs["dci0"]
        else:
           dci0 = None
        lP, stat = \
            cpci.solve(h1, h2, E, c, self.dE_dc(h1pp, h2w), self.norb, self.nelec, 
                    dci0=dci0, tol=self.conv_tol, max_cycle=self.max_cycle)
        if stat != 0:
            raise Exception("CP-CI not converged")

        self.lP = lP

        return self.make_rdm12(self.lP)
