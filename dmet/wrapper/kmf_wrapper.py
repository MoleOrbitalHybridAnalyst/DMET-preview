class MF:
    '''
    wrap a Gamma-point kmf such it looks like a molecular MF
    '''
    def __init__(self, kmf):
        self.kmf = kmf
        assert len(self.kmf.kpts) == 1
    def kernel(self):
        return self.kmf.kernel()
    def make_rdm1(self):
        return self.kmf.make_rdm1()[0].real
    def get_hcore(self):
        return self.kmf.get_hcore()[0].real
    def get_fock(self):
        return self.kmf.get_fock()[0].real
    def get_ovlp(self):
        return self.kmf.get_ovlp()[0].real
    @property
    def mo_coeff(self):
        return self.kmf.mo_coeff[0].real
    @property
    def _eri(self):
        if self.kmf._eri is None:
            nao = self.mol.nao
            self.kmf._eri = \
                self.kmf.with_df.get_ao_eri(compact=False).\
                reshape((nao,nao,nao,nao)).real
        return self.kmf._eri
    @property
    def mo_occ(self):
        return self.kmf.mo_occ[0]
    @property
    def mo_energy(self):
        return self.kmf.mo_energy[0]
    @property
    def mol(self):
        return self.kmf.mol
    def get_veff(self, dm=None):
        return self.kmf.get_veff(dm_kpts=[dm])[0].real
    def nuc_grad_method(self):
        return self.kmf.nuc_grad_method()
    def __getattr__(self, item):
        if item in self.__dict__:
            return getattr(self, item)
        elif hasattr(self.kmf, item):
            return getattr(self.kmf, item)
        else:
            raise AttributeError()
    def to_rhf(self):
        from pyscf.pbc.scf import KRHF
        mf = KRHF(self.kmf.cell)
        mf.__dict__.update(self.kmf.__dict__)
        mf.converged = False
        return MF(mf)
