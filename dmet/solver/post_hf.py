from dmet.solver import Solver
from dmet.solver.rhf import RHF
from dmet.basis import Basis

class POST_HF(Solver):
    def kernel(self, h1e, eri, **kwargs):
        if self.nelec is None:
            self.nelec = h1e.shape[0] # restricted and half-filling

        # HF
        if self.conv_tol is not None:
            hf_conv_tol = 1e-2 * self.conv_tol # TODO allow flexible tol for HF
        else:
            hf_conv_tol = None
        if 'hf_conv_tol' in kwargs:
            hf_conv_tol = kwargs.pop('hf_conv_tol')
        if 'hf_max_cycle' in kwargs:
            hf_max_cycle = kwargs.pop('hf_max_cycle')
        else:
            hf_max_cycle = None

        rhf = RHF(nelec=self.nelec, verbose=self.verbose, 
                conv_tol=hf_conv_tol, max_cycle=hf_max_cycle)
        self.rhf = rhf
        self.Ehf = rhf.kernel(h1e, eri, **kwargs)[0]
        self.mf = rhf._solver

        # define AO to MO basis
        mf = rhf._solver
        self.basis = Basis()
        self.basis.make_ao_mo(mf.mo_coeff, mf.get_ovlp(mf.mol))
