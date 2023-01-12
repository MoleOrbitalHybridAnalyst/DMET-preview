from dmet.mu_fitting.solver import SolverMuResponse

from pyscf.scf import cphf

import numpy as np

class RHFMuResponse(SolverMuResponse):
    def __init__(self, base, nimp, conv_tol=0.0001, max_cycle=20, verbose=0):
        super().__init__(base, nimp, conv_tol, max_cycle, verbose)

    @property
    def mf(self):
        return self.base._solver

    def check_consistency(self):
        from dmet.solver.rhf import RHF
        assert type(self.base) is RHF

    def dN_dc(self):
        '''
        d N / d U_ai = 
        d N / d C^MO_pi  C^MO_pa
        '''
        occ = self.mf.mo_occ>0
        vir = self.mf.mo_occ==0
        return \
            2 * np.einsum('pi,pa->ai', 
                self.mf.mo_coeff[:self.nimp, occ] * self.mf.mo_occ[occ],
                self.mf.mo_coeff[:self.nimp, vir])

    def dot_lambda(self):
        '''
        dot d M / d mu with lmbda
        '''
        CC = np.einsum('pa,pi->ai', 
            self.mf.mo_coeff[:self.nimp, self.mf.mo_occ==0], 
            self.mf.mo_coeff[:self.nimp, self.mf.mo_occ>0])
        def ldMdmu(lmbda):
            return - (CC.ravel() @ lmbda.ravel())
        return ldMdmu

    def kernel(self, h1, h2, **kwargs):
        mf = self.mf
        occ = mf.mo_occ>0
        vir = mf.mo_occ==0

        # cphf
        def fvind(dm_mo_VO):
            dm = mf.mo_coeff[:, vir] @ dm_mo_VO @ mf.mo_coeff.T[occ]
            dm = dm + dm.T
            v = mf.get_veff(dm=dm)
            v_mo = mf.mo_coeff.T[vir] @ v @ mf.mo_coeff[:,occ]
            return v_mo * 2
        self.Z = cphf.solve(fvind, mf.mo_energy, mf.mo_occ, self.dN_dc(), 
                tol=self.conv_tol, max_cycle=self.max_cycle, verbose=self.verbose)[0]

        # TODO convergence check for cphf

        self.dNdmu = self.dot_lambda()(self.Z)
        return self.dNdmu