from dmet.grad.solver.solver import SolverGradients
from dmet import DMET  
from dmet.exception import SolverNotConvergedError

from pyscf import lib
from pyscf.lib import einsum

class RHFGradients(SolverGradients):
    def check_consistency(self):
        from dmet.solver.rhf import RHF
        assert isinstance(self.base, RHF)

    def lambda_size(self):
        mf = self.base._solver
        nocc = sum(mf.mo_occ > 0)
        nvir = sum(mf.mo_occ == 0)
        return nvir * nocc

    '''
    separating dot_lambda, dE_dc and make_rdm12 from kernel 
    is needed for nuc_grad with mufit
    '''

    def dot_lambda(self, h1, h2):
        '''
        now this is basically a copy of pyscf.scf.cphf
        '''
        mf = self.base._solver
        mo_energy = mf.mo_energy
        occ = mf.mo_occ > 0
        vir = ~occ
        vo_shape = (sum(vir), sum(occ))

        e_a = mo_energy[vir]
        e_i = mo_energy[occ]
        e_ai = lib.direct_sum('a-i->ai', e_a, e_i)

        def fvind(dm_mo_VO):
            dm = mf.mo_coeff[:, vir] @ dm_mo_VO @ mf.mo_coeff.T[occ]
            dm = dm + dm.T
            v = mf.get_veff(dm=dm)
            v_mo = mf.mo_coeff.T[vir] @ v @ mf.mo_coeff[:,occ]
            return v_mo * 2
        def vind_vo(Z):
            '''
            [(e_a - e_i) delta_{ai,bj} + A_{ai,bj}] Z_bj
            = Z_ai + fvind(Z)
            '''
            v = fvind(Z.reshape(vo_shape)).reshape(vo_shape)
            v += Z.reshape(vo_shape) * e_ai
            return v.ravel()
        return vind_vo

    def dE_dc(self, h1pp, h2w):
        '''
        d E^DMET / d U_ai 
        = d E^DMET / d C_pi C_pa = 2 (F_ai + F_ia)
        '''
        base = self.base
        mf = base._solver
        occ = mf.mo_occ > 0
        vir = ~occ

        F = h1pp + DMET.get_veff(h2w, base.r1)
        F += F.T
        Fvo = einsum('uv,ua,vi->ai', F, 
            mf.mo_coeff[:,vir], mf.mo_coeff[:,occ])

        return 2 * Fvo 

    def make_rdm12(self, z):
        base = self.base
        mf = base._solver
        occ = mf.mo_occ > 0
        vir = ~occ
        vo_shape = (sum(vir), sum(occ))
        z = -z.reshape(vo_shape)

        # z d Fvo / d h1 
        r1bar = einsum('ai,ua,vi->uv', z, 
                mf.mo_coeff[:,vir], mf.mo_coeff[:,occ])

        # z d Fvo / d h2
        r2bar =  einsum('uv,wx->uvwx', r1bar, base.r1)
        r2bar -= 0.5 * einsum('ux,vw->uvwx', r1bar, base.r1)
        r2bar += r2bar.transpose(2,3,0,1)

        return r1bar, r2bar


    def kernel(self, h1pp, h2w, h1, h2, **kwargs):
        import scipy.sparse.linalg as sla

        base = self.base
        # NOTE that h1 and h2 are not used since we have mf
        mf = base._solver
        # TODO redo mf if h1 and h2 is not consistent with mf?
        assert mf.get_hcore() is h1
        assert mf._eri is h2
        occ = mf.mo_occ > 0
        vir = ~occ
        vo_shape = (sum(vir), sum(occ))
        vo_size = self.lambda_size()

        # cphf
        M = sla.LinearOperator((vo_size,vo_size), self.dot_lambda(h1, h2))
        dEdc = self.dE_dc(h1pp, h2w).ravel()
        self.Z, stat = sla.gmres(M, dEdc, tol=self.conv_tol, 
            atol=self.conv_tol*lib.norm(dEdc), maxiter=self.max_cycle)
        if stat != 0:
            raise SolverNotConvergedError("CP-HF not converged")
        self.Z = self.Z.reshape(vo_shape)

        return self.make_rdm12(self.Z)
