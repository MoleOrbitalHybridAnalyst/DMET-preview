from dmet import basis
from dmet.solver import Solver
from dmet.exception import SCFNotConvergedError

from pyscf import ao2mo, gto, scf
from pyscf.lib import einsum, logger

import numpy as np
from copy import copy 

class DMET:
    from dmet.solver import solver_dict
    lo_dict = {"Lowdin": basis.LowdinBasis}

    def __init__(self, mf, fragments, 
            localization="Lowdin", solver="FCI", bath_tol=-1,
            conv_tol=None, max_cycle=None,
            cache4grad=False, verbose=None):
        '''
        mf:         global mean field object
        fragments:  list of Fragment defining what are impurities
        cache4grad: save some intermediate results for grad
        '''
        from sys import stdout

        # sanity check 
        assert isinstance(fragments, list)
        from pyscf.dft.rks import KohnShamDFT
        if isinstance(mf, KohnShamDFT) or \
            (hasattr(mf, 'kmf') and isinstance(mf.kmf, KohnShamDFT)):
            self.dft = mf
            self.mf = self.dft.to_rhf()
        else:
            self.mf = mf
        if solver is None:
            for f in fragments:
                try:
                    f.solver
                except:
                    raise Exception("cannot find solver info anywhere")
                assert isinstance(f.solver, Solver)
        elif solver not in self.solver_dict:
            raise NotImplementedError
        if localization not in self.lo_dict:
            raise NotImplementedError

        self.mol = mf.mol
        self.fragments = fragments
        self.conv_tol = conv_tol
        self.max_cycle = max_cycle
        self.cache4grad = cache4grad
        self.lo_method = self.lo_dict[localization]
        self.bath_tol = bath_tol
        if solver is not None:
            self.solver_method = self.solver_dict[solver]
        if verbose is None:
            self.verbose = mf.mol.verbose
        self.log = logger.Logger(stdout, self.verbose)
        logger.TIMER_LEVEL = 3

        if solver is not None:
            for f in self.fragments:
                f.solver = self.solver_method(
                        verbose=self.verbose, 
                        conv_tol=self.conv_tol, max_cycle=self.max_cycle)

        # attributes to be computed
        self.basis = None
        self.Edmet_tot = None

    @staticmethod
    def get_veff(eri, dm):
        return einsum('ijkl,lk->ij', eri, dm) \
                - 0.5 * einsum('iklj,kl->ij', eri, dm)

    def kernel(self, cache4grad=None):
        if cache4grad is not None:
            self.cache4grad = cache4grad

        # decrease some depth
        fragments = self.fragments

        # global mean-field
        if hasattr(self, 'dft'):
            mf = self.dft
        else:
            mf = self.mf
        if not mf.converged:
            if hasattr(mf, 'with_df') and mf.with_df is not None:
                mf.with_df.build()
            self.Ehf = mf.kernel()
        else:
            self.Ehf = mf.e_tot
        if not mf.converged:
            raise SCFNotConvergedError()
        self.aodm = mf.make_rdm1()
        self.aohcore = mf.get_hcore()
        if hasattr(self, 'dft'):
            self.mf = mf.to_rhf()
        
        mf = self.mf
        self.aofock = mf.get_fock() 
        
        # make basis
        basis = self.lo_method()
        basis.make_ao_lo(mf.get_ovlp())
        basis.Cao2mo = mf.mo_coeff
        self.basis = basis
        lodm = basis.transform_dm(self.aodm, 'aa,ll')

        ## make emb basis ##
        # for each fragment, get the AO to EO coefficients
        for ifrag, f in enumerate(fragments):
            f.basis = copy(basis)
            f.basis.make_lo_eo(lodm, f, bath_tol=self.bath_tol)
            f.basis.make_ao_eo()

        ## Embedding H2 ##
        # for each fragment, project eri into EO and set f.h2
        self.make_h2()

        ## Embedding H1 ##
        # construct h1 for each fragment
        self.make_h1()

        ## Solve impurity problems ##
        self.solve_impurity()
        
        ## Using impurity solution to comptue DMET energy ##
        self.Edmet_tot = self.compute_energy()
        
        if self.cache4grad:
            self.lodm = lodm

        return self.Edmet_tot + mf.energy_nuc()

    def transform_eri(self):
        mf = self.mf
        if mf._eri is not None:
            self.log.note("ERI transform using incore")
            #mf._eri = ao2mo.restore(1, mf._eri, mf.mol.nao)
        else:
            self.log.note("ERI transform using outcore")
            if hasattr(mf, 'with_df') and mf.with_df is not None:
                self.log.warn("Mean-field has DF; you want dmet.density_fit maybe?")

        for ifrag, f in enumerate(self.fragments):
            nao = mf.mol.nao
            neo = f.basis.Cao2eo.shape[-1]

            ''' without DF '''
            if self.cache4grad:
                cput0 = (logger.process_clock(), logger.perf_counter())
                if mf._eri is not None:
                    f.eri_aeee = \
                        f.basis.transform_eri(mf._eri, 'aaaa,aeee', nao)
                else:
                    f.eri_aeee = \
                        f.basis.transform_eri(mf.mol, 'aaaa,aeee', nao)
                if f.eri_aeee.ndim == 2:
                    dim1, dim2 = f.eri_aeee.shape
                    f.eri_aeee = f.eri_aeee.reshape(dim1, neo, neo)
                    f.eri_aeee = f.eri_aeee.reshape(-1, neo, neo, neo)
                self.log.timer(f'ERI transformation of fragment {ifrag}', *cput0)
                f.h2 = einsum('ujkl,ui->ijkl', f.eri_aeee, f.basis.Cao2eo)
            else:
                if self.mf._eri is not None:
                    f.h2 = f.basis.transform_eri(mf._eri, 'aaaa,eeee')
                else:
                    f.h2 = f.basis.transform_eri(mf.mol, 'aaaa,eeee')
                f.h2 = ao2mo.restore(1, f.h2, neo)
    
    make_h2 = transform_eri

    def make_h1(self):
        mf = self.mf
        for ifrag, f in enumerate(self.fragments):
            nao = mf.mol.nao
            neo = f.basis.Cao2eo.shape[-1]

            ## Embedding H1 ##
            # project global 1rdm into EO
            f.dm_proj = f.basis.transform_dm(self.aodm, 'aa,ee') 
            # local veff build
            vloc = self.get_veff(f.h2, f.dm_proj)
            if self.cache4grad:
                f.vloc = vloc
            # H1 = F_proj - veff_loc
            f.h1  = f.basis.transform_h(self.aofock, 'aa,ee')
            f.h1 -= vloc

    def solve_impurity(self):
        mf = self.mf
        for ifrag, f in enumerate(self.fragments):
            ## Solve impurity problem ##
            f.solver.nelec = min(2 * f.nimp, mf.mol.nelectron) # restricted and half-filling
            f.Esolv, f.r1, f.r2 = f.solver.kernel(f.h1, f.h2, dm0=f.dm_proj)

    def compute_energy(self, new_aodm=None, new_aoveff=None):
        fragments = self.fragments
        if new_aoveff is None:
            # will build new_aoveff from new_aodm
            if new_aodm is None:
                new_aodm = 0
                for ifrag, f in enumerate(fragments):
                    neo = f.basis.Cao2eo.shape[-1]

                    f.set_w(neo)
                    f.r1w = f.r1 * f.w1
                    f.r2w = f.r2 * f.w2

                    # back project emb r1 to get new_aodm
                    new_aodm += f.basis.transform_dm(f.r1w, 'EE,AA')
                if self.cache4grad:
                    self.new_aodm = new_aodm
            else:
                pass # use input ao_dm

            # global fock build with improved global dm
            new_aoveff = self.mf.get_veff(dm=new_aodm)
            if self.cache4grad:
                self.new_aoveff = new_aoveff
        else:
            pass # use input new_aoveff

        Edmet_tot = 0
        for f in fragments:
            ## Embedding H1 for energy ##
            if not self.cache4grad:
                f.h1p = f.basis.transform_h(self.aohcore+0.5*new_aoveff, 'aa,ee')
            else:
                f.hcore = f.basis.transform_h(self.aohcore, 'aa,ee')
                f.new_vproj = f.basis.transform_h(new_aoveff, 'aa,ee')
                f.h1p = f.hcore + 0.5 * f.new_vproj
            new_vloc = self.get_veff(f.h2, f.r1)
            f.h1p -= 0.5 * new_vloc

            ## DMET energy ##
            f.Edmet  = einsum('ji,ij->', f.r1w, f.h1p)
            #f.Edmet += 0.5 * einsum('ijkl,ijkl->', f.r2w, f.h2)
            f.Edmet += 0.5 * f.r2w.flatten() @ f.h2.flatten()

            Edmet_tot += f.Edmet
        return Edmet_tot

    def nelectron_tot(self):
        nelec = 0
        for f in self.fragments:
            assert hasattr(f, "r1")
            nelec += np.trace(f.r1[:f.nimp,:f.nimp])
        return nelec

    def density_fit(self, df="GDF"):
        if df == "GDF":
            from dmet.df.gdf import GDF_DMET
            return GDF_DMET(self)
        else:
            raise NotImplementedError()

    def mu_fitting(self, *args, mufit="Newton-Raphson", **kwargs):
        if mufit == "Newton-Raphson" or mufit == "Newton_Raphson":
            from dmet.mu_fitting.newton_raphson import MufitNewton_DMET
            return MufitNewton_DMET(self, *args, **kwargs)
        elif mufit == "Secant" or mufit == "secant":
            from dmet.mu_fitting.secant import MufitSecant_DMET
            return MufitSecant_DMET(self, *args, **kwargs)
        else:
            raise NotImplementedError()

    def nuc_grad_method(self, **kwargs):
        from dmet.grad import DMETGradients
        return DMETGradients(self, **kwargs)

    def as_scanner(self):
        from pyscf.lib import SinglePointScanner

        class DMET_Scanner(self.__class__, SinglePointScanner):
            def __init__(self, dmet_obj):
                self.dmet_obj = dmet_obj

            def __getattr__(self, item):
                '''
                we need this because some attributes are added 
                at run-time
                '''
                if item in self.__dict__:
                    return getattr(self, item)
                elif item in self.dmet_obj.__dict__:
                    return getattr(self.dmet_obj, item)
                else:
                    raise AttributeError()

            def reset(self, mol=None):
                if mol is not None:
                    self.mol = mol
                self.mf.reset(mol)

            def __call__(self, mol, **kwargs):
                self.reset(mol)

                dm0 = None
                if 'dm0' in kwargs:
                    dm0 = kwargs.pop('dm0')
                elif self.mf.mo_coeff is not None:
                    dm0 = self.mf.make_rdm1()

                self.mf.kernel(dm0=dm0)
                E = self.dmet_obj.kernel(**kwargs)
                return E

        return DMET_Scanner(self)

if __name__ == "__main__":
    mol = gto.Mole()
    mol.atom =  '''
       O  8.70835e-01  6.24543e+00  5.08445e+00
       H  8.51421e-01  6.33649e+00  6.05969e+00
       H  1.66206e+00  5.60635e+00  5.02479e+00
       O  2.38299e+00  8.36926e+00  4.10083e+00
       H  1.76679e+00  7.63665e+00  4.17552e+00
       H  2.40734e+00  8.80363e+00  4.99023e+00
       O  2.41917e+00  1.04168e+01  2.48601e+00
       H  2.55767e+00  9.70422      3.12008e+00
       H  3.10835e+00  1.02045e+01  1.83352e+00 '''
    mol.basis = 'sto-3g'
    mol.build()

    from pyscf.scf import RHF
    mf = RHF(mol)

    from dmet.fragment import Fragment
    frags = list()
    for i in range(9):
        frag = Fragment()
        frag.set_imp_by_atom(mol, [i])
        frags.append(frag)
    d = DMET(mf, frags, cache4grad=True)
    Edmet = d.kernel()

    grad = d.nuc_grad_method()
    gdmet = grad.kernel()

