import numpy as np
import scipy.linalg as la

from pyscf.lib import einsum, dot, logger, unpack_tril

class DMETGradients:
    def __init__(self, base, conv_tol_solver=None, conv_tol_cphf=None, 
            verbose=None, **kwargs):
        '''
        base: DMET object
        '''
        if base.cache4grad is False:
            raise NotImplementedError(
                "please turn on cache4grad when computing energy")
        self.base = base
        self.mol = base.mol
        self.mfgrad = base.mf.nuc_grad_method()
        if hasattr(base, 'dft'):
            self.dftgrad = base.dft.nuc_grad_method()
        self.conv_tol_solver = conv_tol_solver
        self.conv_tol_cphf = conv_tol_cphf
        if 'auxbasis_response' in kwargs:
            self.auxbasis_response = kwargs.pop('auxbasis_response')
        else:
            self.auxbasis_response = False
        if verbose is None:
            self.verbose = self.base.verbose
        else:
            self.verbose = verbose

        for f in self.base.fragments:
            if self.conv_tol_solver is not None:
                f.solvergrad = f.solver.gradient(
                        conv_tol=self.conv_tol_solver, verbose=self.verbose)
            else:
                f.solvergrad = f.solver.gradient(verbose=self.verbose)

    @staticmethod
    def get_veff(eri, dm):
        return einsum('xijkl,lk->xij', eri, dm) \
                - 0.5 * einsum('xiklj,kl->xij', eri, dm)

    def make_h1pp(self):
        base = self.base
        for f in base.fragments:
            h1pp = f.h1p * f.w1
            f.vloc_w = base.get_veff(f.h2, f.r1w)
            h1pp -= 0.5 * f.vloc_w
            f.new_vproj = f.basis.transform_h(base.new_aoveff, 'aa,ee')
            h1pp += 0.5 * (f.new_vproj * f.w1)
            f.h1pp = h1pp 

    def grad_solver(self, **kwargs):
        '''
        for each fragment
        compute r1bar and r2bar such that
        dEdmet/dc * dc/dR = r1bar * dh1/dR + r2bar * dh2/dR
        where c is the solver solution
        '''
        base = self.base
        self.make_h1pp()
        for f in base.fragments:
            f.r1bar, f.r2bar = f.solvergrad.kernel(f.h1pp, f.h2*f.w2, f.h1, f.h2, **kwargs)

            f.r1bar = (f.r1bar + f.r1bar.T) / 2
            f.r2bar = (f.r2bar + f.r2bar.transpose(1,0,2,3)) / 2
            f.r2bar = (f.r2bar + f.r2bar.transpose(0,1,3,2)) / 2

    def grad_lodm(self):
        '''
        DMET gradient due to LO DM change
        Fkl \defeq dEdmet/dlodm 
        '''
        # decrease some depth
        mf = self.base.mf
        nao = mf.mol.nao
        fragments = self.base.fragments
        get_veff = self.base.get_veff
        aofock = self.base.aofock
        aodm = self.base.aodm
        aohcore = self.base.aohcore
        new_aoveff = self.base.new_aoveff

        for f in fragments:
            neo = f.basis.Cao2eo.shape[1]

            f.vlocbar = get_veff(f.h2, f.r1bar)                                  

            # eq 139 1st
            Fae  = f.basis.transform_h(aofock, 'aa,ae')
            # eq 139 2nd
            Fae -= get_veff(f.eri_aeee, f.dm_proj)
            f.preD = einsum('jt,bj->bt', f.r1bar, Fae)

            # eq 139 3rd
            vtemp = get_veff(f.eri_aeee, f.r1bar)
            f.preD -= einsum('jt,bj->bt', f.dm_proj, vtemp)

            # eq 139 4th
            dmtemp = f.basis.transform_dm(aodm, 'aa,ae')
            dmtemp = einsum('ui,uv->vi', dmtemp, f.basis.ovlp)
            f.preD -= einsum('jt,bj->bt', f.vlocbar, dmtemp)
            
            # end of eq 139
            f.preD *= 2

            # eq 140 
            f.r2sum = (f.r2w + f.r2w.transpose(1,0,2,3)) / 2
            f.r2sum = (f.r2sum + f.r2sum.transpose(0,1,3,2)) / 2
            f.r2sum += f.r2bar
            #f.preD += 2 * einsum('tjkl,bjkl->bt', f.r2sum, f.eri_aeee)
            f.preD += 2 * f.eri_aeee.reshape(nao,-1) @ f.r2sum.reshape(neo,-1).T

            # eq 143 1st
            Fae  = 2 * f.basis.transform_h(aohcore+new_aoveff, 'aa,ae')

            # eq 143 2nd
            Fae -= get_veff(f.eri_aeee, f.r1)
            f.preD += einsum('jt,bj->bt', f.r1w, Fae)

            # eq 143 3rd
            vtemp = get_veff(f.eri_aeee, f.r1w)
            f.preD -= einsum('jt,bj->bt', f.r1, vtemp)

            # finalize eq 143
            preD = f.basis.transform_h(f.preD, 'ae,le')
            # Fkl = dEdmet / dgamma_kl
            f.Fkl = f.basis.svd.gradient(
                preD[f.env,f.nimp:], contract='pre', mask=f.basis.bath_mask)[0]

    def grad_aodm(self):
        '''
        DMET gradient due to AO DM change
        Fbar \defeq dEdmet/daodm 
        '''
        from pyscf.scf import cphf

        # decrease some depth
        mf = self.base.mf
        fragments = self.base.fragments

        # eq 138 1st term
        self.aodmbar = 0
        for f in fragments:
            self.aodmbar += f.basis.transform_dm(f.r1bar, 'ee,aa')
        self.aoveffbar = mf.get_veff(dm=self.aodmbar)
        aoFbar = self.aoveffbar.copy()

        for f in fragments:
            # eq 138 2nd 
            aoFbar -= f.basis.transform_h(f.vlocbar, 'ee,aa')

            # finalize eq 143
            aoFbar += einsum('kl,ku,lv->uv', f.Fkl, 
                    f.basis.Clo2ao[f.env], f.basis.Clo2ao[f.imp])

        aoFbar = (aoFbar + aoFbar.T) / 2

        # eq 145
        self.aoFbar = aoFbar
        self.moFbar = mf.mo_coeff.T @ aoFbar @ mf.mo_coeff
        self.moFbar *= 2 # *2 because 2-electron occ each orb for restricted

        if hasattr(self.base, 'dft'):
            mf = self.base.dft
        occ = mf.mo_occ > 0
        vir = ~occ
        vresp = mf.gen_response(singlet=None, hermi=1)
        if self.conv_tol_cphf is None or self.conv_tol_cphf > 0:
            Fvo = self.moFbar[np.ix_(vir,occ)]
            def fvind(dm_mo_VO):
                dm = mf.mo_coeff[:, vir] @ dm_mo_VO @ mf.mo_coeff.T[occ]
                dm = dm + dm.T
                v = vresp(dm)
                v_mo = mf.mo_coeff.T[vir] @ v @ mf.mo_coeff[:,occ]
                return v_mo * 2
            if self.conv_tol_cphf is not None:
                self.Z = cphf.solve(fvind, mf.mo_energy, mf.mo_occ, 2*Fvo, 
                        tol=self.conv_tol_cphf)[0]
            else:
                self.Z = cphf.solve(fvind, mf.mo_energy, mf.mo_occ, 2*Fvo)[0]
        else:
            self.Z = np.zeros((sum(vir), sum(occ)))

    def grad_ovlp(self):
        '''
        DMET gradient due to AO overlap change
        Ebar \defeq dEdmet/dovlp
        '''
        fragments = self.base.fragments
        aodm = self.base.aodm
        ovlp = self.base.basis.ovlp
        nao = self.base.mf.mol.nao

        dm_al = self.base.basis.transform_dm(aodm, 'aa,al')
        dmtemp = einsum('ui,uv->vi', dm_al, ovlp)

        self.Ebar = 0
        for f in fragments:
            # eq 152 last term
            dm_ae = f.basis.transform_dm(aodm, 'aa,ae')
            temp = einsum('ij,lj->il', f.vlocbar, dm_ae)
            self.Ebar -= 2 * einsum('il,si->ls', temp, f.basis.Cao2eo)

            # eq 152 153 154 terms involving Cao2lo grad 
            temp = einsum('ut,it->ui', f.preD, f.basis.Clo2eo)
            self.Ebar += f.basis.grad_Cao2lo_ovlp(temp)

            # eq 152 153 154 terms involving Clo2eo grad (part I.)
            temp = np.zeros((nao, nao))
            temp[:,f.env] = einsum('kl,vl->vk', f.Fkl, dmtemp[:,f.imp])
            temp[:,f.imp] = einsum('kl,vk->vl', f.Fkl, dmtemp[:,f.env])
            self.Ebar += f.basis.grad_Cao2lo_ovlp(temp)

            # eq 152 153 154 terms involving Clo2eo grad (part II.)
            temp = einsum('kl,ul->uk', f.Fkl, dm_al[:,f.imp])
            self.Ebar += einsum('uk,vk->uv', temp, f.basis.Cao2lo[:,f.env])
            temp = einsum('kl,uk->ul', f.Fkl, dm_al[:,f.env])
            self.Ebar += einsum('ul,vl->uv', temp, f.basis.Cao2lo[:,f.imp])

    def contract_ao(self, atmlst=None, mmgrad=False):
        # sanity check
        if mmgrad:
            ''' this is for qm/mm '''
            assert hasattr(self.mfgrad, 'grad_nuc_mm')
            assert hasattr(self.base.mf, 'mm_mol')
        # decrease some depth
        mf = self.base.mf
        mfgrad = self.mfgrad
        fragments = self.base.fragments
        nao = mf.mol.nao
        aodm = self.base.aodm
        new_aodm = self.base.new_aodm
        aodmbar = self.aodmbar
        moFbar = self.moFbar

        # AO derivatives
        ovlpR = mfgrad.get_ovlp()
        # NOTE f.eri_aeee will be replaced by its deriv
        if mf._eri is not None:
            self.base.log.note("ERI deriv transformation using incore")
            mf._eri = None  # to save some space
            aoeriR = -mf.mol.intor('int2e_ip1')
            for ifrag, f in enumerate(fragments):
                cput0 = (logger.process_clock(), logger.perf_counter())
                neo = f.basis.Cao2eo.shape[1]
                f.eri_aeee = np.zeros((3,nao,neo,neo,neo))
                for k in range(3):
                    f.eri_aeee[k] = \
                        f.basis.transform_eri(aoeriR[k], 'aaaa,aeee',
                                norbs=nao, contract_order=[3,0,1,2])
                self.base.log.timer(
                    f'ERI deriv transformation of fragment {ifrag}', *cput0)
            # keeping this aoeriR and using self.get_veff with explcit ERI
            # will be faster than mfgrad.get_veff
            #del aoeriR
        else: 
            self.base.log.note("ERI deriv transformation using outcore")
            from pyscf.ao2mo import general
            for ifrag, f in enumerate(fragments):
                cput0 = (logger.process_clock(), logger.perf_counter())
                C = f.basis.Cao2eo
                neo = C.shape[1]
                f.eri_aeee = -general(\
                    mf.mol, (np.eye(nao),C,C,C), intor='int2e_ip1', 
                    comp=3, aosym='s2kl', compact=False)
                f.eri_aeee = f.eri_aeee.reshape(3,nao*neo,neo,neo)
                f.eri_aeee = f.eri_aeee.reshape(3,nao,neo,neo,neo)
                self.base.log.timer(
                    f'ERI deriv transformation of fragment {ifrag}', *cput0)

        try:
            cput0 = (logger.process_clock(), logger.perf_counter())
            vhfR = self.get_veff(aoeriR, aodm)    # V_u^Rvwx aodm_wx
            self.base.log.timer('vhfR', *cput0)
            cput0 = (logger.process_clock(), logger.perf_counter())
            new_vhfR = self.get_veff(aoeriR, new_aodm)
            self.base.log.timer('new_vhfR', *cput0)
        except:
            vhfR = mfgrad.get_veff()    # V_u^Rvwx aodm_wx
            new_vhfR = mfgrad.get_veff(dm=new_aodm)
        gen_hcore = mfgrad.hcore_generator()

        # common useful
        aoslices = mf.mol.aoslice_by_atom()
        occ = mf.mo_occ > 0
        vir = ~occ
        Zuj = mf.mo_coeff[:,vir] @ self.Z
        Zuv  = Zuj @ mf.mo_coeff[:,occ].T
        Zsymm = (Zuv + Zuv.T) / 2                    
        if atmlst is None:
            atmlst = range(mf.mol.natm)
        de = np.zeros((len(atmlst),3))

        # eq 145 1st, eq 156 1st, eq 158 1st
        if not hasattr(self.base, 'dft'):
            r1temp = Zuv + aodmbar
            r1temp1 = (r1temp + r1temp.T) / 2     # vhfR only has bra, transpose for ket 
            r1temp2 = r1temp + self.base.new_aodm # hcore no need for transpose
            try:
                cput0 = (logger.process_clock(), logger.perf_counter())
                vhftempR = self.get_veff(aoeriR, r1temp1)
                self.base.log.timer('vhftempR', *cput0)
            except:
                vhftempR = mfgrad.get_veff(dm=r1temp1)

            if mmgrad:
                self.de_mm = mfgrad.contract_hcore_mm(r1temp2)
            for k, ia in enumerate(atmlst):
                p0, p1 = aoslices[ia,2:]
                h1ao = gen_hcore(ia)
                de[k] += einsum('xij,ij->x', h1ao, r1temp2)
                # *2 for adding r1_uv V_uvRwx aodm_wx
                de[k] += einsum('xij,ij->x', vhfR[:,p0:p1], r1temp1[p0:p1]) * 2
                # *2 for adding aodm_uv V_uvRwx r1_wx 
                de[k] += einsum('xij,ij->x', vhftempR[:,p0:p1], aodm[p0:p1]) * 2
                # *2 for adding aodm'_uv V_uvRwx aodm'_wx
                # 1/2 in eq 158 
                # *2 for adding aodm'_uv V_uvwRx aodm'_wx + aodm'_uv V_uvwxR aodm'_wx
                de[k] += einsum('xij,ij->x', new_vhfR[:,p0:p1], new_aodm[p0:p1]) * 2
            del r1temp
            del r1temp1
            del r1temp2
        else:
            from dmet.utils.grad.rks import get_veff_ket
            # if dft used as mean-field
            # fock in BR of eq 145 is not the same as fock in eq 156
            r1temp2 = Zuv + aodmbar + self.base.new_aodm # hcore no need for transpose
            aodmbar = (aodmbar + aodmbar.T) / 2          # in case aodmbar not symmetrized
            dft = self.base.dft
            ni = dft._numint
            # TODO reduce duplicated vj (and potentially vk) inn veffR and vhfR
            veffR = self.dftgrad.get_veff(dm=self.base.aodm) 
            vefftempR = get_veff_ket(self.dftgrad, dm=Zsymm)
            try:
                cput0 = (logger.process_clock(), logger.perf_counter())
                vhftempR = self.get_veff(aoeriR, aodmbar)
                self.base.log.timer('vhftempR', *cput0)
            except:
                vhftempR = mfgrad.get_veff(dm=aodmbar)

            if mmgrad:
                self.de_mm = mfgrad.contract_hcore_mm(r1temp2)
            for k, ia in enumerate(atmlst):
                p0, p1 = aoslices[ia,2:]
                h1ao = gen_hcore(ia)
                # hcore in eq 145, 156 and 158
                de[k] += einsum('xij,ij->x', h1ao, r1temp2)
                # *2 for ket; eq 145 cphf
                de[k] += einsum('xij,ij->x', veffR[:,p0:p1], Zsymm[p0:p1]) * 2
                de[k] += einsum('xij,ij->x', vefftempR[:,p0:p1], aodm[p0:p1]) * 2
                # *2 for ket; eq 156
                de[k] += einsum('xij,ij->x', vhfR[:,p0:p1], aodmbar[p0:p1]) * 2
                de[k] += einsum('xij,ij->x', vhftempR[:,p0:p1], aodm[p0:p1]) * 2
                # eq 158
                de[k] += einsum('xij,ij->x', new_vhfR[:,p0:p1], new_aodm[p0:p1]) * 2
            del r1temp2
            del veffR
            del vefftempR

        # eq 145 1st in cphf
        if hasattr(self.base, 'dft'):
            mf = self.base.dft
        vresp = mf.gen_response(singlet=None, hermi=1)
        Etemp  = einsum('uj,vj,j->uv', Zuj, mf.mo_coeff[:,occ], -mf.mo_energy[occ])
        Etemp -= 0.5 * aodm @ vresp(Zsymm) @ aodm
        # eq 145 2nd
        Etemp -= mf.mo_coeff[:,occ] @ moFbar[np.ix_(occ,occ)] @ mf.mo_coeff[:,occ].T
        # eq 152-154
        Etemp += self.Ebar
        Etemp += Etemp.T            # ovlpR only has bra, transpose for ket 
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia,2:]
            de[k] += einsum('xij,ij->x', ovlpR[:,p0:p1], Etemp[p0:p1])
        del Etemp

        def rVr(rx, V, ry, Cui):
            '''
            rx_ji V_u^Rjkl ry_kl C_ui
            '''
            veff = self.get_veff(V, ry)
            temp = einsum('xuj,ji->xui', veff, rx)
            return einsum('xui,ui->xu', temp, Cui)
        gtemp = 0
        for f in fragments:
            Cao2eo = f.basis.Cao2eo
            neo = Cao2eo.shape[1]
            eriR = f.eri_aeee
            # eq 156 2nd
            gtemp -= 2 * rVr(f.r1bar, eriR, f.dm_proj, Cao2eo)
            # eq 156 3rd
            gtemp -= 2 * rVr(f.dm_proj, eriR, f.r1bar, Cao2eo)
            # eq 157
            #temp = einsum('ijkl,xujkl->xui', f.r2sum, eriR)
            temp = eriR.reshape(3,nao,-1) @ f.r2sum.reshape(neo,-1).T
            gtemp += 2 * einsum('xui,ui->xu', temp, Cao2eo)
            # eq 158 2nd
            gtemp -= rVr(f.r1w, eriR, f.r1, Cao2eo)
            # eq 158 3rd
            gtemp -= rVr(f.r1, eriR, f.r1w, Cao2eo)
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia,2:]
            de[k] += einsum('xu->x', gtemp[:,p0:p1])
        del gtemp

        self.de = de 
#        self.local = locals()

    def kernel(self, atmlst=None, mmgrad=False):
        # solve impurity lambda equation and get r1bar and r2bar
        self.grad_solver()
        # get Fbar = dEdmet / dgamma 
        self.grad_lodm()
        # solve global CP-HF and 
        # get z-vector
        self.grad_aodm()
        # get Ebar = dEdmet / dS
        self.grad_ovlp()
        # contract all the intermediates with AO derivatives
        self.contract_ao(atmlst, mmgrad)

        if not mmgrad:
            return self.de + self.mfgrad.grad_nuc(atmlst=atmlst)
        else:
            grad_full = np.empty((len(self.de)+len(self.de_mm), 3))
            grad_full[:len(self.de)] = self.de + \
                self.mfgrad.grad_nuc(atmlst=atmlst)
            grad_full[len(self.de):] = self.de_mm + \
                self.mfgrad.grad_nuc_mm()
            return grad_full

    def as_scanner(self):
        from pyscf.lib import GradScanner
        class DMET_GradScanner(self.__class__, GradScanner):
            def __init__(self, gdmet_obj):
                self.gdmet_obj = gdmet_obj
                self.base = gdmet_obj.base.as_scanner()

            def __getattr__(self, item):
                '''
                we need this because some attributes are added 
                at run-time
                '''
                if item in self.__dict__:
                    return getattr(self, item)
                elif item in self.gdmet_obj.__dict__:
                    return getattr(self.gdmet_obj, item)
                else:
                    raise AttributeError()

            def __call__(self, mol, **kwargs):
                e_scanner = self.base
                E = e_scanner(mol, cache4grad=True)
                self.mol = mol

                grad = self.gdmet_obj.kernel(**kwargs)

                return E, grad

        return DMET_GradScanner(self)
