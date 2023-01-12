from dmet.mu_fitting.dmet import Mufit_DMET

class MufitSecant_DMET(Mufit_DMET):
    def __init__(self, dmet_obj, mu0=0, nelec_tol=0.001, max_cycle_mufit=10, stepsize=0.02):
        super().__init__(dmet_obj, mu0, nelec_tol, max_cycle_mufit)
        self.stepsize = stepsize
        assert self.stepsize != 0
        self.last_mu = None
        self.last_nelec = None
        for f in self.fragments:
            try:
                # we need mu_response.dN_dc for gradients
                f.solvermu = f.solver.mu_response(f.nimp)
            except:
                # if failed, ignore because we may just want energy
                pass

    def get_dmu(self):
        nelec = self.nelectron_tot()
        dN = nelec - self.mol.nelectron
        self.log.note(f"Mu Fitting Iter {self.mufit_iter}:")
        self.log.note(f"    dNelec = {dN}")
        if abs(dN) < self.nelec_tol:
            self.log.note(f"    nelec_tol satisfied")
            return 0
        else:
            if self.last_mu is None:
                if dN < 0:
                    # mu should increase
                    dmu = abs(self.stepsize)
                else:
                    dmu = -abs(self.stepsize)
            else:
                dNdmu = (nelec - self.last_nelec) / (self.mu - self.last_mu)
                dmu = -dN / dNdmu
            self.last_mu = self.mu
            self.last_nelec = nelec
            return dmu