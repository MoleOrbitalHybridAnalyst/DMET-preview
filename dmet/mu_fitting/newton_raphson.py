from dmet.mu_fitting.dmet import Mufit_DMET

class MufitNewton_DMET(Mufit_DMET):
    def __init__(self, dmet_obj, conv_tol_musolver=None, max_cycle_musolver=None, **kwargs):
        super().__init__(dmet_obj, **kwargs)
        self.conv_tol_musolver = conv_tol_musolver
        self.max_cycle_musolver = max_cycle_musolver
        kwargs_solver = dict()
        if self.conv_tol_musolver is not None:
            kwargs_solver['conv_tol'] = self.conv_tol_musolver
        if self.max_cycle_musolver is not None:
            kwargs_solver['max_cycle'] = self.max_cycle_musolver
        for f in self.fragments:
            f.solvermu = f.solver.mu_response(f.nimp, **kwargs_solver)   

    def get_dmu(self):
        nelec = self.nelectron_tot()
        dN = nelec - self.mol.nelectron
        self.log.note(f"Mu Fitting Iter {self.mufit_iter}:")
        self.log.note(f"    mu = {self.mu}")
        self.log.note(f"    dNelec = {dN}")
        if abs(dN) < self.nelec_tol:
            self.log.note(f"    nelec_tol satisfied")
            return 0
        else:
            dNdmu = 0
            for f in self.fragments:
                dNdmu += f.solvermu.kernel(f.h1, f.h2)
            self.log.note(f"    dN / dmu = {dNdmu}")
            self.dNdmu = dNdmu
            return -dN / dNdmu