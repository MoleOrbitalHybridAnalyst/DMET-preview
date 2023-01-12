from dmet import DMET
from dmet.exception import *

from pyscf.lib import logger

class Mufit_DMET(DMET):
    def __init__(self, dmet_obj, mu0=0, nelec_tol=1e-3, max_cycle_mufit=10):
        self.__dict__.update(dmet_obj.__dict__)
        self.mu = 0
        self.nelec_tol = nelec_tol
        self.max_cycle_mufit = max_cycle_mufit
        assert self.nelec_tol >= 0

    def make_h1(self):
        DMET.make_h1(self)
        for f in self.fragments:
            f.h1[(range(f.nimp),range(f.nimp))] -= self.mu 

    def solve_impurity(self):
        '''
        solve impurity problem with mu fitting
        '''
        self.mufit_iter = 0
        while(True):
            cput0 = (logger.process_clock(), logger.perf_counter())
            # solve impurity with current mu
            DMET.solve_impurity(self)
            self.log.timer("solving all impurities", *cput0)

            cput0 = (logger.process_clock(), logger.perf_counter())
            # get response
            dmu = self.get_dmu()
            self.log.timer("mu response of all impurites", *cput0)
            if dmu == 0:
                break

            # add dmu to h1 and mu
            self.mu += dmu
            for f in self.fragments:
                f.h1[(range(f.nimp),range(f.nimp))] -= dmu

            self.mufit_iter += 1
            if self.mufit_iter > self.max_cycle_mufit:
                raise MuNotConvergedError()

    def nuc_grad_method(self, **kwargs):
        from dmet.grad.mu_fitting.dmet import Mufit_DMETGradients
        return Mufit_DMETGradients(self, **kwargs)