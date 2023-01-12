from dmet.grad.dmet import DMETGradients

class Mufit_DMETGradients(DMETGradients):
    def __init__(self, base, conv_tol_solver=1e-9, conv_tol_cphf=None, verbose=None, **kwargs):
        super().__init__(base, conv_tol_solver, conv_tol_cphf, verbose, **kwargs)
        from dmet.mu_fitting.dmet import Mufit_DMET
        assert isinstance(base, Mufit_DMET)

    def grad_solver(self, **kwargs):
        import numpy as np
        import scipy.sparse.linalg as sla
        base = self.base

        # h1pp will be used to compute dE_dc
        DMETGradients.make_h1pp(self)

        # since impurity solutions are connected via a common mu
        # it is impossible to solve impurity energy response independently
        # (I think this is true?)
        p0 = list()           # the starting position of i-th impurity lambda
        p1 = list()           # the ending position of i-th impurity lambda
        i0 = 0
        for f in base.fragments:
            i1 = i0 + f.solvergrad.lambda_size()
            p0.append(i0)
            p1.append(i1)
            i0 = i1
        tot_size = i1 + 1
        
        full_dE_dc = np.zeros(tot_size)  # the last one is dE / dmu = 0
        full_dN_dc = np.zeros(tot_size)  # the last one is also zero
        for i0, i1, f in zip(p0, p1, base.fragments):
            full_dE_dc[i0:i1] = f.solvergrad.dE_dc(f.h1pp, f.h2*f.w2).ravel()
            full_dN_dc[i0:i1] = f.solvermu.dN_dc().ravel()
        
        full_ldMdc = list()
        full_ldMdmu = list()
        for f in base.fragments:
            full_ldMdc.append(f.solvergrad.dot_lambda(f.h1, f.h2))
            full_ldMdmu.append(f.solvermu.dot_lambda())

        def M(lmbda):
            lM = np.zeros(tot_size)
            for i0, i1, lMc in zip(p0, p1, full_ldMdc):
                lM[i0:i1] = lMc(lmbda[i0:i1]).ravel() + lmbda[-1] * full_dN_dc[i0:i1]
            for i0, i1, lMm in zip(p0, p1, full_ldMdmu):
                lM[-1] += lMm(lmbda[i0:i1])
            lM[-1] += full_dN_dc[-1] * lmbda[-1]
            return lM

        M = sla.LinearOperator((tot_size,tot_size), M)

        kwargs_gmres = dict()
        if 'lambda0' in kwargs:
            kwargs_gmres['x0'] = kwargs.pop('lambda0').ravel()
        if self.conv_tol_solver is not None:
            kwargs_gmres['tol'] = self.conv_tol_solver
            kwargs_gmres['atol'] = self.conv_tol_solver * np.linalg.norm(full_dE_dc)
        self.full_lambda, stat = \
            sla.gmres(M, full_dE_dc, **kwargs_gmres)
        if stat != 0:
            raise Exception("Solver response not converged")

        for i0, i1, f in zip(p0, p1, base.fragments):
            f.r1bar, f.r2bar = f.solvergrad.make_rdm12(self.full_lambda[i0:i1])
            f.r1bar = (f.r1bar + f.r1bar.T) / 2
            f.r2bar = (f.r2bar + f.r2bar.transpose(1,0,2,3)) / 2
            f.r2bar = (f.r2bar + f.r2bar.transpose(0,1,3,2)) / 2

        self.p0 = p0
        self.p1 = p1
