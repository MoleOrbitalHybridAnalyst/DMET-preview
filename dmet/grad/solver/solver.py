class SolverGradients:
    def __init__(self, base, conv_tol=1e-10, max_cycle=20, verbose=0):
        '''
        base: a Solver object
        '''
        self.base = base
        self.check_consistency()
        self.conv_tol = conv_tol
        self.max_cycle = max_cycle
        self.verbose = verbose

    @property
    def norb(self):
        return self.base.norb

    @property
    def nelec(self):
        return self.base.nelec

    @property
    def basis(self):
        return self.base.basis

    def kernel(self, *args, **kwargs):
        '''
        given h1'' and h2w
        such that dEDMET/dc = h'' dr1/dc + 0.5 * h2w dr2/dc

        return r1bar and r2bar 
        such that dEDMET/dc * dc/dR = r1bar * dh1/dR + 0.5 * r2bar * dh2/dR
        where h1 and h2 are really system's Hamiltionian

        i.e. r1bar = \lambda dM/dh1
             r2bar = \lambda dM/dh2
             where \lambda = dEDMET/dc (dM/dc)^{-1}
             where M(c)=0 is the solver equation
        '''
        raise Exception("virtual function called")

    def check_consistency(self):
        '''
        check if base type consistent with this gradient
        '''
        raise Exception("virtual function called")

