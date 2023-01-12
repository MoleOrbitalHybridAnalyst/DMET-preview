class SolverMuResponse:
    def __init__(self, base, nimp, conv_tol=1e-4, max_cycle=20, verbose=0):
        '''
        base: a Solver object
        '''
        self.nimp = nimp
        self.base = base
        self.check_consistency()
        self.conv_tol = conv_tol
        self.max_cycle = max_cycle
        self.verbose = verbose
    
    @property
    def nelec(self):
        return self.base.nelec

    @property
    def norb(self):
        return self.base.norb

    @property
    def basis(self):
        return self.base.basis

    def dN_dc(self):
        '''
        d Nelec / d c
        where Nelec is the number of electron in this impurity
              c is the impurity solution
        '''
        raise Exception("virtual function called")

    def dot_lambda(self, lmbda):
        '''
        dot d M / d mu with lmbda
        where M = 0 is the impurity equation
              lmbda will be the multiplier of impurity solution
        '''
        raise Exception("virtual function called")

    def kernel(self, *args, **kwargs):
        '''
        given h1 (with mu) and h2
        return dN / dmu

        where h1 and h2 are the embedding Hamiltionian
              N is the number of electron computed from r1_imp_imp trace
        '''
        raise Exception("virtual function called")

    def check_consistency(self):
        '''
        check if base type consistent with this gradient
        '''
        raise Exception("virtual function called")
