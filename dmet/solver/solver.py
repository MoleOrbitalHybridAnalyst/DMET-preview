class Solver:
    def __init__(self, nelec=None, verbose=None, 
            conv_tol=None, max_cycle=None):
        self.nelec = nelec
        self.verbose = verbose
        self.conv_tol = conv_tol
        self.max_cycle = max_cycle
        self._solver = None

    @property
    def solution(self):
        raise Exception("virtual function called")

    def kernel(self, h1e, eri):
        raise Exception("virtual function called")

    def gradient(self, **kwargs):
        raise Exception("virtual function called")

    def mu_response(self, *args, **kwargs):
        raise Exception("virtual function called")

    def __getattr__(self, item):
        if item in self.__dict__:
            return getattr(self, item)
        elif item in self._solver.__dict__:
            return getattr(self._solver, item)
        else:
            raise AttributeError()
