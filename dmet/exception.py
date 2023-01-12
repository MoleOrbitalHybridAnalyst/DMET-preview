class SolverNotConvergedError(Exception):
    pass

class SCFNotConvergedError(Exception):
    pass

class FittingNotConvergedError(Exception):
    pass

class MuNotConvergedError(FittingNotConvergedError):
    pass

class DMETSCFNotConvergedError(Exception):
    pass