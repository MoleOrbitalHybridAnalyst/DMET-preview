from dmet.solver.solver import Solver
from dmet.solver import fci, rhf, fci_mo, rccsd, rcisd, rccsd_t

solver_dict = {"FCI": fci.FCI, "RHF": rhf.RHF, 
        "FCIMO": fci_mo.FCIMO, "FCI_MO": fci_mo.FCIMO,
        "RCCSD": rccsd.RCCSD, "RCCSD_APPROX": rccsd.RCCSD_APPROX,
        "APPROX_RCCSD": rccsd.RCCSD_APPROX, "RCISD": rcisd.RCISD,
        "RCCSD_T": rccsd_t.RCCSD_T, "RCCSD(T)": rccsd_t.RCCSD_T}
