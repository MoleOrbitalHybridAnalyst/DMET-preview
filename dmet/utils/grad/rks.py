from pyscf import lib
from pyscf.lib import logger

from dmet.utils.dft.numint import nr_rksgrad_fxc

def get_veff_ket(ks_grad, mol=None, dm=None):
    '''
    part of Fock skeleton derivative 
    see CPHF/CPKS eq 24 & 25 blue font
    '''
    if mol is None: mol = ks_grad.mol
    if dm is None: dm = ks_grad.base.make_rdm1()
    t0 = (logger.process_clock(), logger.perf_counter())

    mf = ks_grad.base
    ni = mf._numint
    if ks_grad.grids is not None:
        grids = ks_grad.grids
    else:
        grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)

    if mf.nlc != '':
        raise NotImplementedError
    #enabling range-separated hybrids
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, ks_grad.max_memory*.9-mem_now)
    if ks_grad.grid_response:
        raise NotImplementedError("DFT Grid Response")
    else:
        vxc = -nr_rksgrad_fxc(ni, mol, grids, mf.xc, mf.make_rdm1(),
            dm, max_memory=max_memory, verbose=ks_grad.verbose)
    t0 = logger.timer(ks_grad, 'vxc', *t0)

    if abs(hyb) < 1e-10 and abs(alpha) < 1e-10:
        vj = ks_grad.get_j(mol, dm)
        vxc += vj
    else:
        vj, vk = ks_grad.get_jk(mol, dm)
        vk *= hyb
        if abs(omega) > 1e-10:  # For range separated Coulomb operator
            with mol.with_range_coulomb(omega):
                vk += ks_grad.get_k(mol, dm) * (alpha - hyb)
        vxc += vj - vk * .5

    return vxc