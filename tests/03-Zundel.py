from pyscf import gto
from pyscf.scf import RHF
from dmet.fragment import Fragment
from dmet import DMET
from time import time

import numpy as np
mol = gto.Mole()
mol.atom =  '''
 O   0.54044984       -0.97485007       -0.21657970
 O   2.92484146       -0.83925223        0.15239669
 H   0.18385954       -1.25804198       -1.07875142
 H   1.74328720       -0.90927800       -0.10039502
 H   3.29706729       -1.59324218        0.64725255
 H   3.53032937       -0.60813812       -0.57607828
 H   0.04233158       -0.19485993        0.09228101 '''
mol.basis = 'sto-3g'
mol.charge = 1
mol.build()

mf = RHF(mol)
mf.conv_tol = 1e-14

frags = list()
# slice by atom
for i in range(7):
    frag = Fragment()
    frag.set_imp_by_atom(mol, [i])
    frags.append(frag)
assert sum([f.nimp for f in frags]) == mol.nao
d = DMET(mf, frags, cache4grad=True, conv_tol=1e-14)
t = time() 
Edmet = d.kernel()
print("timing for Edmet", time()-t)
assert abs(Edmet-(-150.53024123915446)) < 1e-7

grad = d.nuc_grad_method()
t = time()
gdmet = grad.kernel()
print("timing for DMET gradient", time()-t)
ref_grad = \
np.array([[-6.98507774e-02,  2.93728835e-02, -3.52014881e-02],
         [ 7.69584197e-02, -3.08564131e-02, -1.02611584e-02],
         [ 1.66436670e-02,  6.09113650e-03,  3.88363697e-02],
         [ 3.30799663e-04, -6.91033817e-05, -2.19777771e-03],
         [-1.80522940e-02,  3.39271921e-02, -1.59278925e-02],
         [-2.77632883e-02, -3.77808348e-03,  3.22188452e-02],
         [ 2.17334733e-02, -3.46876122e-02, -7.46689820e-03]])
assert np.abs(gdmet-ref_grad).sum() < 1e-7

from sys import argv
if len(argv) == 2:
    from pyscf.fci import direct_spin1
    from pyscf.ao2mo import full
    t = time()
    fci = direct_spin1.FCI()
    h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    h2e = full(mf._eri, mf.mo_coeff)
    E_fci = fci.kernel(h1e, h2e, mol.nao, mol.nelectron)[0]
    print("timing for FCI", time()-t)
    print("mean-field - FCI =", mf.energy_elec()[0] - E_fci)
    print("DMET - FCI =", d.Edmet_tot - E_fci)
