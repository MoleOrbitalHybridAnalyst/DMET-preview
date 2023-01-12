from pyscf import gto
from pyscf.scf import RHF
from dmet.fragment import Fragment
from dmet import DMET
from time import time
from copy import copy

import numpy as np
mol = gto.Mole()
mol.atom =  '''
   O  8.70835e-01  6.24543e+00  5.08445e+00
   H  8.51421e-01  6.33649e+00  6.05969e+00
   H  1.66206e+00  5.60635e+00  5.02479e+00
   O  2.38299e+00  8.36926e+00  4.10083e+00
   H  1.76679e+00  7.63665e+00  4.17552e+00
   H  2.40734e+00  8.80363e+00  4.99023e+00 '''
mol.basis = 'sto-3g'
mol.verbose = 0
mol.build()

mf = RHF(mol)
mf.conv_tol = 1e-14

frags = list()
frags2 = list()
# slice by shell
ranges = [range(2),range(2,5),[5],[6],range(7,9),range(9,12),[12],[13]]
for r in ranges:
    mask = np.zeros(mol.nao, dtype=bool)
    mask[r] = True
    frag = Fragment(mask)
    frags.append(frag)
    frag = Fragment(mask)
    frags2.append(frag)
d =  DMET(mf, frags,  cache4grad=True, conv_tol=1e-14)
d2 = DMET(mf, frags2, cache4grad=True, conv_tol=1e-14, solver="FCI_MO")
t = time() 
Edmet = d.kernel()
print("timing for Edmet (FCI AO)", time()-t)
t = time() 
Edmet2 = d2.kernel()
print("timing for Edmet (FCI MO)", time()-t)
assert abs(Edmet-(-150.04258874655244)) < 1e-7
assert abs(Edmet2-(-150.04258874655244)) < 1e-7

grad = d.nuc_grad_method()
t = time()
gdmet = grad.kernel()
print("timing for DMET gradient (FCI AO)", time()-t)
grad2 = d2.nuc_grad_method()
t = time()
gdmet2 = grad2.kernel()
print("timing for DMET gradient (FCI MO)", time()-t)
ref_grad = \
np.array([[ 0.00060422, -0.00046076,  0.05429722],
       [-0.00068642, -0.00459803, -0.04895548],
       [-0.00198123,  0.00414253, -0.00195498],
       [-0.06316797, -0.04844727,  0.06627756],
       [ 0.05018439,  0.04776844, -0.03428971],
       [ 0.015047  ,  0.00159509, -0.03537461]])
assert np.abs(gdmet-ref_grad).sum() < 1e-7
assert np.abs(gdmet2-ref_grad).sum() < 1e-7

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
