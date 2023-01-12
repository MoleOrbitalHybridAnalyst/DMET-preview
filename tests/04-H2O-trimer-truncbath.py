from pyscf import gto
from pyscf.scf import RHF
from dmet.fragment import Fragment
from dmet import DMET

mol = gto.Mole()
mol.atom =  f'''
   O  8.70835e-01  6.24543e+00  5.08445e+00
   H  8.51421e-01  6.33649e+00  6.05969e+00
   H  1.66206e+00  5.60635e+00  5.02479e+00
   O  2.38299e+00  8.36926e+00  4.10083e+00
   H  1.76679e+00  7.63665e+00  4.17552e+00
   H  2.40734e+00  8.80363e+00  4.99023e+00
   O  2.41917e+00  1.04168e+01  2.48601e+00
   H  2.55767e+00  9.70422e+00  3.12008e+00
   H  3.10835e+00  1.02045e+01  1.83352e+00 '''
mol.basis = 'sto-3g'
mol.build()

mf = RHF(mol)
mf.conv_tol = 1e-14

frags = list()
for i in range(3):
    frag = Fragment()
    frag.set_imp_by_atom(mol, range(3*i,3*(i+1)))
    frags.append(frag)
d = DMET(mf, frags, cache4grad=True, conv_tol=1e-12, bath_tol=1e-6)
Edmet = d.kernel()
assert abs(Edmet-(-225.06900089613316)) < 1e-7

grad = d.nuc_grad_method()
gdmet = grad.kernel()
import numpy as np
ref_grad = \
np.array([[-0.00331896, -0.00161265,  0.05291345],
         [ 0.00201528, -0.00618817, -0.04964583],
         [-0.00219674,  0.00545392,  0.00060041],
         [-0.06043677, -0.04110658,  0.04273357],
         [ 0.04896313,  0.04504788, -0.02405742],
         [ 0.01145184,  0.0009366 , -0.02923462],
         [ 0.04656923, -0.06655946,  0.00586268],
         [-0.00397982,  0.05051332, -0.03498174],
         [-0.03906719,  0.01351515,  0.03580951]])
assert np.abs(gdmet-ref_grad).sum() < 5e-7

from pyscf.cc import rccsd
mycc = rccsd.RCCSD(mf)
ecc, t1, t2 = mycc.kernel()
print("mean-field - CCSD =", -ecc)
print("DMET - CCSD =", Edmet - (mf.e_tot+ecc))
