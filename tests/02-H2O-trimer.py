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
for i in range(9):
    frag = Fragment()
    frag.set_imp_by_atom(mol, [i])
    frags.append(frag)
d = DMET(mf, frags, cache4grad=True, conv_tol=1e-14)
Edmet = d.kernel()
assert abs(Edmet-(-225.0963451190771)) < 1e-7

grad = d.nuc_grad_method()
gdmet = grad.kernel()
import numpy as np
ref_grad = \
np.array([[-0.00381584, -0.00381311,  0.05596588],
       [ 0.00200471, -0.00658842, -0.05106448],
       [-0.00294977,  0.00607471,  0.00078414],
       [-0.06519883, -0.04643658,  0.04567362],
       [ 0.05367285,  0.04997226, -0.02580984],
       [ 0.01126567,  0.0002809 , -0.02995144],
       [ 0.04849031, -0.07259943,  0.00980584],
       [-0.00342541,  0.05832948, -0.04122229],
       [-0.04004369,  0.01478019,  0.03581856]])
assert np.abs(gdmet-ref_grad).sum() < 5e-7
