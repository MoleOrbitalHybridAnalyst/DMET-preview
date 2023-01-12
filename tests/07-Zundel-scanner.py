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
grad = d.nuc_grad_method()
g_scanner = grad.as_scanner()
Edmet, gdmet = g_scanner(mol)
assert abs(Edmet-(-150.53024123915446)) < 1e-7
ref_grad = \
np.array([[-6.98507774e-02,  2.93728835e-02, -3.52014881e-02],
         [ 7.69584197e-02, -3.08564131e-02, -1.02611584e-02],
         [ 1.66436670e-02,  6.09113650e-03,  3.88363697e-02],
         [ 3.30799663e-04, -6.91033817e-05, -2.19777771e-03],
         [-1.80522940e-02,  3.39271921e-02, -1.59278925e-02],
         [-2.77632883e-02, -3.77808348e-03,  3.22188452e-02],
         [ 2.17334733e-02, -3.46876122e-02, -7.46689820e-03]])
assert np.abs(gdmet-ref_grad).sum() < 5e-7

# change a config
mol.set_geom_(
np.array([[17.51350929, 11.81324492, 13.38277585],
          [13.75784869, 14.6623661 , 13.10396566],
          [18.76424342, 12.83775994, 12.74467203],
          [18.09008363, 12.02495094, 15.24243643],
          [14.17351285, 15.87369945, 11.86022351],
          [12.43112867, 13.72819889, 12.02529109],
          [15.65071176, 13.31491579, 12.92372358]]), unit='Bohr')

Edmet, gdmet = g_scanner(mol)
assert abs(Edmet-(-150.49938398826475)) < 1e-7
ref_grad = \
np.array([[ 0.11718471,  0.02631133, -0.05083482],
          [-0.01765802,  0.09986343, -0.02782612],
          [-0.11636829, -0.07307818,  0.07360473],
          [ 0.02190052,  0.01166648,  0.00055296],
          [-0.02379087, -0.09171873,  0.06381279],
          [ 0.00612849,  0.00845888, -0.01484436],
          [ 0.01260345,  0.01849679, -0.04446516]])
assert np.abs(gdmet-ref_grad).sum() < 5e-7
