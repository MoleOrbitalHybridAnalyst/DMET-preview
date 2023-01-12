from pyscf import gto
from pyscf.scf import RHF
from dmet.fragment import Fragment
from dmet import DMET

import numpy as np

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
mol.verbose = 0
mol.build()

mf = RHF(mol)
mf.conv_tol = 1e-12

frags = list()
for i in range(9):
    frag = Fragment()
    frag.set_imp_by_atom(mol, [i])
    frags.append(frag)
d = DMET(mf, frags, cache4grad=True, conv_tol=1e-14)
Edmet = d.kernel()
print("Nelec error before mu fitting", d.nelectron_tot() - mol.nelectron)
print("Energy before mu fitting", Edmet)

d = DMET(mf, frags, cache4grad=True, conv_tol=1e-14)
d = d.mu_fitting(nelec_tol=1e-8, conv_tol_musolver=1e-4)
Edmet_mufit = d.kernel()
assert abs(Edmet_mufit - (-225.09804865422012)) < 1e-7
print("Nelec error after mu fitting", d.nelectron_tot() - mol.nelectron)
print("Energy after mu fitting", Edmet_mufit)

g = d.nuc_grad_method()
gdmet_mufit = g.kernel()
ref_grad = \
np.array([[-3.09840783e-03, -3.90022769e-03,  5.64946414e-02],
          [ 1.86218094e-03, -6.75394408e-03, -5.15418919e-02],
          [-3.45908652e-03,  6.27011925e-03,  8.10576191e-04],
          [-6.58715176e-02, -4.66840235e-02,  4.61769516e-02],
          [ 5.41861627e-02,  5.05832616e-02, -2.58948745e-02],
          [ 1.12582701e-02, -1.77790537e-06, -3.03950063e-02],
          [ 4.89178041e-02, -7.35269968e-02,  1.01873833e-02],
          [-3.41490774e-03,  5.90885338e-02, -4.19441544e-02],
          [-4.03804982e-02,  1.49250553e-02,  3.61063748e-02]])
assert np.linalg.norm(gdmet_mufit - ref_grad) < 5e-7

d = DMET(mf, frags, cache4grad=True, conv_tol=1e-14, solver="FCI_MO")
d = d.mu_fitting(nelec_tol=1e-8, conv_tol_musolver=1e-4)
new_Edmet_mufit = d.kernel()
assert abs(new_Edmet_mufit - (-225.09804865422012)) < 1e-7

g = d.nuc_grad_method(conv_tol_solver=1e-9)
new_gdmet_mufit = g.kernel()
new_ref_grad = \
np.array([[-3.09946629e-03, -3.90006549e-03,  5.64946055e-02],
          [ 1.86254320e-03, -6.75430599e-03, -5.15423617e-02],
          [-3.45841285e-03,  6.27030627e-03,  8.11117014e-04],
          [-6.58717468e-02, -4.66838125e-02,  4.61767850e-02],
          [ 5.41862869e-02,  5.05832216e-02, -2.58948658e-02],
          [ 1.12583760e-02, -1.91157858e-06, -3.03948915e-02],
          [ 4.89174555e-02, -7.35267084e-02,  1.01873538e-02],
          [-3.41464943e-03,  5.90885544e-02, -4.19442123e-02],
          [-4.03803863e-02,  1.49247217e-02,  3.61064699e-02]])
assert np.linalg.norm(new_gdmet_mufit - new_ref_grad) < 5e-7
