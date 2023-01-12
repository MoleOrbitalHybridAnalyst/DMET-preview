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
mol.verbose = 0
mol.build()

mf = RHF(mol)
mf.conv_tol = 1e-14

frags = list()
for i in range(3):
    frag = Fragment()
    frag.set_imp_by_atom(mol, range(3*i,3*(i+1)))
    frags.append(frag)
d = DMET(mf, frags, cache4grad=True, solver="RCCSD", conv_tol=1e-12, bath_tol=1e-6)
Edmet = d.kernel()
assert abs(Edmet--225.06832736427867) < 1e-7

frags = list()
for i in range(3):
    frag = Fragment()
    frag.set_imp_by_atom(mol, range(3*i,3*(i+1)))
    frags.append(frag)
d = DMET(mf, frags, cache4grad=True, solver="APPROX_RCCSD", conv_tol=1e-12, bath_tol=1e-6)
Edmet2 = d.kernel()
assert abs(Edmet2--225.06850488804577) < 1e-7

from pyscf.cc import ccsd
mcc = ccsd.RCCSD(mf)
E_cc, t1, t2 = mcc.kernel()
print("mean-field - CCSD =", -E_cc)
print("DMET1 - CCSD =", Edmet - (mf.e_tot+E_cc))
print("DMET2 - CCSD =", Edmet2 - (mf.e_tot+E_cc))
