__version__ = "0.1"

__doc__ = \
"""
DMET   version %s
A DMET library for ab initio system in gas phase
""" % (__version__)

from dmet import solver
from dmet import basis
from dmet import linalg
from dmet._dmet import DMET
from dmet.fragment import Fragment

from dmet import grad

from dmet.exception import *
from dmet import wrapper

def AtomicFragments(mol, atmlst=None):
    frags = list()
    if atmlst is None:
        for iatm in range(mol.natm):
            f = Fragment()
            f.set_imp_by_atom(mol, [iatm])
            frags.append(f)
    else:
        for atoms in range(atmlst):
            f = Fragment()
            f.set_imp_by_atom(mol, atoms)
            frags.append(f)
    return frags
