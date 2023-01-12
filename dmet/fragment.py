import numpy as np

class Fragment:
    def __init__(self, imp=None, solver=None):
        '''
        imp: a mask to indicate which LOs are imp
        '''
        if imp is not None:
            self.imp = np.array(imp, dtype=bool)
            self.env = ~self.imp
            self.env_imp = np.ix_(self.env,self.imp)
            self.nimp = sum(self.imp)

        self.solver = solver

    def set_imp_by_atom(self, mol, atmlst):
        aoslices = mol.aoslice_by_atom()[:,2:]
        imp_idx = []
        for a in atmlst:
            imp_idx += list(range(aoslices[a][0],aoslices[a][1]))
        mask = np.zeros(mol.nao, dtype=bool)
        mask[imp_idx] = True
        self.__init__(mask)

    def set_imp_by_atomshell(self, mol, iatom, ishell):
        '''
        Note atom id starts from 0
             shell id starts from 1
        '''
        aolabels = mol.ao_labels()
        imp_idx = []
        for iao, label in enumerate(aolabels):
            i, e, l = label.split()
            if int(i) == iatom and int(l[0]) == ishell:
                imp_idx.append(iao)
        mask = np.zeros(mol.nao, dtype=bool)
        mask[imp_idx] = True
        self.__init__(mask)

    def set_imp_by_atomlabel(self, mol, iatom, labels):
        aolabels = mol.ao_labels()
        imp_idx = []
        for iao, label in enumerate(aolabels):
            i, e, l = label.split()
            if int(i) == iatom and l in labels:
                imp_idx.append(iao)
        mask = np.zeros(mol.nao, dtype=bool)
        mask[imp_idx] = True
        self.__init__(mask)

    def set_w(self, neo=None):
        nimp = self.nimp
        if neo is None:
            neo = 2 * nimp
        self.w1 = np.ones((neo, neo)) * 0.5
        self.w1[:nimp,:nimp] = 1
        self.w1[nimp:,nimp:] = 0
        self.w2 = np.ones((neo,neo,neo,neo)) * 0.5
        self.w2[:nimp,:nimp,:nimp,:nimp] = 1
        self.w2[nimp:,nimp:,nimp:,nimp:] = 0
        self.w2[nimp:,:nimp,:nimp,:nimp] = 0.75
        self.w2[:nimp,nimp:,:nimp,:nimp] = 0.75
        self.w2[:nimp,:nimp,nimp:,:nimp] = 0.75
        self.w2[:nimp,:nimp,:nimp,nimp:] = 0.75
        self.w2[:nimp,nimp:,nimp:,nimp:] = 0.25
        self.w2[nimp:,:nimp,nimp:,nimp:] = 0.25
        self.w2[nimp:,nimp:,:nimp,nimp:] = 0.25
        self.w2[nimp:,nimp:,nimp:,:nimp] = 0.25

if __name__ == "__main__":
    f = Fragment([True,True,False,False,False])
