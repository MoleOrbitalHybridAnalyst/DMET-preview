from pexpect import ExceptionPexpect
from dmet.linalg.eigh import EIGH
from dmet.linalg.svd import SVD

def unitary_generator(theta, nao, nocc):
    '''
    where theta has vir * occ size
    return exp([[0, theta], [-theta.T, 0]])
    '''
    import numpy as np
    import scipy.linalg as la
    nvir = nao - nocc
    if theta.ndim == 2:
        assert theta.shape == (nvir, nocc)
    elif theta.ndim == 1:
        assert len(theta) == nvir * nocc
        theta = theta.reshape(nvir, nocc)
    else:
        raise Exception
    X = np.zeros((nao, nao))
    X[nocc:, :nocc] =  theta.reshape(nvir, nocc)
    X[:nocc, nocc:] = -X[nocc:, :nocc].T
    w, v = la.eig(X)
    return ((v * np.exp(w)) @ la.inv(v)).real


if __name__ == "__main__":
    import scipy.linalg as la
    import numpy as np
    norm = la.norm

    # check eigh
    A  = np.random.random((5,5))
    A  = A + A.T
    dA = np.random.random((5,5))
    dA = (dA / norm(dA)) * 1e-3
    dA = dA + dA.T
    # d eigh / dA * dA
    eigA = EIGH(A)
    _, u = eigA.kernel()
    grad = eigA.gradient(dA)
    _, u2 = la.eigh(A+dA)
    _, u1 = la.eigh(A-dA)
    for i in range(5):
        if norm(u1[:,i]-u[:,i]) > 0.5 * norm(u[:,i]):
            u1[:,i] = -u1[:,i]
        if norm(u2[:,i]-u[:,i]) > 0.5 * norm(u[:,i]):
            u2[:,i] = -u2[:,i]
    assert norm((u2 - u1) / 2 - grad) / norm(grad) < 1e-5
    # B * d eigh / dA (version 1)
    B  = np.random.random((5,5))
    grad = eigA.gradient(B, contract='pre')
    assert abs(
        ((np.einsum('ij,ij->', B, u2)-np.einsum('ij,ij->', B, u1))/2 
       - np.einsum('xkl,kl->', grad, dA)) / np.einsum('xkl,kl->', grad, dA)) < 1e-5
    # B * d eigh / dA (version 2)
    B  = np.random.random((5,5))
    grad = eigA.gradient_predecomp(np.einsum('ij,im->jm', B, u))
    assert abs(
        ((np.einsum('ij,ij->', B, u2)-np.einsum('ij,ij->', B, u1))/2 
       - np.einsum('kl,kl->', grad, dA)) / np.einsum('kl,kl->', grad, dA)) < 1e-5

    # chekc svd
    A  = np.random.random((13,5))
    dA = np.random.random((13,5))
    dA = (dA / norm(dA)) * 1e-3
    svdA = SVD(A)
    u, _, _  = svdA.kernel()
    # d svd / dA * dA
    grad = svdA.gradient(dA)
    u2, _, _ = la.svd(A+dA, full_matrices=False)
    u1, _, _ = la.svd(A-dA, full_matrices=False)
    for i in range(5):
        if norm(u1[:,i]-u[:,i]) > 0.5 * norm(u[:,i]):
            u1[:,i] = -u1[:,i]
        if norm(u2[:,i]-u[:,i]) > 0.5 * norm(u[:,i]):
            u2[:,i] = -u2[:,i]
    assert norm((u2 - u1) / 2 - grad) / norm(grad) < 1e-5
    # B * d svd / dA
    B  = np.random.random((13,5))
    grad = svdA.gradient(B, contract="pre")
    assert norm(
        (np.einsum('ij,ij->', B, u2)-np.einsum('ij,ij->', B, u1))/2 
       - np.einsum('xkl,kl->', grad, dA) ) / np.einsum('xkl,kl->', grad, dA) < 1e-5