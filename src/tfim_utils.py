"""
tfim_utils.py

Reusable functions for TFIM modular spectral analysis.

Includes:
- BdG Hamiltonian construction
- Ground state correlation matrix
- Entanglement spectrum (f, eps, U)
- Operator definitions (local_n, sum_z)
- μ(λ) computation with positive pairwise weights
"""

import numpy as np
import scipy.linalg as la

# ---------------------------
# Model + correlation matrix
# ---------------------------

def build_tfim_bdg(L, J=1.0, h=0.5):
    """
    Construct Bogoliubov–de Gennes matrix for TFIM chain of length L.
    Hamiltonian: H = -J Σ σ^x σ^x - h Σ σ^z
    """
    A = np.zeros((L, L))
    B = np.zeros((L, L))
    for i in range(L):
        A[i, i] = -h
    for i in range(L - 1):
        A[i, i+1] = -J/2
        A[i+1, i] = -J/2
        B[i, i+1] = -J/2
        B[i+1, i] = J/2
    top = np.hstack((A, B))
    bottom = np.hstack((-B, -A))
    return np.vstack((top, bottom))

def groundstate_correlation_matrix(L, J=1.0, h=0.5):
    """
    Compute correlation matrix for TFIM Gaussian ground state.
    Returns L×L matrix restricted to fermions (post JW).
    """
    M = build_tfim_bdg(L, J, h)
    vals, vecs = la.eigh(M)
    neg_idx = np.where(vals < 0)[0]
    Uv = vecs[:, neg_idx]
    Ltot = L
    C = np.zeros((Ltot, Ltot))
    for alpha in range(Uv.shape[1]):
        v = Uv[Ltot:, alpha]
        C += np.outer(v.conj(), v.conj()).real
    return C

# ---------------------------
# Entanglement spectrum
# ---------------------------

def entanglement_modes(C_A):
    """
    Diagonalize block correlation matrix → occupations f, entanglement energies eps, eigenbasis U.
    """
    f, U = la.eigh(C_A)
    f = np.clip(f, 1e-14, 1 - 1e-14)
    eps = np.log((1 - f) / f)
    return f, eps, U

# ---------------------------
# Operators in subsystem
# ---------------------------

def operator_local_n(L_A, site_idx=None):
    """
    Local number operator at site_idx (default: center of subsystem).
    """
    if site_idx is None:
        site_idx = L_A // 2
    M = np.zeros((L_A, L_A))
    M[site_idx, site_idx] = 1.0
    return M

def operator_sum_z(L_A):
    """
    Uniform sum of number operators over subsystem.
    """
    return np.eye(L_A)

# ---------------------------
# μ(λ) computation
# ---------------------------

def compute_mu(f, eps, U, A_matrix, nbins=300, lambda_max=50.0):
    """
    Compute μ_A(λ) histogram from entanglement spectrum and operator matrix.

    Args:
        f: occupations (array)
        eps: entanglement energies
        U: eigenbasis matrix
        A_matrix: operator in site basis (L_A×L_A)
        nbins: number of log bins
        lambda_max: upper cutoff

    Returns dict with centers, density, lambdas, weights.
    """
    # operator in entanglement mode basis
    a = U.conj().T @ A_matrix @ U
    abs_a_sq = np.abs(a)**2
    fp = f[:, None]
    fq = f[None, :]
    W = abs_a_sq * (fp * (1 - fq))
    D = eps[None, :] - eps[:, None]

    # keep positive λ only
    mask = D > 0
    lambdas = D[mask].ravel()
    weights = W[mask].ravel()
    pos = (lambdas > 0) & (weights > 0)
    lambdas = lambdas[pos]
    weights = weights[pos]

    # histogram
    lam_min = np.min(lambdas)
    lam_max = max(lambda_max, np.max(lambdas) * 1.1)
    bins = np.logspace(np.log10(lam_min), np.log10(lam_max), nbins)
    hist, edges = np.histogram(lambdas, bins=bins, weights=weights)
    centers = np.sqrt(edges[:-1] * edges[1:])
    widths = edges[1:] - edges[:-1]
    density = hist / (widths + 1e-30)

    return dict(centers=centers, density=density,
                lambdas=lambdas, weights=weights)
