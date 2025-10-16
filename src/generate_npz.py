import numpy as np
import scipy.linalg as la

def build_tfim_bdg(L, J=1.0, h=0.5):
    """Construct Bogoliubov–de Gennes matrix for TFIM chain length L."""
    A = np.zeros((L, L))
    B = np.zeros((L, L))
    for i in range(L):
        A[i, i] = -h
    for i in range(L-1):
        A[i, i+1] = -J/2
        A[i+1, i] = -J/2
        B[i, i+1] = -J/2
        B[i+1, i] = J/2
    top = np.hstack((A, B))
    bottom = np.hstack((-B, -A))
    return np.vstack((top, bottom))

def groundstate_correlation_matrix(L, J=1.0, h=0.5):
    """Compute correlation matrix of TFIM Gaussian ground state."""
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

def entanglement_modes(C_A):
    """Diagonalize correlation matrix block -> occupations and entanglement energies."""
    f, U = la.eigh(C_A)
    f = np.clip(f, 1e-14, 1 - 1e-14)
    eps = np.log((1 - f) / f)
    return f, eps, U

def operator_matrix(L_A, site_idx=None):
    """Local number operator at site_idx (default: center)."""
    if site_idx is None:
        site_idx = L_A // 2
    M = np.zeros((L_A, L_A))
    M[site_idx, site_idx] = 1.0
    return M

def compute_mu(f, eps, U, A_site_matrix, nbins=300, lambda_max=50.0):
    """Compute histogram approximation of μ(λ)."""
    a = U.conj().T @ A_site_matrix @ U
    abs_a_sq = np.abs(a)**2
    fp = f[:, None]; fq = f[None, :]
    W = abs_a_sq * (fp * (1 - fq))
    D = eps[None, :] - eps[:, None]
    mask = D > 0
    lambdas = D[mask].ravel()
    weights = W[mask].ravel()
    pos = (lambdas > 0) & (weights > 0)
    lambdas = lambdas[pos]; weights = weights[pos]
    lam_min = np.min(lambdas)
    lam_max = max(lambda_max, np.max(lambdas)*1.1)
    bins = np.logspace(np.log10(lam_min), np.log10(lam_max), nbins)
    hist, edges = np.histogram(lambdas, bins=bins, weights=weights)
    centers = np.sqrt(edges[:-1] * edges[1:])
    widths = edges[1:] - edges[:-1]
    density = hist / (widths + 1e-30)
    return dict(centers=centers, density=density,
                lambdas=lambdas, weights=weights)

if __name__ == "__main__":
    L_A = 80
    L = 2 * L_A
    print("Generating data for L_A=80, local operator...")

    # correlation matrix restricted to subsystem
    Cfull = groundstate_correlation_matrix(L, J=1.0, h=0.5)
    C_A = Cfull[:L_A, :L_A]

    # entanglement modes
    f, eps, U = entanglement_modes(C_A)

    # operator matrix (local n)
    A_site = operator_matrix(L_A)

    # spectral density
    mu = compute_mu(f, eps, U, A_site, nbins=300, lambda_max=50.0)

    # save
    out_path = "pairwise_corrected_LA80_local_n.npz"
    np.savez_compressed(out_path, **mu)
    print(f"Saved: {out_path}")
