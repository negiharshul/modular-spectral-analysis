import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import sys

def load_mu_data(npz_path):
    """Load spectral density data from .npz file."""
    data = np.load(npz_path)
    centers = data["centers"]
    density = data["density"]
    return centers, density

def fit_power_law(centers, density, lam_max=1.0, min_fit_points=5):
    """
    Fit log(mu) vs log(lambda) in [lambda_min, lam_max].
    Returns (alpha, alpha_err, fit_x, fit_y) or None if fit fails.
    """
    mask = (centers > 0) & (density > 0) & (centers <= lam_max)
    if mask.sum() < min_fit_points:
        return None
    x = centers[mask]
    y = density[mask]

    def linlog(xx, m, c):
        return m * np.log(xx) + c

    try:
        popt, pcov = curve_fit(linlog, x, np.log(y), p0=[-0.5, np.log(y.mean())])
        m, c = popt
        alpha = m + 1
        alpha_err = np.sqrt(np.diag(pcov))[0] if pcov is not None else None
        fit_y = np.exp(linlog(x, m, c))
        return alpha, alpha_err, x, fit_y
    except Exception as e:
        print(f"Fit failed: {e}")
        return None

def plot_mu(npz_path, out_dir="results", lam_max=1.0):
    centers, density = load_mu_data(npz_path)
    base = os.path.splitext(os.path.basename(npz_path))[0]
    out_png = os.path.join(out_dir, base + ".png")

    # raw data (positive only)
    mask = (centers > 0) & (density > 0)
    centers_plot = centers[mask]
    density_plot = density[mask]

    # do fit
    fit_result = fit_power_law(centers_plot, density_plot, lam_max=lam_max)

    plt.figure(figsize=(7,5))
    plt.loglog(centers_plot, density_plot, "o", markersize=4, label="μ(λ)")

    if fit_result is not None:
        alpha, alpha_err, x_fit, y_fit = fit_result
        plt.loglog(x_fit, y_fit, "--", label=f"fit: α={alpha:.2f} ± {alpha_err:.2f}")

    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$\mu_A(\lambda)$")
    plt.title(base)
    plt.legend()
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    print(f"Saved plot → {out_png}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/plot_mu.py results/your_data_file.npz [lam_max]")
        sys.exit(1)

    npz_file = sys.argv[1]
    lam_max = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    plot_mu(npz_file, lam_max=lam_max)
