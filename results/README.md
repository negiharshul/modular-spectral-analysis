# Modular Spectral Analysis

This repository contains code for analyzing modular-time correlations 
in Gaussian quantum states (e.g. TFIM). 
It implements the **pairwise spectral measure method**:

\[
\mu_A(\lambda) = \sum_{p,q} |a_{pq}|^2 f_p (1-f_q) \, \delta(\lambda - \varepsilon_q + \varepsilon_p).
\]

### Features
- Build entanglement Hamiltonians for TFIM subsystems
- Compute operator-weighted spectral density μ(λ)
- Save results as `.npz`
- Plot μ(λ) on log–log scale with fitted power-law exponent α

### Getting Started
Clone the repo and install dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/modular-spectral-analysis.git
cd modular-spectral-analysis
pip install -r requirements.txt
