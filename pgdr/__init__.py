"""
PGDR — Posterior-Guided Domain Randomization.

Resolves the tension between system identification (precision) and domain
randomization (robustness) by repurposing the CMA-ES covariance from
parameter identification as the DR distribution for policy training.

Pipeline:
    1. Collect reference trajectories (real robot or held-out Sim A).
    2. Run CMA-ES over parallel MJX rollouts → p* (mean) + Σ (covariance).
    3. Train policy with parameters sampled from N(p*, αΣ).
"""

__version__ = "0.1.0"
