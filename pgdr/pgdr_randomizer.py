"""
PGDR Randomizer — replaces hand-tuned DR with uncertainty-shaped sampling.

Implements all four experimental conditions from the proposal:
    C1: Uniform DR       — default Playground ranges (baseline)
    C2: Pure sys-id      — fixed at p*, no randomization
    C3: Isotropic DR     — N(p*, β²I) with tr(β²I) = tr(αΣ)
    C4: PGDR             — N(p*, αΣ) — the proposed method

The key property of PGDR (C4) is anisotropy: randomization is tight along
well-identified parameter directions and wide along uncertain ones.
C3 is the critical ablation — same center, same total variance, but isotropic.
Any performance difference between C3 and C4 isolates the value of anisotropy.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import jax
import jax.numpy as jnp
from mujoco import mjx

from pgdr.param_space import ParamSpace, inject_contact_params_to_all_feet


class DRMode(Enum):
    UNIFORM = "uniform"       # C1
    NONE = "none"             # C2 (pure sys-id)
    ISOTROPIC = "isotropic"   # C3
    PGDR = "pgdr"             # C4


class PGDRRandomizer:
    """
    Domain randomizer that samples physical parameters from one of four
    distributions, depending on the experimental condition.

    For PGDR (C4), samples from N(p*, αΣ) using Cholesky decomposition
    for efficient, numerically stable sampling.
    """

    def __init__(
        self,
        param_space: ParamSpace,
        mode: DRMode = DRMode.PGDR,
        p_star: Optional[jnp.ndarray] = None,
        Sigma: Optional[jnp.ndarray] = None,
        alpha: float = 1.0,
        uniform_ranges: Optional[dict] = None,
        foot_geom_ids: Optional[list[int]] = None,
    ):
        """
        Args:
            param_space:     Parameter space definition.
            mode:            Which experimental condition (C1-C4).
            p_star:          [d] identified mean (normalized). Required for C2-C4.
            Sigma:           [d, d] CMA-ES covariance (normalized). Required for C3-C4.
            alpha:           Covariance scale factor. Only used in C3 and C4.
            uniform_ranges:  Dict of {param_name: (low, high)} for C1.
                             If None, uses ±1 in normalized space.
            foot_geom_ids:   Foot geom IDs for contact param propagation.
        """
        self.param_space = param_space
        self.mode = mode
        self.d = param_space.d
        self.p_star = p_star
        self.Sigma = Sigma
        self.alpha = alpha
        self.foot_geom_ids = foot_geom_ids or []

        # Initialize attributes that may not be set by all modes
        self.L = None
        self.beta = None
        self._uniform_low = None
        self._uniform_high = None

        # --- Precompute sampling matrices ---

        if mode == DRMode.PGDR:
            assert p_star is not None and Sigma is not None
            assert p_star.shape == (self.d,), f"p_star shape {p_star.shape} != ({self.d},)"
            assert Sigma.shape == (self.d, self.d), f"Sigma shape {Sigma.shape} != ({self.d}, {self.d})"
            # Cholesky of αΣ for efficient N(p*, αΣ) sampling
            cov = alpha * Sigma + 1e-6 * jnp.eye(self.d)
            self.L = jnp.linalg.cholesky(cov)

        elif mode == DRMode.ISOTROPIC:
            assert p_star is not None and Sigma is not None
            # β² = tr(αΣ) / d  →  same total variance as PGDR
            total_var = jnp.trace(alpha * Sigma)
            self.beta = jnp.sqrt(total_var / self.d)

        elif mode == DRMode.NONE:
            assert p_star is not None

        elif mode == DRMode.UNIFORM:
            # Default: uniform in [-1, 1] normalized space (≈ ±scale of each param)
            if uniform_ranges is not None:
                self._uniform_low = jnp.array([
                    uniform_ranges.get(p.name, (-1.0, 1.0))[0]
                    for p in param_space.params
                ])
                self._uniform_high = jnp.array([
                    uniform_ranges.get(p.name, (-1.0, 1.0))[1]
                    for p in param_space.params
                ])
            else:
                self._uniform_low = -jnp.ones(self.d)
                self._uniform_high = jnp.ones(self.d)

    def sample(self, rng_key: jax.Array, num_envs: int) -> jnp.ndarray:
        """
        Sample num_envs parameter vectors in normalized space.

        Args:
            rng_key:   JAX PRNG key.
            num_envs:  Number of environments to sample for.

        Returns:
            [num_envs, d] array of normalized parameter vectors.
        """
        if self.mode == DRMode.PGDR:
            # N(p*, αΣ) via Cholesky: p = p* + L @ z, z ~ N(0, I)
            z = jax.random.normal(rng_key, (num_envs, self.d))
            samples = self.p_star + z @ self.L.T

        elif self.mode == DRMode.ISOTROPIC:
            # N(p*, β²I)
            z = jax.random.normal(rng_key, (num_envs, self.d))
            samples = self.p_star + self.beta * z

        elif self.mode == DRMode.NONE:
            # All environments use p* exactly
            samples = jnp.broadcast_to(self.p_star, (num_envs, self.d))

        elif self.mode == DRMode.UNIFORM:
            # U[low, high] per parameter
            samples = jax.random.uniform(
                rng_key, (num_envs, self.d),
                minval=self._uniform_low, maxval=self._uniform_high,
            )

        else:
            raise ValueError(f"Unknown DR mode: {self.mode}")

        # Clip to physically valid ranges (in normalized space)
        # We map lower/upper physical bounds back to normalized space for clipping
        norm_lower = self.param_space.to_normalized_vec(self.param_space._lowers)
        norm_upper = self.param_space.to_normalized_vec(self.param_space._uppers)

        # Handle inf bounds: only clip where finite
        finite_lower = jnp.where(jnp.isfinite(norm_lower), norm_lower, -1e6)
        finite_upper = jnp.where(jnp.isfinite(norm_upper), norm_upper, 1e6)
        samples = jnp.clip(samples, finite_lower, finite_upper)

        return samples

    def apply_to_model(
        self,
        rng_key: jax.Array,
        default_model: mjx.Model,
    ) -> mjx.Model:
        """
        Sample parameters and inject into a single mjx.Model.

        For use in non-batched environments. For batched environments,
        use sample() + inject via jax.vmap.

        Args:
            rng_key:        JAX PRNG key.
            default_model:  The base model to modify.

        Returns:
            Modified mjx.Model with sampled parameters.
        """
        p = self.sample(rng_key, 1)[0]
        model = self.param_space.inject(default_model, p)
        if self.foot_geom_ids:
            model = inject_contact_params_to_all_feet(
                model, self.foot_geom_ids, self.param_space, p
            )
        return model

    def apply_batch(
        self,
        rng_key: jax.Array,
        default_model: mjx.Model,
        num_envs: int,
    ) -> tuple[mjx.Model, jnp.ndarray]:
        """
        Sample parameters for a batch of environments and inject.

        Returns a pytree-batched mjx.Model (first axis = env index)
        suitable for use with jax.vmap(mjx.step).

        Args:
            rng_key:        JAX PRNG key.
            default_model:  The base model.
            num_envs:       Number of environments.

        Returns:
            (batched_model, sampled_params):
                batched_model has arrays with shape [num_envs, ...].
                sampled_params has shape [num_envs, d] (normalized).
        """
        params = self.sample(rng_key, num_envs)

        def inject_single(p):
            m = self.param_space.inject(default_model, p)
            if self.foot_geom_ids:
                m = inject_contact_params_to_all_feet(
                    m, self.foot_geom_ids, self.param_space, p
                )
            return m

        batched_model = jax.vmap(inject_single)(params)
        return batched_model, params

    # ---- Diagnostics ---- #

    def describe(self) -> str:
        """Human-readable summary of the randomization distribution."""
        if self.mode == DRMode.PGDR:
            eigvals = jnp.linalg.eigvalsh(self.alpha * self.Sigma)
            return (
                f"PGDR: N(p*, {self.alpha}Σ), d={self.d}\n"
                f"  Total variance (trace): {float(jnp.sum(eigvals)):.4f}\n"
                f"  Max eigenvalue: {float(jnp.max(eigvals)):.4f}\n"
                f"  Min eigenvalue: {float(jnp.min(eigvals)):.6f}\n"
                f"  Condition number: {float(jnp.max(eigvals) / jnp.max(jnp.array([jnp.min(eigvals), 1e-10]))):.1f}"
            )
        elif self.mode == DRMode.ISOTROPIC:
            return (
                f"Isotropic: N(p*, {float(self.beta)**2:.4f}·I), d={self.d}\n"
                f"  Total variance (trace): {float(self.beta**2 * self.d):.4f}\n"
                f"  Per-param std: {float(self.beta):.4f}"
            )
        elif self.mode == DRMode.NONE:
            return f"Pure sys-id: fixed at p*, d={self.d}"
        elif self.mode == DRMode.UNIFORM:
            return f"Uniform DR: U[low, high] per param, d={self.d}"
        return f"Unknown mode: {self.mode}"

    def get_per_param_std(self) -> jnp.ndarray:
        """Return the per-parameter standard deviation of the DR distribution."""
        if self.mode == DRMode.PGDR:
            cov = self.alpha * self.L @ self.L.T
            return jnp.sqrt(jnp.diag(cov))
        elif self.mode == DRMode.ISOTROPIC:
            return jnp.full(self.d, self.beta)
        elif self.mode == DRMode.NONE:
            return jnp.zeros(self.d)
        elif self.mode == DRMode.UNIFORM:
            # Std of uniform distribution
            return (self._uniform_high - self._uniform_low) / jnp.sqrt(12.0)
        return jnp.zeros(self.d)


# ---------------------------------------------------------------------------
# Factory: build randomizer from identification results
# ---------------------------------------------------------------------------

def build_randomizer(
    param_space: ParamSpace,
    mode: str,
    p_star: Optional[jnp.ndarray] = None,
    Sigma: Optional[jnp.ndarray] = None,
    alpha: float = 1.0,
    foot_geom_ids: Optional[list[int]] = None,
) -> PGDRRandomizer:
    """
    Convenience factory to build a randomizer from string mode name.

    Args:
        mode: One of "uniform", "none", "isotropic", "pgdr".
    """
    dr_mode = DRMode(mode)
    return PGDRRandomizer(
        param_space=param_space,
        mode=dr_mode,
        p_star=p_star,
        Sigma=Sigma,
        alpha=alpha,
        foot_geom_ids=foot_geom_ids,
    )
