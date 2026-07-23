"""
FiniteGMM: a fixed-component Gaussian Mixture Model for the DPMM-vs-GMM ablation.

This is the paper's DPMM with the Dirichlet-Process prior switched OFF. It is written
as a subclass of DPMM so their dpmm.py stays completely untouched.

A finite GMM differs from the DPMM in exactly two places, and this class overrides
exactly those:

  1. Mixing weights (calculate_pi / calculate_log_pi): a GMM uses the normalised
     component responsibilities directly, instead of the DPMM's stick-breaking
     construction. => no Dirichlet-Process pressure to prune components.

  2. Training update (m_step): a GMM keeps the number of components FIXED at K, so it
     runs the same EM mean/covariance update but WITHOUT updating the stick-breaking
     ratios v or adapting the concentration alpha.

Everything else -- the Gaussian components, the EM responsibilities, checkpointing, and
all seven anomaly-score maps -- is inherited unchanged from DPMM, so the evaluation
pipeline is identical and the DPMM-vs-GMM comparison is apples-to-apples.

This is to compare dpmm and gmm using four different K for gmm.

"""

import torch

from src.DirichletProcessMixture.dpmm import DPMM


class FiniteGMM(DPMM):
    """DPMM with the Dirichlet-Process prior disabled -> a plain K-component GMM."""

    def calculate_pi(self) -> torch.Tensor:
        # Mixing weight of each component = share of the data it explains (normalised
        # responsibilities). No stick-breaking, so nothing pushes components to zero.
        total = self.resp_stat.sum()
        if total <= self.eps:                          # before the first M-step: uniform weights
            return torch.full((self.K,), 1.0 / self.K, device=self.device)
        return self.resp_stat / total

    def calculate_log_pi(self) -> torch.Tensor:
        # log of the above; an empty (zero-weight) component maps to -inf, which cleanly
        # drops it from the weighted log-prob / logsumexp used for scoring.
        return torch.log(self.calculate_pi())

    def m_step(self, data: torch.Tensor, resp: torch.Tensor):
        # Identical EM update to DPMM.m_step for the sufficient statistics, means and
        # covariances -- but WITHOUT the stick-breaking (v) and concentration (alpha)
        # updates at the end, because a finite GMM keeps K fixed. That omission is the
        # entire difference between this model and the DPMM.
        n_samples = data.shape[0]

        # Updating sufficient statistics.
        step_size = self.get_step_size()
        self.resp_stat *= (1 - step_size)
        self.mean_stat *= (1 - step_size)
        self.cov_stat *= (1 - step_size)

        data_cov = (data * data.transpose(-2, -1))[:, None, :, :]      # Nx1xDxD
        self.resp_stat += step_size * resp.mean(0)
        self.mean_stat += step_size * (data * resp[:, :, None]).mean(0)
        self.cov_stat  += step_size * torch.einsum("...ndf,nk->kdf", data_cov.transpose(0, 1), resp) / n_samples

        # Updating cluster means.
        self.mean = self.mean_stat / (self.resp_stat[:, None] + self.eps)

        # Updating cluster covariances.
        if self.cov_type == "full":
            self.cov = self.estimate_covariance_full()
        elif self.cov_type == "diag":
            self.cov = self.estimate_covariance_diagonal()
        elif self.cov_type == "spherical":
            self.cov = self.estimate_covariance_spherical()
        else:
            raise NotImplementedError()
        self.compute_covariance_cholesky()
        # NOTE: no v / alpha update here -- that is the whole point of the ablation.
