"""Unit tests for Bayesian Optimization and Global Optimization classes."""

from absl.testing import absltest
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition.analytic import LogExpectedImprovement
import numpy as np
import torch

from bellows_arm_control.bayesopt import CEM
from bellows_arm_control.bayesopt import BayesOpt
from bellows_arm_control.bayesopt import BayesOptCEM
from bellows_arm_control.bayesopt import BayesOptCMAES
from bellows_arm_control.bayesopt import RandomSearch


def synthetic_quadratic(X: torch.Tensor) -> torch.Tensor:
    """Concave quadratic objective maximized at x = 0.5 with max value 0.0."""
    if len(X.shape) == 1:
        X = X.unsqueeze(0)
    val = -torch.sum((X - 0.5) ** 2, dim=-1, keepdim=True)
    return val.to(dtype=torch.float64)


class BayesOptCoreTest(absltest.TestCase):
    """Test core optimization classes on a fast synthetic objective."""

    def setUp(self):
        super().setUp()
        np.random.seed(42)
        torch.manual_seed(42)
        self.d = 4
        self.n_init = 5
        self.bounds = torch.tensor([[0.0] * self.d, [1.0] * self.d], dtype=torch.float64)
        self.X_init = torch.rand(self.n_init, self.d, dtype=torch.float64)
        self.Y_init = synthetic_quadratic(self.X_init)

    def test_bayesopt_ucb_and_lei(self):
        for acq_factory in [
            lambda gp: UpperConfidenceBound(gp, beta=0.9),
            lambda gp: LogExpectedImprovement(gp, best_f=gp.train_targets.max()),
            lambda gp: ExpectedImprovement(gp, best_f=gp.train_targets.max()),
        ]:
            bo = BayesOpt(
                self.X_init.clone(),
                self.Y_init.clone(),
                synthetic_quadratic,
                bounds=self.bounds,
                acq_func=acq_factory,
            )
            init_best = float(bo.Y[bo.best_idx].item())
            bo.step(n_restarts=2, raw_samples=8, verbose=False)
            bo.step(n_restarts=2, raw_samples=8, verbose=False)

            self.assertEqual(bo.X.shape, (self.n_init + 2, self.d))
            self.assertEqual(bo.Y.shape, (self.n_init + 2, 1))
            self.assertGreaterEqual(float(bo.Y[bo.best_idx].item()), init_best)
            # Verify that the acquisition function was updated to reference the latest GP
            self.assertIs(bo.acq_func.model, bo.gp)

    def test_bayesopt_cem(self):
        bo_cem = BayesOptCEM(
            self.X_init.clone(),
            self.Y_init.clone(),
            synthetic_quadratic,
            bounds=self.bounds,
            acq_func=lambda gp: LogExpectedImprovement(gp, best_f=gp.train_targets.max()),
            pop_size=20,
            elite_size=0.2,
            cem_steps=3,
        )
        init_best = float(bo_cem.Y[bo_cem.best_idx].item())
        bo_cem.step(n_restarts=2, raw_samples=8, verbose=False)

        self.assertEqual(bo_cem.X.shape, (self.n_init + 1, self.d))
        self.assertGreaterEqual(float(bo_cem.Y[bo_cem.best_idx].item()), init_best)

    def test_bayesopt_cmaes(self):
        bo_cma = BayesOptCMAES(
            self.X_init.clone(),
            self.Y_init.clone(),
            synthetic_quadratic,
            bounds=self.bounds,
            acq_func=lambda gp: UpperConfidenceBound(gp, beta=0.5),
            pop_size=8,
            elite_size=0.25,
            maxiter=3,
        )
        init_best = float(bo_cma.Y[bo_cma.best_idx].item())
        bo_cma.step(n_restarts=2, raw_samples=8, verbose=False)

        self.assertEqual(bo_cma.X.shape, (self.n_init + 1, self.d))
        self.assertGreaterEqual(float(bo_cma.Y[bo_cma.best_idx].item()), init_best)

    def test_cem_optimizer(self):
        cem = CEM(
            self.X_init.clone(),
            self.Y_init.clone(),
            synthetic_quadratic,
            bounds=self.bounds,
            pop_size=16,
            elite_ratio=0.25,
        )
        init_best = float(cem.Y[cem.best_idx].item())
        for _ in range(3):
            candidate = cem.step()
            self.assertEqual(candidate.shape, (self.d,))
            self.assertTrue(torch.all(candidate >= 0.0) and torch.all(candidate <= 1.0))

        self.assertEqual(cem.X.shape, (self.n_init + 3, self.d))
        self.assertGreaterEqual(float(cem.Y[cem.best_idx].item()), init_best)

    def test_random_search_optimizer(self):
        rs = RandomSearch(
            self.X_init.clone(),
            self.Y_init.clone(),
            synthetic_quadratic,
            bounds=self.bounds,
            pop_size=10,
        )
        init_best = float(rs.Y[rs.best_idx].item())
        for _ in range(2):
            candidate = rs.step()
            self.assertEqual(candidate.shape, (self.d,))

        self.assertEqual(rs.X.shape, (self.n_init + 2, self.d))
        self.assertGreaterEqual(float(rs.Y[rs.best_idx].item()), init_best)


if __name__ == "__main__":
    absltest.main()
