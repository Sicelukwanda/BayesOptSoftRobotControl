import logging

from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
import cma
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np
import torch

from bellows_arm_control.bayesopt.optim import CEM


def flush_logs():
    for handler in logging.getLogger().handlers:
        handler.flush()


class BayesOpt:
    """
    BO base class
    Performs BO on a given callable objective function.
    """

    def __init__(
        self,
        X,
        Y,
        obj_func,
        bounds,
        kernel=None,
        likelihood=None,
        acq_func=None,
    ):
        """
        Parameters:
        -----------
        X: torch.tensor
            Initial data points
        Y: torch.tensor
            Initial function values
        obj_func: callable
            Objective function to be optimized
        bounds: torch.tensor (shape: (2, d))
            Bounds of the param space
        """
        self.d = X.shape[1]
        self.X = X.to(dtype=torch.float64)
        self.Y = Y.reshape(-1, 1).to(dtype=torch.float64)

        # best candidate
        self.best_idx = torch.argmax(self.Y)
        self.best_x = self.X[self.best_idx, :]

        self.obj_func = obj_func

        # bounds
        self.bounds = bounds.to(dtype=self.X.dtype, device=self.X.device)

        # init gp
        if kernel is None:
            kernel = RBFKernel()
        if likelihood is None:
            likelihood = GaussianLikelihood()
        self._kernel = kernel
        self._likelihood = likelihood

        self.gp = SingleTaskGP(self.X, self.Y)
        if acq_func is None:
            self._acq_func_factory = lambda gp: UpperConfidenceBound(gp, beta=0.1)
        elif callable(acq_func) and not isinstance(acq_func, torch.nn.Module):
            self._acq_func_factory = acq_func
        else:
            self._acq_func_factory = None
        self.acq_func = self._acq_func_factory(self.gp) if self._acq_func_factory is not None else acq_func

    def _refresh_acq_func(self):
        if self._acq_func_factory is not None:
            self.acq_func = self._acq_func_factory(self.gp)

    def objective(self, x):
        """Objective function wrapper"""
        y = self.obj_func(x)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=self.Y.dtype, device=self.Y.device)
        return y.reshape(-1, 1).to(dtype=self.Y.dtype, device=self.Y.device)

    def update_gp_data(self, new_X, new_Y):
        self.X = torch.cat([self.X, new_X.to(self.X)], dim=0)
        self.Y = torch.cat([self.Y, new_Y.reshape(-1, 1).to(self.Y)], dim=0)

        # train new GP on updated dataset
        self.gp = SingleTaskGP(self.X, self.Y)

        # update best candidate
        self.best_idx = torch.argmax(self.Y)
        self.best_x = self.X[self.best_idx, :]
        self._refresh_acq_func()

    def train_gp(self, gp_optim_opts=None):
        if gp_optim_opts is None:
            gp_optim_opts = {}
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_mll(mll, **gp_optim_opts)
        self._refresh_acq_func()

    def step(self, q=1, n_restarts=10, raw_samples=10, gp_optim_opts=None, verbose=True):
        """Performs one step of BO"""
        if gp_optim_opts is None:
            gp_optim_opts = {}

        self.train_gp(gp_optim_opts=gp_optim_opts)

        # optimize acquisition function
        candidates, _ = optimize_acqf(
            self.acq_func,
            bounds=self.bounds,
            q=q,
            num_restarts=n_restarts,
            raw_samples=raw_samples,
        )

        # evaluate objective function
        new_X = candidates.detach()
        if verbose:
            logging.debug(f"step: {self.X.shape[0]}")
            logging.debug(f"new_X: {new_X}")

        new_Y = self.objective(new_X)

        # update gp data
        self.update_gp_data(new_X, new_Y)


class BayesOptCEM(BayesOpt):
    def __init__(
        self,
        X,
        Y,
        obj_func,
        bounds,
        kernel=None,
        likelihood=None,
        acq_func=None,
        pop_size=10,
        elite_size=0.2,
        cem_steps=10,
    ):
        super().__init__(X, Y, obj_func, bounds, kernel, likelihood, acq_func)
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.cem_steps = cem_steps

    def step(self, q=1, n_restarts=10, raw_samples=10, gp_optim_opts=None, verbose=True):
        """Performs one step of BO with CEM refinement on the acquisition function."""
        if gp_optim_opts is None:
            gp_optim_opts = {}

        self.train_gp(gp_optim_opts=gp_optim_opts)

        # optimize acquisition function
        candidates, _ = optimize_acqf(
            self.acq_func,
            bounds=self.bounds,
            q=q,
            num_restarts=n_restarts,
            raw_samples=raw_samples,
        )

        with torch.no_grad():
            init_X = candidates
            init_Y = self.acq_func(init_X).reshape(-1, 1)

            logging.debug(f"init_X: {init_X}")
            cem_optimizer = CEM(
                init_X,
                init_Y,
                self.acq_func,
                self.bounds,
                pop_size=self.pop_size,
                elite_ratio=self.elite_size,
            )

            for _ in range(self.cem_steps):
                cem_optimizer.step()

            new_X = cem_optimizer.best_x.unsqueeze(0)

        if verbose:
            logging.debug(f"step: {self.X.shape[0]}")
            logging.debug(f"new_X: {new_X}")

        new_Y = self.objective(new_X)

        # update gp data
        self.update_gp_data(new_X, new_Y)


class BayesOptCMAES(BayesOpt):
    def __init__(
        self,
        X,
        Y,
        obj_func,
        bounds,
        kernel=None,
        likelihood=None,
        acq_func=None,
        pop_size=10,
        elite_size=0.2,
        maxiter=20,
    ):
        super().__init__(X, Y, obj_func, bounds, kernel, likelihood, acq_func)
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.maxiter = maxiter

    def step(self, q=1, n_restarts=10, raw_samples=10, gp_optim_opts=None, verbose=True):
        """Performs one step of BO with CMA-ES refinement on the acquisition function."""
        if gp_optim_opts is None:
            gp_optim_opts = {}

        self.train_gp(gp_optim_opts=gp_optim_opts)

        # optimize acquisition function
        candidates, _ = optimize_acqf(
            self.acq_func,
            bounds=self.bounds,
            q=q,
            num_restarts=n_restarts,
            raw_samples=raw_samples,
        )

        with torch.no_grad():
            init_X = candidates
            logging.debug(f"init_X: {init_X}")

            lower_bounds = self.bounds[0].detach().cpu().numpy().tolist()
            upper_bounds = self.bounds[1].detach().cpu().numpy().tolist()
            es = cma.CMAEvolutionStrategy(
                x0=init_X.detach().cpu().numpy().flatten(),
                sigma0=0.05,
                inopts={
                    "bounds": [lower_bounds, upper_bounds],
                    "popsize": self.pop_size,
                    "maxiter": self.maxiter,
                    "verbose": -9,
                },
            )

            while not es.stop():
                xs = es.ask()
                X_cand = torch.tensor(np.array(xs), device=init_X.device, dtype=init_X.dtype)
                Y_cand = self.acq_func(X_cand.unsqueeze(-2))
                y = -1.0 * Y_cand.view(-1).double().detach().cpu().numpy()
                es.tell(xs, y)

            new_X = torch.from_numpy(es.best.x).to(init_X).unsqueeze(0)

        if verbose:
            logging.debug(f"step: {self.X.shape[0]}")
            logging.debug(f"new_X: {new_X}")

        new_Y = self.objective(new_X)

        # update gp data
        self.update_gp_data(new_X, new_Y)
