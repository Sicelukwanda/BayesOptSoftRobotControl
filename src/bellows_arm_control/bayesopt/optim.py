import logging

import numpy as np
from scipy.stats import truncnorm
import torch
import torch.distributions as dist


class CEM:
    """
    CEM base class
    Performs optimization on a given callable objective function using the Cross-Entropy Method.
    """

    def __init__(
        self,
        X,
        Y,
        obj_func,
        bounds,
        pop_size=1000,
        elite_ratio=0.2,
        init_variance=None,
    ):
        """
        Parameters:
        -----------
        X: torch.tensor (shape: (N, d))
            Initial data points, used to fit starting distribution
        Y: torch.tensor (shape: (N, 1) or (N,))
            Initial function values
        obj_func: callable
            Objective function to be optimized.
        bounds: torch.tensor (shape: (2, d))
            Bounds of the parameter space.
        pop_size: int
            Population size per iteration.
        elite_ratio: float
            Ratio of samples considered as elite (or integer count if >= 1).
        init_variance: float or torch.Tensor, optional
            Initial variance of the search distribution.
        """
        self.d = X.shape[1]
        self.X = X.to(dtype=torch.float64)
        self.Y = Y.reshape(-1, 1).to(dtype=torch.float64)

        # best candidate
        self.best_idx = torch.argmax(self.Y)
        self.best_x = self.X[self.best_idx, :]

        # init distribution params
        if init_variance is not None:
            if isinstance(init_variance, torch.Tensor):
                assert init_variance.shape == (self.d,)
                self.variance = init_variance.to(dtype=torch.float64)
            else:
                self.variance = torch.ones_like(self.X[0]) * float(init_variance)
        else:
            self.variance = torch.ones_like(self.X[0]) * float((bounds.max() - bounds.min()) ** 2)
        self.mean = torch.mean(self.X, dim=0)

        self.obj_func = obj_func
        self.bounds = bounds.to(dtype=torch.float64)
        self.pop_size = int(pop_size)
        if elite_ratio >= 1 and isinstance(elite_ratio, int):
            self.elite_size = max(1, min(self.pop_size, elite_ratio))
        else:
            self.elite_size = max(1, int(self.pop_size * float(elite_ratio)))

    def update_cem_data(self, new_X, new_Y):
        """Update data, keep track of best so far"""
        self.X = torch.cat([self.X, new_X.to(self.X)], dim=0)
        self.Y = torch.cat([self.Y, new_Y.reshape(-1, 1).to(self.Y)], dim=0)

        # update best candidate
        self.best_idx = torch.argmax(self.Y)
        self.best_x = self.X[self.best_idx, :]

    def sample_torch(self, num_samples):
        """
        Generate samples using a normal distribution and transform them to be within specified bounds.
        """
        base_distribution = dist.MultivariateNormal(
            loc=self.mean,
            covariance_matrix=torch.diag(self.variance + 1e-6),
        )

        lower_bounds, upper_bounds = self.bounds[0], self.bounds[1]
        scale = upper_bounds - lower_bounds
        shift = lower_bounds

        transformations = [
            dist.transforms.SigmoidTransform(),
            dist.transforms.AffineTransform(loc=shift, scale=scale),
        ]

        transformed_distribution = dist.TransformedDistribution(base_distribution, transformations)
        transformed_samples = transformed_distribution.sample((num_samples,))

        return transformed_samples

    def sample_np(self, num_samples):
        samples = np.random.multivariate_normal(
            mean=self.mean.detach().cpu().numpy().flatten(),
            cov=torch.diag(self.variance + 1e-8).detach().cpu().numpy(),
            size=num_samples,
        )
        clipped_samples = np.clip(
            samples,
            self.bounds[0].detach().cpu().numpy(),
            self.bounds[1].detach().cpu().numpy(),
        )
        return torch.tensor(clipped_samples, dtype=torch.float64, device=self.X.device)

    def sample_scipy(self, num_samples):
        # Based on: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
        a = self.bounds[0, :].detach().cpu().numpy()
        b = self.bounds[1, :].detach().cpu().numpy()
        loc = self.mean.detach().cpu().numpy()
        scale = torch.sqrt(torch.clamp(self.variance, min=1e-8)).detach().cpu().numpy()

        a_transformed, b_transformed = (a - loc) / scale, (b - loc) / scale
        rv = truncnorm(a_transformed, b_transformed, loc=loc, scale=scale)
        return torch.tensor(rv.rvs(size=(num_samples, self.d)), dtype=torch.float64, device=self.X.device)

    def objective(self, x):
        """Objective function wrapper"""
        y = self.obj_func(x)
        if isinstance(y, torch.Tensor):
            y = y.detach()
        else:
            y = torch.tensor(y, dtype=torch.float64)
        return y.reshape(-1).squeeze(0)

    def step(self, maximize=True, verbose=False, sampler=None):
        """Performs one step of CEM"""
        if sampler is None:
            sampler = self.sample_scipy

        samples = sampler(self.pop_size)

        scores = torch.stack([self.objective(sample.unsqueeze(0)) for sample in samples]).reshape(-1)

        if maximize:
            best_sample_idx = torch.argmax(scores)
        else:
            best_sample_idx = torch.argmin(scores)

        self.update_cem_data(
            samples[best_sample_idx].unsqueeze(0),
            scores[best_sample_idx].reshape(-1, 1),
        )

        # Select elite samples
        elite_idxs = scores.topk(self.elite_size, largest=maximize).indices
        elite_samples = samples[elite_idxs]

        # Update distribution parameters
        self.mean = elite_samples.mean(dim=0)
        if self.elite_size > 1:
            self.variance = elite_samples.var(dim=0, unbiased=False) + 1e-8
        else:
            self.variance = torch.ones_like(self.mean) * 1e-4

        if verbose:
            logging.info(f"Best so far: {samples[best_sample_idx].unsqueeze(0)}")
            logging.info(f"New mean: {self.mean}")
            logging.info(f"New Variance: {self.variance}")

        return samples[best_sample_idx]


CrossEntropyMethod = CEM


class RandomSearch:
    """
    Performs optimization on a given callable objective function using Random Search.
    """

    def __init__(
        self,
        X,
        Y,
        obj_func,
        bounds,
        pop_size=50,
    ):
        """
        Parameters:
        -----------
        X: torch.tensor (shape: (N, d))
            Initial data points
        Y: torch.tensor (shape: (N, 1) or (N,))
            Initial function values
        obj_func: callable
            Objective function to be optimized.
        bounds: torch.tensor (shape: (2, d))
            Bounds of the parameter space.
        pop_size: int
            Population size evaluated per step.
        """
        self.d = X.shape[1]
        self.X = X.to(dtype=torch.float64)
        self.Y = Y.reshape(-1, 1).to(dtype=torch.float64)
        self.population_size = int(pop_size)

        # best candidate
        self.best_idx = torch.argmax(self.Y)
        self.best_x = self.X[self.best_idx, :]

        self.obj_func = obj_func
        self.bounds = bounds.to(dtype=torch.float64)

    def update_data(self, new_X, new_Y):
        """Update data, keep track of best so far"""
        self.X = torch.cat([self.X, new_X.to(self.X)], dim=0)
        self.Y = torch.cat([self.Y, new_Y.reshape(-1, 1).to(self.Y)], dim=0)

        # update best candidate
        self.best_idx = torch.argmax(self.Y)
        self.best_x = self.X[self.best_idx, :]

    def sample_np(self, num_samples=1):
        clipped_samples = np.random.uniform(
            self.bounds[0].detach().cpu().numpy(),
            self.bounds[1].detach().cpu().numpy(),
            size=(num_samples, self.d),
        )
        return torch.tensor(clipped_samples, dtype=torch.float64, device=self.X.device)

    def objective(self, x):
        """Objective function wrapper"""
        y = self.obj_func(x)
        if isinstance(y, torch.Tensor):
            y = y.detach()
        else:
            y = torch.tensor(y, dtype=torch.float64)
        return y.reshape(-1).squeeze(0)

    def step(self, maximize=True, verbose=False):
        """Performs one step of Random Search"""
        samples = self.sample_np(num_samples=self.population_size)

        scores = torch.stack([self.objective(sample.unsqueeze(0)) for sample in samples]).reshape(-1)

        if maximize:
            best_sample_idx = torch.argmax(scores)
        else:
            best_sample_idx = torch.argmin(scores)

        self.update_data(
            samples[best_sample_idx].unsqueeze(0),
            scores[best_sample_idx].reshape(-1, 1),
        )

        if verbose:
            logging.info(f"Best so far: {samples[best_sample_idx].unsqueeze(0)}")

        return samples[best_sample_idx]
