# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Any, Callable, Literal, Optional, Tuple, Union

import torch
from torch import Tensor

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.potentials.base_potential import BasePotential
from sbi.samplers.importance.importance_sampling import importance_sample
from sbi.samplers.importance.sir import sampling_importance_resampling
from sbi.sbi_types import Shape, TorchTransform
from sbi.utils.sbiutils import mcmc_transform
from sbi.utils.torchutils import ensure_theta_batched


class ImportanceSamplingPosterior(NeuralPosterior):
    r"""Provides importance sampling to sample from the posterior.

    SNLE or SNRE train neural networks to approximate the likelihood(-ratios).
    `ImportanceSamplingPosterior` allows to estimate the posterior log-probability by
    estimating the normlalization constant with importance sampling. It also allows to
    perform importance sampling (with `.sample()`) and to draw approximate samples with
    sampling-importance-resampling (SIR) (with `.sir_sample()`)
    """

    def __init__(
        self,
        potential_fn: Union[Callable, BasePotential],
        proposal: Any,
        theta_transform: Optional[TorchTransform] = None,
        method: Literal["sir", "importance"] = "sir",
        oversampling_factor: int = 32,
        max_sampling_batch_size: int = 10_000,
        device: Optional[Union[str, torch.device]] = None,
        x_shape: Optional[torch.Size] = None,
    ):
        """
        Args:
            potential_fn: The potential function from which to draw samples. Must be a
                `BasePotential` or a `Callable` which takes `theta` and `x_o` as inputs.
            proposal: The proposal distribution.
            theta_transform: Transformation that is applied to parameters. Is not used
                during but only when calling `.map()`.
            method: Either of [`sir`|`importance`]. This sets the behavior of the
                `.sample()` method. With `sir`, approximate posterior samples are
                generated with sampling importance resampling (SIR). With
                `importance`, the `.sample()` method returns a tuple of samples and
                corresponding importance weights.
            oversampling_factor: Number of proposed samples from which only one is
                selected based on its importance weight.
            max_sampling_batch_size: The batch size of samples being drawn from the
                proposal at every iteration.
            device: Device on which to sample, e.g., "cpu", "cuda" or "cuda:0". If
                None, `potential_fn.device` is used.
            x_shape: Deprecated, should not be passed.
        """
        super().__init__(
            potential_fn,
            theta_transform=theta_transform,
            device=device,
            x_shape=x_shape,
        )

        self.proposal = proposal
        self._normalization_constant = None
        self.method = method
        self.theta_transform = theta_transform

        self.oversampling_factor = oversampling_factor
        self.max_sampling_batch_size = max_sampling_batch_size

        self._purpose = (
            "It provides sampling-importance resampling (SIR) to .sample() from the "
            "posterior and can evaluate the _unnormalized_ posterior density with "
            ".log_prob()."
        )
        self.x_shape = x_shape

    def to(self, device: Union[str, torch.device]) -> None:
        """
        Move the potential, the proposal and x_o to a new device.

        It also reinstantiates the posterior with the new device.

        Args:
            device: Device on which to move the posterior to.
        """
        self.device = device
        self.potential_fn.to(device)  # type: ignore
        self.proposal.to(device)
        x_o = None
        if hasattr(self, "_x") and (self._x is not None):
            x_o = self._x.to(device)

        self.theta_transform = mcmc_transform(self.proposal, device=device)
        super().__init__(
            self.potential_fn,
            theta_transform=self.theta_transform,
            device=device,
            x_shape=self.x_shape,
        )
        # super().__init__ erases the self._x, so we need to set it again
        if x_o is not None:
            self.set_default_x(x_o)

    def log_prob(
        self,
        theta: Tensor,
        x: Optional[Tensor] = None,
        track_gradients: bool = False,
        normalization_constant_params: Optional[dict] = None,
    ) -> Tensor:
        r"""Returns the log-probability of theta under the posterior.

        The normalization constant is estimated with importance sampling.

        Args:
            theta: Parameters $\theta$.
            track_gradients: Whether the returned tensor supports tracking gradients.
                This can be helpful for e.g. sensitivity analysis, but increases memory
                consumption.
            normalization_constant_params: Parameters passed on to
                `estimate_normalization_constant()`.

        Returns:
            `len($\theta$)`-shaped log-probability.
        """
        x = self._x_else_default_x(x)
        self.potential_fn.set_x(x)

        theta = ensure_theta_batched(torch.as_tensor(theta))

        with torch.set_grad_enabled(track_gradients):
            potential_values = self.potential_fn(
                theta.to(self._device), track_gradients=track_gradients
            )

            if normalization_constant_params is None:
                normalization_constant_params = dict()  # use defaults
            normalization_constant = self.estimate_normalization_constant(
                x, **normalization_constant_params
            )

            return (potential_values - torch.log(normalization_constant)).to(
                self._device
            )

    @torch.no_grad()
    def estimate_normalization_constant(
        self, x: Tensor, num_samples: int = 10_000, force_update: bool = False
    ) -> Tensor:
        """Returns the normalization constant via importance sampling.

        Args:
            num_samples: Number of importance samples used for the estimate.
            force_update: Whether to re-calculate the normlization constant when x is
                unchanged and have a cached value.
        """
        # Check if the provided x matches the default x (short-circuit on identity).
        is_new_x = self.default_x is None or (
            x is not self.default_x and (x != self.default_x).any()
        )

        not_saved_at_default_x = self._normalization_constant is None

        if is_new_x:  # Calculate at x; don't save.
            _, log_importance_weights = importance_sample(
                self.potential_fn,
                proposal=self.proposal,
                num_samples=num_samples,
            )
            return torch.mean(torch.exp(log_importance_weights))
        elif not_saved_at_default_x or force_update:  # Calculate at default_x; save.
            assert self.default_x is not None
            _, log_importance_weights = importance_sample(
                self.potential_fn,
                proposal=self.proposal,
                num_samples=num_samples,
            )
            self._normalization_constant = torch.mean(torch.exp(log_importance_weights))

        return self._normalization_constant.to(self._device)  # type: ignore

    def sample(
        self,
        sample_shape: Shape = torch.Size(),
        x: Optional[Tensor] = None,
        method: Optional[str] = None,
        oversampling_factor: int = 32,
        max_sampling_batch_size: int = 10_000,
        sample_with: Optional[str] = None,
        show_progress_bars: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Return samples from the approximate posterior distribution.

        Args:
            sample_shape: Shape of samples that are drawn from posterior.
            x: Observed data.
            method: Either of [`sir`|`importance`]. This sets the behavior of the
                `.sample()` method. With `sir`, approximate posterior samples are
                generated with sampling importance resampling (SIR). With
                `importance`, the `.sample()` method returns a tuple of samples and
                corresponding importance weights.
            oversampling_factor: Number of proposed samples from which only one is
                selected based on its importance weight.
            max_sampling_batch_size: The batch size of samples being drawn from the
                proposal at every iteration.
            show_progress_bars: Whether to show a progressbar during sampling.
        """

        method = self.method if method is None else method

        if sample_with is not None:
            raise ValueError(
                f"You set `sample_with={sample_with}`. As of sbi v0.18.0, setting "
                f"`sample_with` is no longer supported. You have to rerun "
                f"`.build_posterior(sample_with={sample_with}).`"
            )

        self.potential_fn.set_x(self._x_else_default_x(x))

        if method == "sir":
            return self._sir_sample(
                sample_shape,
                oversampling_factor=oversampling_factor,
                max_sampling_batch_size=max_sampling_batch_size,
                show_progress_bars=show_progress_bars,
            )
        elif method == "importance":
            return self._importance_sample(sample_shape)
        else:
            raise NameError

    def sample_batched(
        self,
        sample_shape: Shape,
        x: Tensor,
        max_sampling_batch_size: int = 10000,
        show_progress_bars: bool = True,
    ) -> Tensor:
        raise NotImplementedError(
            "Batched sampling is not implemented for ImportanceSamplingPosterior. \
           Alternatively you can use `sample` in a loop \
           [posterior.sample(theta, x_o) for x_o in x]."
        )

    def _importance_sample(
        self,
        sample_shape: Shape = torch.Size(),
        show_progress_bars: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Returns samples from the proposal and log of their importance weights.

        Args:
            sample_shape: Desired shape of samples that are drawn from posterior.
            sample_with: This argument only exists to keep backward-compatibility with
                `sbi` v0.17.2 or older. If it is set, we instantly raise an error.
            show_progress_bars: Whether to show sampling progress monitor.

        Returns:
            Samples and logarithm of corresponding importance weights.
        """
        num_samples = torch.Size(sample_shape).numel()
        samples, log_importance_weights = importance_sample(
            self.potential_fn,
            proposal=self.proposal,
            num_samples=num_samples,
            show_progress_bars=show_progress_bars,
        )

        samples = samples.reshape((*sample_shape, -1)).to(self._device)
        return samples, log_importance_weights.to(self._device)

    def _sir_sample(
        self,
        sample_shape: Shape = torch.Size(),
        oversampling_factor: int = 32,
        max_sampling_batch_size: int = 10_000,
        show_progress_bars: bool = False,
    ):
        r"""Returns approximate samples from posterior $p(\theta|x)$ via SIR.

        Args:
            sample_shape: Desired shape of samples that are drawn from posterior. If
                sample_shape is multidimensional we simply draw `sample_shape.numel()`
                samples and then reshape into the desired shape.
            x: Observed data.
            sample_with: This argument only exists to keep backward-compatibility with
                `sbi` v0.17.2 or older. If it is set, we instantly raise an error.
            oversampling_factor: Number of proposed samples form which only one is
                selected based on its importance weight.
            max_sampling_batch_size: The batchsize of samples being drawn from
                the proposal at every iteration. Used only in `sir_sample()`.
            show_progress_bars: Whether to show sampling progress monitor.

        Returns:
            Samples from posterior.
        """
        # Replace arguments that were not passed with their default.
        oversampling_factor = (
            self.oversampling_factor
            if oversampling_factor is None
            else oversampling_factor
        )
        max_sampling_batch_size = (
            self.max_sampling_batch_size
            if max_sampling_batch_size is None
            else max_sampling_batch_size
        )

        num_samples = torch.Size(sample_shape).numel()
        samples = sampling_importance_resampling(
            self.potential_fn,
            proposal=self.proposal,
            num_samples=num_samples,
            num_candidate_samples=oversampling_factor,
            show_progress_bars=show_progress_bars,
            max_sampling_batch_size=max_sampling_batch_size,
            device=self._device,
        )

        return samples.reshape((*sample_shape, -1)).to(self._device)

    def map(
        self,
        x: Optional[Tensor] = None,
        num_iter: int = 1_000,
        num_to_optimize: int = 100,
        learning_rate: float = 0.01,
        init_method: Union[str, Tensor] = "proposal",
        num_init_samples: int = 1_000,
        save_best_every: int = 10,
        show_progress_bars: bool = False,
        force_update: bool = False,
    ) -> Tensor:
        r"""Returns the maximum-a-posteriori estimate (MAP).

        The method can be interrupted (Ctrl-C) when the user sees that the
        log-probability converges. The best estimate will be saved in `self._map` and
        can be accessed with `self.map()`. The MAP is obtained by running gradient
        ascent from a given number of starting positions (samples from the posterior
        with the highest log-probability). After the optimization is done, we select the
        parameter set that has the highest log-probability after the optimization.

        Warning: The default values used by this function are not well-tested. They
        might require hand-tuning for the problem at hand.

        For developers: if the prior is a `BoxUniform`, we carry out the optimization
        in unbounded space and transform the result back into bounded space.

        Args:
            x: Deprecated - use `.set_default_x()` prior to `.map()`.
            num_iter: Number of optimization steps that the algorithm takes
                to find the MAP.
            learning_rate: Learning rate of the optimizer.
            init_method: How to select the starting parameters for the optimization. If
                it is a string, it can be either [`posterior`, `prior`], which samples
                the respective distribution `num_init_samples` times. If it is a
                tensor, the tensor will be used as init locations.
            num_init_samples: Draw this number of samples from the posterior and
                evaluate the log-probability of all of them.
            num_to_optimize: From the drawn `num_init_samples`, use the
                `num_to_optimize` with highest log-probability as the initial points
                for the optimization.
            save_best_every: The best log-probability is computed, saved in the
                `map`-attribute, and printed every `save_best_every`-th iteration.
                Computing the best log-probability creates a significant overhead
                (thus, the default is `10`.)
            show_progress_bars: Whether to show a progressbar during sampling from the
                posterior.
            force_update: Whether to re-calculate the MAP when x is unchanged and
                have a cached value.
            log_prob_kwargs: Will be empty for SNLE and SNRE. Will contain
                {'norm_posterior': True} for SNPE.

        Returns:
            The MAP estimate.
        """
        return super().map(
            x=x,
            num_iter=num_iter,
            num_to_optimize=num_to_optimize,
            learning_rate=learning_rate,
            init_method=init_method,
            num_init_samples=num_init_samples,
            save_best_every=save_best_every,
            show_progress_bars=show_progress_bars,
            force_update=force_update,
        )
