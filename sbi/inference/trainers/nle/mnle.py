# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from copy import deepcopy
from typing import Any, Callable, Dict, Literal, Optional, Union

from torch.distributions import Distribution

from sbi.inference.posteriors import MCMCPosterior, RejectionPosterior, VIPosterior
from sbi.inference.potentials import likelihood_estimator_based_potential
from sbi.inference.trainers.nle.nle_base import LikelihoodEstimatorTrainer
from sbi.neural_nets.estimators import MixedDensityEstimator
from sbi.sbi_types import TensorboardSummaryWriter, TorchModule
from sbi.utils.sbiutils import del_entries
from sbi.utils.user_input_checks import check_prior


class MNLE(LikelihoodEstimatorTrainer):
    """Mixed Neural Likelihood Estimation (MNLE) [1].

    Like NLE, but designed to be applied to data with mixed types, e.g., continuous
    data and discrete data like they occur in decision-making experiments
    (reation times and choices).

    [1] Flexible and efficient simulation-based inference for models of
    decision-making, Boelts et al. 2021,
    https://www.biorxiv.org/content/10.1101/2021.12.22.473472v2
    """

    def __init__(
        self,
        prior: Optional[Distribution] = None,
        density_estimator: Union[str, Callable] = "mnle",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[TensorboardSummaryWriter] = None,
        show_progress_bars: bool = True,
    ):
        r"""Initialize MNLE.

        Args:
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them. If `None`, the
                prior must be passed to `.build_posterior()`.
            density_estimator: If it is a string, it must be "mnle" to use the
                preconfiugred neural nets for MNLE. Alternatively, a function
                that builds a custom neural network can be provided. The function will
                be called with the first batch of simulations (theta, x), which can
                thus be used for shape inference and potentially for z-scoring. It
                needs to return a PyTorch `nn.Module` implementing the density
                estimator. The density estimator needs to provide the methods
                `.log_prob`, `.log_prob_iid()` and `.sample()`.
            device: Training device, e.g., "cpu", "cuda" or "cuda:{0, 1, ...}".
            logging_level: Minimum severity of messages to log. One of the strings
                INFO, WARNING, DEBUG, ERROR and CRITICAL.
            summary_writer: A tensorboard `SummaryWriter` to control, among others, log
                file location (default is `<current working directory>/logs`.)
            show_progress_bars: Whether to show a progressbar during simulation and
                sampling.
        """

        if isinstance(density_estimator, str):
            assert (
                density_estimator == "mnle"
            ), f"""MNLE can be used with preconfigured 'mnle' density estimator only,
                not with {density_estimator}."""
        kwargs = del_entries(locals(), entries=("self", "__class__"))
        super().__init__(**kwargs)

    def train(
        self,
        training_batch_size: int = 200,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: int = 2**31 - 1,
        clip_max_norm: Optional[float] = 5.0,
        resume_training: bool = False,
        discard_prior_samples: bool = False,
        retrain_from_scratch: bool = False,
        show_train_summary: bool = False,
        dataloader_kwargs: Optional[Dict] = None,
    ) -> MixedDensityEstimator:
        density_estimator = super().train(
            **del_entries(locals(), entries=("self", "__class__"))
        )
        assert isinstance(
            density_estimator, MixedDensityEstimator
        ), f"""Internal net must be of type
            MixedDensityEstimator but is {type(density_estimator)}."""
        return density_estimator

    def build_posterior(
        self,
        density_estimator: Optional[TorchModule] = None,
        prior: Optional[Distribution] = None,
        sample_with: Literal["mcmc", "rejection", "vi"] = "mcmc",
        mcmc_method: Literal[
            "slice_np",
            "slice_np_vectorized",
            "hmc_pyro",
            "nuts_pyro",
            "slice_pymc",
            "hmc_pymc",
            "nuts_pymc",
        ] = "slice_np_vectorized",
        vi_method: Literal["rKL", "fKL", "IW", "alpha"] = "rKL",
        mcmc_parameters: Optional[Dict[str, Any]] = None,
        vi_parameters: Optional[Dict[str, Any]] = None,
        rejection_sampling_parameters: Optional[Dict[str, Any]] = None,
    ) -> Union[MCMCPosterior, RejectionPosterior, VIPosterior]:
        r"""Build posterior from the neural density estimator.

        SNLE trains a neural network to approximate the likelihood $p(x|\theta)$. The
        posterior wraps the trained network such that one can directly evaluate the
        unnormalized posterior log probability $p(\theta|x) \propto p(x|\theta) \cdot
        p(\theta)$ and draw samples from the posterior with MCMC or rejection sampling.

        Args:
            density_estimator: The density estimator that the posterior is based on.
                If `None`, use the latest neural density estimator that was trained.
            prior: Prior distribution.
            sample_with: Method to use for sampling from the posterior. Must be one of
                [`mcmc` | `rejection` | `vi`].
            mcmc_method: Method used for MCMC sampling, one of `slice_np`,
                `slice_np_vectorized`, `hmc_pyro`, `nuts_pyro`, `slice_pymc`,
                `hmc_pymc`, `nuts_pymc`. `slice_np` is a custom
                numpy implementation of slice sampling. `slice_np_vectorized` is
                identical to `slice_np`, but if `num_chains>1`, the chains are
                vectorized for `slice_np_vectorized` whereas they are run sequentially
                for `slice_np`. The samplers ending on `_pyro` are using Pyro, and
                likewise the samplers ending on `_pymc` are using PyMC.
            vi_method: Method used for VI, one of [`rKL`, `fKL`, `IW`, `alpha`]. Note
                some of the methods admit a `mode seeking` property (e.g. rKL) whereas
                some admit a `mass covering` one (e.g fKL).
            mcmc_parameters: Additional kwargs passed to `MCMCPosterior`.
            vi_parameters: Additional kwargs passed to `VIPosterior`.
            rejection_sampling_parameters: Additional kwargs passed to
                `RejectionPosterior`.

        Returns:
            Posterior $p(\theta|x)$  with `.sample()` and `.log_prob()` methods
            (the returned log-probability is unnormalized).
        """
        if prior is None:
            assert (
                self._prior is not None
            ), """You did not pass a prior. You have to pass the prior either at
            initialization `inference = SNLE(prior)` or to `.build_posterior
            (prior=prior)`."""
            prior = self._prior
        else:
            check_prior(prior)

        if density_estimator is None:
            likelihood_estimator = self._neural_net
            # If internal net is used device is defined.
            device = self._device
        else:
            likelihood_estimator = density_estimator
            # Otherwise, infer it from the device of the net parameters.
            device = next(density_estimator.parameters()).device.type

        assert isinstance(
            likelihood_estimator, MixedDensityEstimator
        ), f"""net must be of type MixedDensityEstimator but is {
            type(likelihood_estimator)
        }."""

        (
            potential_fn,
            theta_transform,
        ) = likelihood_estimator_based_potential(likelihood_estimator, prior, x_o=None)

        if sample_with == "mcmc":
            self._posterior = MCMCPosterior(
                potential_fn=potential_fn,
                theta_transform=theta_transform,
                proposal=prior,
                method=mcmc_method,
                device=device,
                **mcmc_parameters or {},
            )
        elif sample_with == "rejection":
            self._posterior = RejectionPosterior(
                potential_fn=potential_fn,
                proposal=prior,
                device=device,
                **rejection_sampling_parameters or {},
            )
        elif sample_with == "vi":
            self._posterior = VIPosterior(
                potential_fn=potential_fn,
                theta_transform=theta_transform,
                prior=prior,  # type: ignore
                vi_method=vi_method,
                device=device,
                **vi_parameters or {},
            )
        else:
            raise NotImplementedError

        # Store models at end of each round.
        self._model_bank.append(deepcopy(self._posterior))

        return deepcopy(self._posterior)
