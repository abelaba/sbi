from dataclasses import asdict
from typing import Callable, Literal, Union

import torch
from torch.distributions import Distribution

from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.posteriors.importance_posterior import ImportanceSamplingPosterior
from sbi.inference.posteriors.mcmc_posterior import MCMCPosterior
from sbi.inference.posteriors.posterior_parameters import (
    DirectPosteriorParameters,
    ImportanceSamplingPosteriorParameters,
    MCMCPosteriorParameters,
    RejectionPosteriorParameters,
    VIPosteriorParameters,
    VectorFieldPosteriorParameters,
)
from sbi.inference.posteriors.rejection_posterior import RejectionPosterior
from sbi.inference.posteriors.vector_field_posterior import VectorFieldPosterior
from sbi.inference.posteriors.vi_posterior import VIPosterior
from sbi.neural_nets.estimators.base import (
    ConditionalDensityEstimator,
    ConditionalEstimator,
    ConditionalVectorFieldEstimator,
)
from sbi.neural_nets.ratio_estimators import RatioEstimator


class PosteriorFactory:
    _creators = {}

    @classmethod
    def register(
        cls,
        sample_with: Literal[
            "mcmc", "rejection", "vi", "importance", "direct", "sde", "ode"
        ],
        creator_func: Callable,
    ) -> None:
        cls._creators[sample_with] = creator_func

    @classmethod
    def create(
        cls,
        estimator: Union[RatioEstimator, ConditionalEstimator],
        sample_with: Literal[
            "mcmc", "rejection", "vi", "importance", "direct", "sde", "ode"
        ],
        prior: Distribution,
        device: Union[str, torch.device],
        posterior_parameters: Union[
            VIPosteriorParameters,
            VectorFieldPosteriorParameters,
            ImportanceSamplingPosteriorParameters,
            MCMCPosteriorParameters,
            DirectPosteriorParameters,
            RejectionPosteriorParameters,
        ],
        get_potential_fn: Callable,
    ) -> NeuralPosterior:
        creator = cls._creators[sample_with]
        return creator(
            estimator,
            sample_with,
            prior,
            device,
            posterior_parameters,
            get_potential_fn,
        )


def _create_mcmc_posterior(
    estimator: Union[RatioEstimator, ConditionalEstimator],
    sample_with: Literal["mcmc"],
    prior: Distribution,
    device: Union[str, torch.device],
    posterior_parameters: MCMCPosteriorParameters,
    get_potential_fn: Callable,
) -> MCMCPosterior:
    potential_fn, theta_transform = get_potential_fn()
    return MCMCPosterior(
        potential_fn=potential_fn,
        theta_transform=theta_transform,
        proposal=prior,
        device=device,
        **asdict(posterior_parameters),
    )


def _create_rejection_posterior(
    estimator: Union[RatioEstimator, ConditionalEstimator],
    sample_with: Literal["rejection"],
    prior: Distribution,
    device: Union[str, torch.device],
    posterior_parameters: RejectionPosteriorParameters,
    get_potential_fn: Callable,
) -> RejectionPosterior:
    potential_fn, _ = get_potential_fn()
    return RejectionPosterior(
        potential_fn=potential_fn,
        proposal=prior,
        device=device,
        **asdict(posterior_parameters),
    )


def _create_vi_posterior(
    estimator: Union[RatioEstimator, ConditionalEstimator],
    sample_with: Literal["vi"],
    prior: Distribution,
    device: Union[str, torch.device],
    posterior_parameters: VIPosteriorParameters,
    get_potential_fn: Callable,
) -> VIPosterior:
    potential_fn, theta_transform = get_potential_fn()
    return VIPosterior(
        potential_fn=potential_fn,
        theta_transform=theta_transform,
        prior=prior,
        device=device,
        **asdict(posterior_parameters),
    )


def _create_importance_posterior(
    estimator: Union[RatioEstimator, ConditionalEstimator],
    sample_with: Literal["importance"],
    prior: Distribution,
    device: Union[str, torch.device],
    posterior_parameters: ImportanceSamplingPosteriorParameters,
    get_potential_fn: Callable,
) -> ImportanceSamplingPosterior:
    potential_fn, _ = get_potential_fn()
    return ImportanceSamplingPosterior(
        potential_fn=potential_fn,
        proposal=prior,
        device=device,
        **asdict(posterior_parameters),
    )


def _create_direct_posterior(
    estimator: Union[RatioEstimator, ConditionalEstimator],
    sample_with: Literal["direct"],
    prior: Distribution,
    device: Union[str, torch.device],
    posterior_parameters: DirectPosteriorParameters,
    get_potential_fn: Callable,
) -> DirectPosterior:
    assert isinstance(estimator, ConditionalDensityEstimator), (
        "Expected estimator to be ConditionalDensityEstimator",
        "got {type(estimator).__name__}",
    )
    return DirectPosterior(
        posterior_estimator=estimator,
        prior=prior,
        device=device,
        **asdict(posterior_parameters),
    )


def _create_vectorfield_posterior(
    estimator: Union[RatioEstimator, ConditionalEstimator],
    sample_with: Literal["sde", "ode"],
    prior: Distribution,
    device: Union[str, torch.device],
    posterior_parameters: VectorFieldPosteriorParameters,
    get_potential_fn: Callable,
) -> VectorFieldPosterior:
    assert isinstance(estimator, ConditionalVectorFieldEstimator), (
        "Expected estimator to be ConditionalVectorFieldEstimator",
        "got {type(estimator).__name__}",
    )
    return VectorFieldPosterior(
        vector_field_estimator=estimator,
        prior=prior,
        device=device,
        sample_with=sample_with,
        **asdict(posterior_parameters),
    )


# Registration
PosteriorFactory.register("mcmc", _create_mcmc_posterior)
PosteriorFactory.register("rejection", _create_rejection_posterior)
PosteriorFactory.register("vi", _create_vi_posterior)
PosteriorFactory.register("importance", _create_importance_posterior)
PosteriorFactory.register("direct", _create_direct_posterior)
PosteriorFactory.register("sde", _create_vectorfield_posterior)
PosteriorFactory.register("ode", _create_vectorfield_posterior)
