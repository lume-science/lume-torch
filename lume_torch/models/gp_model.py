import os
import logging
from pydantic import field_validator

import torch
from torch.distributions import Distribution as TDistribution
from botorch.models import SingleTaskGP, MultiTaskGP, ModelListGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal as GPMultivariateNormal
from botorch.models.transforms.input import ReversibleInputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from linear_operator.utils.cholesky import psd_safe_cholesky
from linear_operator.operators import DiagLinearOperator

from lume_torch.models.prob_model_base import (
    ProbModelBaseModel,
    TorchDistributionWrapper,
)


logger = logging.getLogger(__name__)


class GPModel(ProbModelBaseModel):
    """LUME-model class for Gaussian process (GP) models.

    This class wraps BoTorch/GPyTorch GP models (``SingleTaskGP``, ``MultiTaskGP``,
    or ``ModelListGP``) and exposes them through the LUME probabilistic model
    interface.

    Parameters
    ----------
    model : SingleTaskGP, MultiTaskGP, or ModelListGP
        A single-task or multi-task GP model, or a list of such models wrapped
        in a :class:`ModelListGP`.
    input_transformers : list of transforms, optional
        List of input transformers to apply to the input data. They are applied
        sequentially outside the underlying BoTorch model.
    output_transformers : list of transforms, optional
        List of output transformers to apply to the output data. They are
        applied sequentially outside the underlying BoTorch model.

    Methods
    -------
    get_input_size()
        Return the dimensionality of the input space.
    get_output_size()
        Return the dimensionality of the output space.
    likelihood()
        Return the likelihood module of the underlying GP model.
    mll(x, y)
        Compute the marginal log-likelihood for given inputs and targets.
    _get_predictions(input_dict, observation_noise=False)
        Implement the probabilistic model interface by returning predictive
        distributions for each output variable.

    Notes
    -----
    If ``input_transformers`` or ``output_transformers`` are provided, they are
    applied outside of the BoTorch model, regardless of any internal
    ``input_transform`` or ``outcome_transform`` configured on the model
    itself.

    """

    model: SingleTaskGP | MultiTaskGP | ModelListGP
    input_transformers: list[ReversibleInputTransform | torch.nn.Linear] | None = None
    output_transformers: (
        list[OutcomeTransform | ReversibleInputTransform | torch.nn.Linear] | None
    ) = None

    @field_validator("model", mode="before")
    def validate_gp_model(cls, v):
        if isinstance(v, (str, os.PathLike)):
            if os.path.exists(v):
                logger.info(f"Loading GP model from file: {v}")
                v = torch.load(v, weights_only=False)
            else:
                logger.error(f"GP model file not found: {v}")
                raise OSError(f"File {v} is not found.")
        return v

    @field_validator("input_transformers", "output_transformers", mode="before")
    def validate_transformers(cls, v):
        if v is None:
            return []
        if not isinstance(v, list):
            logger.error(f"Transformers must be a list, got {type(v)}")
            raise ValueError("Transformers must be passed as list.")
        loaded_transformers = []
        for t in v:
            if isinstance(t, (str, os.PathLike)):
                if os.path.exists(t):
                    logger.debug(f"Loading transformer from file: {t}")
                    t = torch.load(t, weights_only=False)
                else:
                    logger.error(f"Transformer file not found: {t}")
                    raise OSError(f"File {t} is not found.")
            loaded_transformers.append(t)
        return loaded_transformers

    def get_input_size(self) -> int:
        """Get the dimensionality of the input space.

        Returns
        -------
        int
            Number of input features expected by the GP model.

        Raises
        ------
        ValueError
            If the underlying model type is not supported.

        """
        if isinstance(self.model, SingleTaskGP):
            return self.model.train_inputs[0].shape[-1]
        elif isinstance(self.model, MultiTaskGP):
            return self.model.train_inputs[0].shape[-1] - 1
        elif isinstance(self.model, ModelListGP):
            first_model = self.model.models[0]
            if isinstance(first_model, SingleTaskGP):
                return first_model.train_inputs[0].shape[-1]
            elif isinstance(first_model, MultiTaskGP):
                return first_model.train_inputs[0].shape[-1] - 1
            else:
                logger.error(
                    f"Unsupported model type in ModelListGP: {type(first_model)}"
                )
                raise ValueError(
                    "ModelListGP must contain SingleTaskGP or MultiTaskGP models."
                )
        else:
            logger.error(f"Unsupported GP model type: {type(self.model)}")
            raise ValueError(
                "Model must be an instance of SingleTaskGP, MultiTaskGP or ModelListGP."
            )

    def get_output_size(self) -> int:
        """Get the dimensionality of the output space.

        Returns
        -------
        int
            Number of output dimensions produced by the GP model.

        Raises
        ------
        ValueError
            If the underlying model type is not supported.

        """
        if isinstance(self.model, ModelListGP):
            return sum(model.num_outputs for model in self.model.models)
        elif isinstance(self.model, (SingleTaskGP, MultiTaskGP)):
            return self.model.num_outputs
        else:
            logger.error(f"Unsupported GP model type: {type(self.model)}")
            raise ValueError(
                "Model must be an instance of SingleTaskGP, MultiTaskGP or ModelListGP."
            )

    def likelihood(self):
        """Return the likelihood module of the underlying GP model.

        Returns
        -------
        gpytorch.likelihoods.Likelihood
            The likelihood object associated with ``self.model``.

        """
        return self.model.likelihood

    def mll(self, x, y):
        """Compute the marginal log-likelihood (MLL).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        y : torch.Tensor
            Target tensor.

        Returns
        -------
        float
            Value of the marginal log-likelihood for the given data.

        """
        self.model.eval()
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        return mll(self.model(x), y).item()

    def _get_predictions(
        self,
        input_dict: dict[str, float | torch.Tensor],
        observation_noise: bool = False,
    ) -> dict[str, TDistribution]:
        """Get predictive distributions from the GP model.

        This implements the abstract method from :class:`ProbModelBaseModel` by
        constructing a BoTorch posterior and wrapping it as a distribution over
        the outputs.

        Parameters
        ----------
        input_dict : dict of str to float or torch.Tensor
            Dictionary of input variable names to values.
        observation_noise : bool, optional
            If ``True``, include observation noise in the posterior.

        Returns
        -------
        dict of str to TDistribution
            Dictionary of output variable names to predictive distributions.

        """
        # Reorder the input dictionary to match the model's input order
        input_dict = super()._arrange_inputs(input_dict)
        # Create tensor from input_dict
        x = super()._create_tensor_from_dict(input_dict)
        # Transform the input
        if self.input_transformers:
            x = self._transform_inputs(x)
        # Get the posterior distribution
        posterior = self._posterior(x, observation_noise=observation_noise)
        # Wrap the distribution in a torch distribution
        distribution = self._get_distribution(posterior)
        # Take mean and covariance of the distribution
        # posterior.mean preserves batch dim, while distribution.mean does not
        mean, covar = posterior.mean, distribution.covariance_matrix
        # Return a dictionary of output variable names to distributions
        return self._create_output_dict((mean, covar))

    def _posterior(self, x: torch.Tensor, observation_noise: bool = False):
        """Compute the posterior distribution at the given inputs.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor at which to compute the posterior.
        observation_noise : bool, optional
            If ``True``, include observation noise in the posterior.

        Returns
        -------
        botorch.posteriors.Posterior
            Posterior object from the model.

        """
        self.model.eval()
        posterior = self.model.posterior(x, observation_noise=observation_noise)
        return posterior

    def _get_distribution(self, posterior) -> TDistribution:
        """Get a torch-like distribution from the posterior.

        Checks that the resulting distribution exposes a
        ``covariance_matrix`` attribute.

        Parameters
        ----------
        posterior : botorch.posteriors.Posterior
            Posterior object from the model.

        Returns
        -------
        TDistribution
            Torch distribution object representing the posterior.

        Raises
        ------
        ValueError
            If the posterior distribution does not have a covariance matrix
            attribute.

        """
        if isinstance(posterior.distribution, TDistribution):
            d = posterior.distribution
        else:
            # Wrap the distribution in a torch distribution
            d = TorchDistributionWrapper(posterior.distribution)

        if not hasattr(d, "covariance_matrix"):
            raise ValueError(
                f"The posterior distribution {type(posterior.distribution)} does not have a covariance matrix attribute."
            )

        return d

    def _create_output_dict(
        self, output: tuple[torch.Tensor, torch.Tensor]
    ) -> dict[str, TDistribution]:
        """Convert GP mean and covariance into output distributions.

        The returned distributions are constructed as multivariate normal
        distributions (one per output variable). Currently only this type of
        distribution is supported.

        Parameters
        ----------
        output : tuple of (torch.Tensor, torch.Tensor)
            Tuple containing mean and covariance of the joint GP output.

        Returns
        -------
        dict of str to TDistribution
            Dictionary of output variable names to multivariate normal
            distributions.

        """
        output_distributions = {}
        mean, cov = output
        ss = mean.shape[1] if len(mean.shape) > 2 else mean.shape[0]  # sample size

        batch = mean.shape[0] if len(mean.shape) > 2 else None
        for i, name in enumerate(self.output_names):
            if batch is None:
                _mean = mean[:, i] if len(mean.shape) > 1 else mean
                _cov = torch.zeros(ss, ss, **self._tkwargs)
                _cov[:, :ss] = cov[i * ss : (i + 1) * ss, i * ss : (i + 1) * ss]
            else:
                _mean = mean[:, :, i]
                _cov = torch.zeros(batch, ss, ss, **self._tkwargs)
                _cov[:, :ss, :ss] = cov[:, i * ss : (i + 1) * ss, i * ss : (i + 1) * ss]

            # Check that the covariance matrix is positive definite
            _cov = self._check_covariance_matrix(_cov)

            if self.output_transformers:
                # TODO: make this more robust?
                # If we have two outputs, but transformer has length 1 (e.g. multitask),
                # we should apply the same transform to both outputs
                _mean = self._transform_mean(_mean, i)
                _cov = self._transform_covar(_cov, i)

            output_distributions[name] = GPMultivariateNormal(_mean, _cov)

        return output_distributions

    def _transform_inputs(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply configured input transformations to the inputs.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Ordered input tensor to be passed to the transformers.

        Returns
        -------
        torch.Tensor
            Transformed input tensor.

        """
        for transformer in self.input_transformers:
            if isinstance(transformer, ReversibleInputTransform):
                input_tensor = transformer.transform(input_tensor)
            else:
                input_tensor = transformer(input_tensor)
        return input_tensor

    def _get_transformer_params(
        self, transformer, i: int
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Extract scale factor and offset from a transformer.

        Parameters
        ----------
        transformer : ReversibleInputTransform or OutcomeTransform
            The transformer to extract parameters from.
        i : int
            Index of the output dimension.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor | None)
            Scale factor and offset (offset is None for covariance transforms).

        Raises
        ------
        NotImplementedError
            If the transformer type is not supported.

        """
        if isinstance(transformer, ReversibleInputTransform):
            try:
                scale_fac = transformer.coefficient[i]
                offset = transformer.offset[i]
            except IndexError:
                # If the transformer has only one coefficient, use it for all outputs
                # This is needed in the case of multitask models
                scale_fac = transformer.coefficient[0]
                offset = transformer.offset[0]
        elif isinstance(transformer, OutcomeTransform):
            try:
                scale_fac = transformer.stdvs.squeeze(0)[i]
                offset = transformer.means.squeeze(0)[i]
            except IndexError:
                # If the transformer has only one coefficient, use it for all outputs
                scale_fac = transformer.stdvs.squeeze(0)[0]
                offset = transformer.means.squeeze(0)[0]
        else:
            raise NotImplementedError(
                f"Output transformer {type(transformer)} is not supported."
            )
        return scale_fac, offset

    def _transform_mean(self, mean: torch.Tensor, i: int) -> torch.Tensor:
        """(Un-)transform the model output mean.

        Parameters
        ----------
        mean : torch.Tensor
            Output mean tensor from the model.
        i : int
            Index of the output dimension.

        Returns
        -------
        torch.Tensor
            (Un-)transformed output mean tensor.

        """
        for transformer in self.output_transformers:
            scale_fac, offset = self._get_transformer_params(transformer, i)
            mean = offset + scale_fac * mean
        return mean

    def _transform_covar(self, cov: torch.Tensor, i: int) -> torch.Tensor:
        """(Un-)transform the model output covariance matrix.

        Parameters
        ----------
        cov : torch.Tensor
            Output covariance matrix tensor from the model.
        i : int
            Index of the output variable.

        Returns
        -------
        torch.Tensor
            (Un-)transformed output covariance matrix tensor.

        """
        for transformer in self.output_transformers:
            scale_fac, _ = self._get_transformer_params(transformer, i)
            scale_fac = scale_fac.expand(cov.shape[:-1])
            scale_mat = DiagLinearOperator(scale_fac)
            cov = scale_mat @ cov @ scale_mat
        return cov

    def _check_covariance_matrix(self, cov: torch.Tensor) -> torch.Tensor:
        """Ensure the covariance matrix is positive definite.

        If the matrix is not positive definite, jitter is added until a
        successful Cholesky factorization is obtained.

        Parameters
        ----------
        cov : torch.Tensor
            Covariance matrix to check.

        Returns
        -------
        torch.Tensor
            Positive definite covariance matrix.

        """
        try:
            torch.linalg.cholesky(cov)
        except torch._C._LinAlgError:
            lm = psd_safe_cholesky(cov)  # determines jitter iteratively
            cov = lm @ lm.transpose(-1, -2)

        return cov
