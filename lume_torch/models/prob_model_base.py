import logging
from typing import Union, Any, Tuple
from abc import abstractmethod

from pydantic import model_validator
import torch
from torch.distributions import Distribution as TDistribution

from lume_torch.variables import DistributionVariable
from lume_torch.models.utils import InputDictModel, format_inputs, itemize_dict
from lume_torch.base import LUMETorch

logger = logging.getLogger(__name__)


class ProbModelBaseModel(LUMETorch):  # TODO: brainstorm a better name
    """Abstract base class for probabilistic models.

    This class provides a common interface for probabilistic models. Subclasses must
    implement :meth:`_get_predictions`, which accepts a dictionary of inputs and returns
    a dictionary of output names mapped to distribution objects.

    The output distributions should be instances of
    :class:`torch.distributions.Distribution` (or be wrapped to follow the same
    interface).

    Attributes
    ----------
    output_variables : list of DistributionVariable
        List of output variables, which must be of :class:`DistributionVariable` type.
    device : torch.device or str
        Device on which the model will be evaluated. Defaults to ``"cpu"``.
    precision : {"double", "single"}
        Precision of the model. ``"double"`` maps to ``torch.double`` and
        ``"single"`` maps to ``torch.float``. Defaults to ``"double"``.

    Methods
    -------
    _get_predictions(input_dict, **kwargs)
        Abstract method that returns a dictionary of output distributions.
    _evaluate(input_dict, **kwargs)
        Evaluates the model and returns a dictionary of output distributions.
    input_validation(input_dict)
        Validates and normalizes the input dictionary prior to evaluation.
    output_validation(output_dict)
        Validates the output dictionary of distributions.

    Notes
    -----
    Subclasses must implement :meth:`_get_predictions`. That method should return a
    dictionary mapping each output variable name to a
    :class:`torch.distributions.Distribution` instance (or equivalent).

    """

    output_variables: list[DistributionVariable]
    device: Union[torch.device, str] = "cpu"
    precision: str = "double"

    @model_validator(mode="before")
    def validate_output_variables(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate output variables as DistributionVariable."""
        for variable in values["output_variables"]:
            if not isinstance(variable, DistributionVariable):
                raise ValueError(
                    "Output variables must be of type DistributionVariable."
                )
        return values

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # No range validation for probabilistic models currently implemented
        # self.input_validation_config = {x: "none" for x in self.input_names}

    @property
    def dtype(
        self,
    ):
        """Returns the data type for the model."""
        if self.precision == "double":
            return torch.double
        elif self.precision == "single":
            return torch.float
        else:
            raise ValueError(
                f"Unknown precision {self.precision}, "
                f"expected one of ['double', 'single']."
            )

    def _arrange_inputs(
        self, d: dict[str, Union[float, torch.Tensor]]
    ) -> dict[str, Union[float, torch.Tensor]]:
        """Enforce the order of input variables before tensor creation.

        Parameters
        ----------
        d : dict of str to float or torch.Tensor
            Dictionary mapping input variable names to scalar values or tensors.

        Returns
        -------
        dict of str to float or torch.Tensor
            New dictionary whose keys are ordered according to ``self.input_names``.

        """
        return {k: d[k] for k in self.input_names}

    @staticmethod
    def _create_tensor_from_dict(
        d: dict[str, Union[float, torch.Tensor]],
    ) -> torch.Tensor:
        """Create a 2D tensor from a dictionary of floats and tensors.

        Parameters
        ----------
        d : dict of str to float or torch.Tensor
            Dictionary of floats or tensors.

        Returns
        -------
        torch.Tensor
            A 2D Torch Tensor.

        """
        tensors = []

        for key, value in d.items():
            if isinstance(value, float):
                tensors.append(torch.tensor([value]))
            elif isinstance(value, torch.Tensor):
                tensors.append(value)
            else:
                raise ValueError(
                    f"Value for key '{key}' must be either a float or a torch tensor."
                )

        if all(isinstance(value, float) for value in d.values()):
            # All values are floats
            return torch.stack(tensors, dim=1)
        elif all(isinstance(tensor, torch.Tensor) for tensor in tensors):
            lengths = [tensor.size(0) for tensor in tensors]
            if len(set(lengths)) != 1:
                raise ValueError("All tensors must have the same length.")
            dim = tensors[0].dim()
            # Stack tensors into a multidimensional tensor
            return torch.stack(tensors, dim=dim)
        else:
            raise ValueError(
                "All values must be either floats or tensors, and all tensors must have the same length."
            )

    @abstractmethod
    def _get_predictions(
        self, input_dict: dict[str, float | torch.Tensor], **kwargs
    ) -> dict[str, TDistribution]:
        """Get predictions from the model.

        Parameters
        ----------
        input_dict : dict of str to float or torch.Tensor
            Dictionary of input variable names to values. Values can be floats or
            tensors of shape ``n`` or ``b x n`` (batch mode).
        **kwargs
            Additional keyword arguments passed through to the concrete
            implementation in subclasses.

        Returns
        -------
        dict of str to TDistribution
            A dictionary of output variable names to distributions.

        """
        pass

    def _evaluate(
        self, input_dict: dict[str, Union[float, torch.Tensor]], **kwargs
    ) -> dict[str, TDistribution]:
        """Evaluate the probabilistic model.

        Parameters
        ----------
        input_dict : dict of str to float or torch.Tensor
            Dictionary of input variable names to values. Values can be floats or
            tensors of shape ``n`` or ``b × n`` (batch mode).
        **kwargs
            Additional keyword arguments forwarded to :meth:`_get_predictions`.

        Returns
        -------
        dict of str to torch.distributions.Distribution
            Dictionary mapping output variable names to predictive distributions.

        """
        # Evaluate and get mean and variance for each output
        output_dict = self._get_predictions(input_dict, **kwargs)
        # Split multi-dimensional output into separate distributions and
        # return output dictionary
        return output_dict

    def input_validation(self, input_dict: dict[str, Union[float, torch.Tensor]]):
        """Validates input dictionary before evaluation.

        Parameters
        ----------
        input_dict : dict of str to float or torch.Tensor
            Input dictionary to validate.

        Returns
        -------
        dict of str to float or torch.Tensor
            Validated input dictionary.

        """
        # validate input type (ints only are cast to floats for scalars)
        validated_input = InputDictModel(input_dict=input_dict).input_dict
        # format inputs as tensors w/o changing the dtype
        formatted_inputs = format_inputs(validated_input)
        # itemize inputs for validation
        itemized_inputs = itemize_dict(formatted_inputs)

        for ele in itemized_inputs:
            # validate values that were in the torch tensor
            # any ints in the torch tensor will be cast to floats by Pydantic
            # but others will be caught, e.g. booleans
            ele = InputDictModel(input_dict=ele).input_dict
            # validate each value based on its var class and config
            super().input_validation(ele)

        # return the validated input dict for consistency w/ casting ints to floats
        if any([isinstance(value, torch.Tensor) for value in validated_input.values()]):
            validated_input = {
                k: v.to(**self._tkwargs).squeeze(-1) for k, v in validated_input.items()
            }

        return validated_input

    def output_validation(self, output_dict: dict[str, TDistribution]):
        """Itemizes tensors before performing output validation.

        Parameters
        ----------
        output_dict : dict of str to TDistribution
            Output dictionary to validate.

        """
        itemized_outputs = itemize_dict(output_dict)
        for ele in itemized_outputs:
            super().output_validation(ele)


class TorchDistributionWrapper(TDistribution):
    """Wrap a custom distribution to provide a torch-like interface.

    This class adapts an arbitrary "distribution-like" object so that it behaves
    like a :class:`torch.distributions.Distribution` from the perspective of
    downstream code.

    Parameters
    ----------
    custom_dist : object
        An instance of a custom distribution with methods or attributes
        providing mean, variance (or covariance), log probability, and
        sampling functionality.

    Attributes
    ----------
    custom_dist : object
        The underlying wrapped distribution object.
    device : torch.device
        Device on which returned tensors are allocated. Defaults to CPU.
    dtype : torch.dtype
        Default floating point dtype for returned tensors. Defaults to
        ``torch.double``.

    """

    def __init__(self, custom_dist):
        """Initialize the TorchDistributionWrapper.

        Parameters
        ----------
        custom_dist : object
            An instance of a custom distribution with methods:
            mean, variance, log_prob, sample, and rsample.

        """
        super().__init__()
        self.custom_dist = custom_dist
        self.device = torch.device("cpu")
        self.dtype = torch.double

    @property
    def _tkwargs(self):
        return {"device": self.device, "dtype": self.dtype}

    @property
    def mean(self) -> torch.Tensor:
        """Return the mean of the custom distribution."""
        attribute_names = ["mean"]
        result, _ = self._get_attr(attribute_names)
        return result

    @property
    def variance(self) -> torch.Tensor:
        """Return the variance of the custom distribution."""
        attribute_names = ["variance", "var", "cov", "covariance", "covariance_matrix"]
        result, attr_name = self._get_attr(attribute_names)

        if attr_name in ["cov", "covariance", "covariance_matrix"]:
            return torch.diagonal(torch.tensor(result))

        return result

    @property
    def covariance_matrix(self) -> torch.Tensor:
        """Return the covariance matrix of the custom distribution."""
        attribute_names = ["covariance_matrix", "cov", "covariance"]
        result, _ = self._get_attr(attribute_names)
        return result

    def confidence_region(self) -> Tuple[torch.tensor, torch.tensor]:
        """Return a 2-sigma confidence region around the mean.

        Adapted from :mod:`gpytorch.distributions.multivariate_normal`.

        Returns
        -------
        tuple of torch.Tensor
            Pair of tensors of size ``... × N``, where ``N`` is the
            dimensionality of the random variable. The first (second) tensor is
            the lower (upper) end of the confidence region.

        Raises
        ------
        AttributeError
            If the wrapped distribution does not expose a variance attribute.

        """
        try:
            stddev = self.variance.sqrt()
            std2 = stddev.mul_(2)
            mean = self.mean
            return mean.sub(std2), mean.add(std2)
        except AttributeError:
            raise AttributeError("The distribution does not have a variance attribute.")

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute the log probability for a given value.

        Parameters
        ----------
        value : torch.Tensor
            The value for which to compute log probability.

        Returns
        -------
        torch.Tensor
            The log probability.

        """
        attribute_names = ["log_prob", "log_likelihood", "logpdf"]
        result, _ = self._get_attr(attribute_names, value)
        return result

    # TODO: check fn signature
    def rsample(self, sample_shape: torch.Size()) -> torch.Tensor:
        """Generate reparameterized samples from the custom distribution.

        Parameters
        ----------
        sample_shape : torch.Size
            The shape of samples to generate.

        Returns
        -------
        torch.Tensor
            Reparameterized samples.

        """
        # Fallback to sample if rsample is not implemented
        attribute_names = ["rsample", "sample", "rvs"]
        result, _ = self._get_attr(attribute_names, sample_shape)
        return result

    def sample(self, sample_shape: torch.Size()) -> torch.Tensor:
        """Generate samples from the custom distribution (non-differentiable if using sample).

        Parameters
        ----------
        sample_shape : torch.Size
            The shape of samples to generate.

        Returns
        -------
        torch.Tensor
            Samples from the distribution.

        """
        attribute_names = ["sample", "rvs"]
        # Assume non-torch.Distribution takes an integer sample_shape
        sample_shape = sample_shape.numel()
        result, _ = self._get_attr(attribute_names, sample_shape)
        return result

    def __repr__(self):
        return f"TorchDistributionWrapper({self.custom_dist})"

    def _get_attr(self, attribute_names, value=None):
        """Return the first available attribute from the wrapped distribution.

        Parameters
        ----------
        attribute_names : sequence of str
            Candidate attribute names to query on ``self.custom_dist``.
        value : Any, optional
            Optional value that is forwarded to the attribute if it is
            callable (e.g. ``log_prob(value)``).

        Returns
        -------
        tuple of (torch.Tensor, str)
            A tuple containing the attribute value converted to a tensor and
            the name of the attribute that was found.

        Raises
        ------
        AttributeError
            If none of the attributes in ``attribute_names`` are found on
            ``self.custom_dist``.

        """
        for attr_name in attribute_names:
            attr_value = getattr(self.custom_dist, attr_name, None)
            if attr_value is not None:
                if callable(attr_value):
                    result = attr_value(value) if value is not None else attr_value()
                else:
                    result = attr_value

                return torch.tensor(result, **self._tkwargs), attr_name

        raise AttributeError(
            f"None of the attributes {attribute_names} found in the distribution."
        )
