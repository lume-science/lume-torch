from typing import Union, Any, Tuple
from abc import abstractmethod

from pydantic import model_validator
import torch
from torch.distributions import Distribution as TDistribution

from lume_model.variables import DistributionVariable
from lume_model.models.utils import format_inputs, itemize_dict
from lume_model.base import LUMEBaseModel


class ProbModelBaseModel(LUMEBaseModel):  # TODO: brainstorm a better name
    """Abstract base class for probabilistic models.

    This class provides a common interface for probabilistic models. All subclasses need to
     implement a `_get_predictions` method that accepts a dictionary input, and returns a
     dictionary of output names to _distributions_. The distributions should be instances of
    `torch.distributions.Distribution`.

    Attributes:
        output_variables: List of output variables, which should be of DistributionVariable type.
        device: Device on which the model will be evaluated. Defaults to "cpu".
        precision: Precision of the model, either "double" or "single". Defaults to "double".
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

    @property
    def _tkwargs(self) -> dict:
        return {"device": self.device, "dtype": self.dtype}

    def _arrange_inputs(
        self, d: dict[str, Union[float, torch.Tensor]]
    ) -> dict[str, Union[float, torch.Tensor]]:
        """Enforces order of input variables before creating a tensor.

        Args:
            d: Dictionary of input variable names to tensors.

        Returns:
            Ordered input tensor.
        """
        return {k: d[k] for k in self.input_names}

    @staticmethod
    def _create_tensor_from_dict(
        d: dict[str, Union[float, torch.Tensor]],
    ) -> torch.Tensor:
        """Create a 2D tensor from a dictionary of floats and tensors.

        Args:
            d: Dictionary of floats or tensors.

        Returns:
            A Torch Tensor."""
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

        Args:
             input_dict: Dictionary of input variable names to values. Values can be floats or
                `n` or `b x n` (batch mode) torch tensors.
        Returns:
            A dictionary of output variable names to distributions.
        """
        pass

    def _evaluate(
        self, input_dict: dict[str, Union[float, torch.Tensor]], **kwargs
    ) -> dict[str, TDistribution]:
        """Evaluate the model.

        Args:
            input_dict: Dictionary of input variable names to values. Values can be floats or
                `n` or `b x n` (batch mode) torch tensors.

        Returns:
            A dictionary of output variable names to distributions.
        """
        # Evaluate and get mean and variance for each output
        output_dict = self._get_predictions(input_dict, **kwargs)
        # Split multi-dimensional output into separate distributions and
        # return output dictionary
        return output_dict

    def input_validation(self, input_dict: dict[str, Union[float, torch.Tensor]]):
        """Validates input dictionary before evaluation.

        Args:
            input_dict: Input dictionary to validate.

        Returns:
            Validated input dictionary.
        """
        # validate original inputs (catches dtype mismatches)
        super().input_validation(input_dict)

        # format inputs as tensors w/o changing the dtype
        formatted_inputs = format_inputs(input_dict.copy())

        # cast tensors to expected dtype and device
        formatted_inputs = {
            k: v.to(**self._tkwargs).squeeze(-1) for k, v in formatted_inputs.items()
        }

        return formatted_inputs

    def output_validation(self, output_dict: dict[str, TDistribution]):
        """Itemizes tensors before performing output validation."""
        itemized_outputs = itemize_dict(output_dict)
        for ele in itemized_outputs:
            super().output_validation(ele)


class TorchDistributionWrapper(TDistribution):
    """Wraps any distribution to provide a torch.distributions-like interface."""

    def __init__(self, custom_dist):
        """
        Args:
            custom_dist: An instance of a custom distribution with methods:
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
        """
        Adapted from gpytorch.distributions.multivariate_normal
        Returns 2 standard deviations above and below the mean.

        Returns:
            Pair of tensors of size `... x N`, where N is the
            dimensionality of the random variable. The first (second) Tensor is the
            lower (upper) end of the confidence region.
        """
        try:
            stddev = self.variance.sqrt()
            std2 = stddev.mul_(2)
            mean = self.mean
            return mean.sub(std2), mean.add(std2)
        except AttributeError:
            raise AttributeError("The distribution does not have a variance attribute.")

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute the log probability for a given value."""
        attribute_names = ["log_prob", "log_likelihood", "logpdf"]
        result, _ = self._get_attr(attribute_names, value)
        return result

    # TODO: check fn signature
    def rsample(self, sample_shape: torch.Size()) -> torch.Tensor:
        """Generate reparameterized samples from the custom distribution."""
        # Fallback to sample if rsample is not implemented
        attribute_names = ["rsample", "sample", "rvs"]
        result, _ = self._get_attr(attribute_names, sample_shape)
        return result

    def sample(self, sample_shape: torch.Size()) -> torch.Tensor:
        """Generate samples from the custom distribution (non-differentiable if using sample)."""
        attribute_names = ["sample", "rvs"]
        # Assume non-torch.Distribution takes an integer sample_shape
        sample_shape = sample_shape.numel()
        result, _ = self._get_attr(attribute_names, sample_shape)
        return result

    def __repr__(self):
        return f"TorchDistributionWrapper({self.custom_dist})"

    def _get_attr(self, attribute_names, value=None):
        """Get the first attribute that is found in the distribution."""
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
