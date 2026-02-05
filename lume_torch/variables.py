"""
This module contains definitions of LUME-model variables for use with lume tools.
Variables are designed as pure descriptors and thus aren't intended to hold actual values,
but they can be used to validate encountered values.
"""

import logging
from typing import Any, Optional, Tuple, Type, Union

import torch
from torch import Tensor
from torch.distributions import Distribution as TDistribution
from pydantic import field_validator, model_validator, ConfigDict

from lume.variables import Variable, ScalarVariable, NDVariable, ConfigEnum

logger = logging.getLogger(__name__)

# Rename base ScalarVariable for internal use
_BaseScalarVariable = ScalarVariable

# Re-export base classes for backward compatibility and clean API
__all__ = [
    "Variable",
    "ScalarVariable",  # This will be TorchScalarVariable for backwards compatibility
    "NDVariable",
    "TorchNDVariable",
    "ConfigEnum",
    "DistributionVariable",
    "get_variable",
]


class DistributionVariable(Variable):
    """Variable for distributions. Must be a subclass of torch.distributions.Distribution.

    Attributes
    ----------
    unit : str, optional
        Unit associated with the variable.

    """

    unit: Optional[str] = None

    def validate_value(self, value: TDistribution, config: ConfigEnum = None):
        """Validates the given value.

        Parameters
        ----------
        value : Distribution
            The value to be validated.
        config : ConfigEnum, optional
            The configuration for validation. Defaults to None.
            Allowed values are "none", "warn", and "error".

        Raises
        ------
        TypeError
            If the value is not an instance of Distribution.

        """
        config = self.default_validation_config if config is None else config
        if isinstance(config, str):
            config = ConfigEnum(config)
        # mandatory validation
        self._validate_value_type(value)
        # optional validation
        if config != ConfigEnum.NULL:
            pass  # not implemented

    @staticmethod
    def _validate_value_type(value: TDistribution):
        if not isinstance(value, TDistribution):
            raise TypeError(
                f"Expected value to be of type {TDistribution}, "
                f"but received {type(value)}."
            )


class TorchScalarVariable(_BaseScalarVariable):
    """Variable for scalar values represented as PyTorch tensors.

    This class extends ScalarVariable to support scalar values as torch.Tensor
    with 0 or 1 dimensions (i.e., a scalar tensor or a single-element tensor).

    Attributes
    ----------
    default_value : Tensor | None
        Default value for the variable (must be 0D or 1D with size 1).
    dtype : torch.dtype | None
        Optional data type of the tensor. If specified, validates that tensor values
        match this exact dtype. If None (default), only validates that the dtype is
        a floating-point type without enforcing a specific precision.
    value_range : tuple[float, float] | None
        Value range that is considered valid for the variable.
    unit : str | None
        Unit associated with the variable.

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    default_value: Optional[Union[Tensor, float]] = None
    dtype: Optional[torch.dtype] = None

    @field_validator("dtype", mode="before")
    @classmethod
    def validate_dtype(cls, value):
        """Convert dtype string to torch dtype if needed and validate it's a float type."""
        if value is None:
            return None

        if isinstance(value, str):
            dtype_map = {
                "float16": torch.float16,
                "float32": torch.float32,
                "float64": torch.float64,
                "half": torch.half,
                "float": torch.float,
                "double": torch.double,
            }
            if value in dtype_map:
                value = dtype_map[value]
            else:
                raise ValueError(
                    f"Unknown or unsupported dtype string: {value}. "
                    f"Supported dtypes are: {list(dtype_map.keys())}"
                )

        # Validate that the dtype is a floating-point type
        if not value.is_floating_point:
            raise ValueError(f"dtype must be a floating-point type, got {value}")
        return value

    @model_validator(mode="after")
    def validate_default_value(self):
        if self.default_value is not None:
            self._validate_value_type(self.default_value)
            self._validate_dtype(self.default_value)
            if self.value_range is not None:
                scalar_value = (
                    self.default_value.item()
                    if isinstance(self.default_value, Tensor)
                    else self.default_value
                )
                if not self._value_is_within_range(scalar_value):
                    raise ValueError(
                        "Default value ({}) is out of valid range: ([{},{}]).".format(
                            scalar_value, *self.value_range
                        )
                    )
        return self

    def validate_value(self, value: Union[Tensor, float], config: ConfigEnum = None):
        """Validates the given tensor or float value.

        Parameters
        ----------
        value : Tensor | float
            The value to be validated. If a tensor, must be 0D or 1D with size 1.
        config : ConfigEnum, optional
            The configuration for validation. Defaults to None.
            Allowed values are "none", "warn", and "error".

        Raises
        ------
        TypeError
            If the value is not a torch.Tensor or float.
        ValueError
            If a tensor has more than 1 dimension, or if 1D with size != 1,
            or if tensor dtype is not a float type, or if value is out of range.

        """
        config = self.default_validation_config if config is None else config
        if isinstance(config, str):
            config = ConfigEnum(config)

        # mandatory validation
        self._validate_value_type(value)
        self._validate_dtype(value)
        self._validate_read_only(value)

        # optional validation
        if config != ConfigEnum.NULL:
            scalar_value = value.item() if isinstance(value, Tensor) else value
            self._validate_value_is_within_range(scalar_value, config=config)

    def _validate_value_type(self, value):
        """Validates that value is a torch.Tensor (0D, 1D or batched 1D) or a regular float/int."""
        if isinstance(value, Tensor):
            if value.ndim == 0:
                pass  # scalar tensor, valid
            elif value.ndim == 1:
                pass  # 1D tensor (single scalar or batch of scalars), valid
            elif value.ndim > 1 and value.shape[-1] == 1:
                pass  # Batched scalars with shape (batch_size, 1, ...), valid
            else:
                raise ValueError(
                    f"Expected tensor with 0 or 1 dimensions, or multi-dimensional tensor "
                    f"with last dimension equal to 1 for batched scalar values, "
                    f"but got {value.ndim} dimensions with shape {value.shape}."
                )
        else:
            # Delegate to parent class for non-tensor validation
            _BaseScalarVariable._validate_value_type(value)

    def _validate_dtype(self, value):
        """Validates the dtype of the tensor is a float type. Skips check for regular floats."""
        if not isinstance(value, Tensor):
            return  # Regular floats don't have dtype to validate
        if not value.dtype.is_floating_point:
            raise ValueError(
                f"Expected tensor dtype to be a floating-point type, got {value.dtype}."
            )
        if self.dtype and value.dtype != self.dtype:
            raise ValueError(f"Expected dtype {self.dtype}, got {value.dtype}")

    def _validate_read_only(self, value: Union[Tensor, float]):
        """Validates that read-only variables match their default value.

        Handles batched tensors by ensuring ALL values in the batch equal the default.
        """
        if not self.read_only:
            return

        if self.default_value is None:
            raise ValueError(
                f"Variable '{self.name}' is read-only but has no default value."
            )

        # Extract scalar value from default if it's a tensor
        if isinstance(self.default_value, Tensor):
            expected_scalar = self.default_value.item()
        else:
            expected_scalar = self.default_value

        # Compare based on actual value type
        if isinstance(value, Tensor):
            # For batched tensors, check that ALL values equal the default
            # Broadcast expected to match value's shape for comparison
            expected_broadcasted = torch.full_like(value, expected_scalar)
            values_match = torch.allclose(
                value, expected_broadcasted, rtol=1e-9, atol=1e-9
            )
        else:
            # Scalar comparison
            values_match = abs(expected_scalar - value) < 1e-9

        if not values_match:
            raise ValueError(
                f"Variable '{self.name}' is read-only and must equal its default value "
                f"({expected_scalar}), but received {value}."
            )


class TorchNDVariable(NDVariable):
    """Variable for PyTorch tensor data.

    Attributes
    ----------
    default_value : Tensor | None
        Default value for the variable.
    dtype : torch.dtype
        Data type of the tensor. Defaults to torch.float32.

    Notes
    -----
    For image data (when num_channels is set), PyTorch uses (C, H, W) convention
    where C is the number of channels, H is height, and W is width.

    """

    default_value: Optional[Tensor] = None
    dtype: torch.dtype = torch.float32

    @field_validator("dtype", mode="before")
    @classmethod
    def validate_dtype(cls, value):
        """Convert dtype string to torch dtype if needed."""
        if isinstance(value, str):
            dtype_map = {
                "float16": torch.float16,
                "float32": torch.float32,
                "float64": torch.float64,
                "int8": torch.int8,
                "int16": torch.int16,
                "int32": torch.int32,
                "int64": torch.int64,
                "bool": torch.bool,
            }
            if value in dtype_map:
                return dtype_map[value]
            raise ValueError(f"Unknown dtype string: {value}")
        return value

    def _validate_array_type(self, value: Any) -> None:
        """Validates that value is a torch.Tensor."""
        if not isinstance(value, Tensor):
            raise TypeError(
                f"Expected value to be a torch.Tensor, but received {type(value)}."
            )

    def _validate_dtype(self, value: Tensor, expected_dtype: torch.dtype) -> None:
        """Validates the dtype of the tensor."""
        if expected_dtype and value.dtype != expected_dtype:
            raise ValueError(f"Expected dtype {expected_dtype}, got {value.dtype}")

    def _validate_read_only(self, value: Tensor) -> None:
        """Validates that read-only ND variables match their default value.

        Uses element-wise comparison for arrays/images.
        Handles batched tensors by comparing each batch element to the default.
        """
        if not self.read_only:
            return

        if self.default_value is None:
            raise ValueError(
                f"Variable '{self.name}' is read-only but has no default value."
            )

        # Get the expected shape dimensions
        expected_ndim = len(self.shape)

        # Check if value is batched
        if value.ndim > expected_ndim:
            # Batched input - compare each batch item to default
            # Reshape value to (batch_size, -1) and default to (-1)
            _ = value.shape[:-expected_ndim] if expected_ndim > 0 else value.shape[:-1]
            value_flat = value.reshape(-1, *self.shape)

            # Check all batch items against default
            for i in range(value_flat.shape[0]):
                if not torch.allclose(
                    value_flat[i], self.default_value, rtol=1e-9, atol=1e-9
                ):
                    raise ValueError(
                        f"Variable '{self.name}' is read-only and must equal its default value, "
                        f"but received different array values in batch element {i}."
                    )
        else:
            # Single input - direct comparison
            if not torch.allclose(value, self.default_value, rtol=1e-9, atol=1e-9):
                raise ValueError(
                    f"Variable '{self.name}' is read-only and must equal its default value, "
                    f"but received different array values."
                )

    def _get_image_shape_for_validation(self, value: Tensor) -> Tuple[int, ...]:
        """Returns image shape for PyTorch (C, H, W) format."""
        expected_ndim = len(self.shape)
        image_shape = value.shape[-expected_ndim:]

        # Validate channel count for PyTorch (C, H, W)
        if len(image_shape) == 2 and self.num_channels != 1:
            raise ValueError(
                f"Expected 1 channel for grayscale image, got {self.num_channels}."
            )
        elif len(image_shape) == 3 and image_shape[0] != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels, got {image_shape[0]}."
            )

        return image_shape

    @model_validator(mode="after")
    def validate_default_value(self):
        if self.default_value is not None:
            self._validate_array_type(self.default_value)
            self._validate_shape(self.default_value, expected_shape=self.shape)
            self._validate_dtype(self.default_value, self.dtype)
            if self.is_image:
                self._validate_image_shape(self.default_value)
        return self

    def validate_value(self, value: Tensor, config: str = None):
        super().validate_value(value, config)
        self._validate_dtype(value, self.dtype)
        self._validate_read_only(value)


# Alias TorchScalarVariable as ScalarVariable for backwards compatibility
ScalarVariable = TorchScalarVariable


def get_variable(name: str) -> Type[Variable]:
    """Returns the Variable subclass with the given name.

    Parameters
    ----------
    name : str
        Name of the Variable subclass.

    Returns
    -------
    Type[Variable]
        Variable subclass with the given name.

    """
    classes = [ScalarVariable, DistributionVariable, TorchNDVariable]
    class_lookup = {c.__name__: c for c in classes}
    # Also allow "ScalarVariable" to map to TorchScalarVariable
    class_lookup["ScalarVariable"] = ScalarVariable
    if name not in class_lookup.keys():
        logger.error(
            f"Unknown variable type '{name}', valid names are {list(class_lookup.keys())}"
        )
        raise KeyError(
            f"No variable named {name}, valid names are {list(class_lookup.keys())}"
        )
    return class_lookup[name]
