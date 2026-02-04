"""
This module contains definitions of LUME-model variables for use with lume tools.
Variables are designed as pure descriptors and thus aren't intended to hold actual values,
but they can be used to validate encountered values.
"""

import logging
from typing import Any, Optional, Tuple, Type

import torch
from torch import Tensor
from torch.distributions import Distribution as TDistribution
from pydantic import field_validator, model_validator

from lume.variables import Variable, ScalarVariable, NDVariable, ConfigEnum

logger = logging.getLogger(__name__)

# Re-export base classes for backward compatibility and clean API
__all__ = [
    "Variable",
    "ScalarVariable",
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
    if name not in class_lookup.keys():
        logger.error(
            f"Unknown variable type '{name}', valid names are {list(class_lookup.keys())}"
        )
        raise KeyError(
            f"No variable named {name}, valid names are {list(class_lookup.keys())}"
        )
    return class_lookup[name]
