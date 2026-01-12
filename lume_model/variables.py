"""
This module contains definitions of LUME-model variables for use with lume tools.
Variables are designed as pure descriptors and thus aren't intended to hold actual values,
but they can be used to validate encountered values.

For now, only scalar floating-point variables are supported.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Type, Tuple, Literal, Callable, Union
from enum import Enum
import math

import numpy as np
from numpy.typing import NDArray
import torch
from torch.distributions import Distribution as TDistribution
from pydantic import BaseModel, field_validator, model_validator, ConfigDict
from pydantic_core import core_schema


class ConfigEnum(str, Enum):
    """Enum for configuration options during validation."""

    NULL = "none"
    WARN = "warn"
    ERROR = "error"


class Variable(BaseModel, ABC):
    """Abstract variable base class.

    Attributes:
        name: Name of the variable.
    """

    name: str

    @property
    @abstractmethod
    def default_validation_config(self) -> ConfigEnum:
        """Determines default behavior during validation."""
        return None

    @abstractmethod
    def validate_value(self, value: Any, config: dict[str, bool] = None):
        pass

    def model_dump(self, **kwargs) -> dict[str, Any]:
        config = super().model_dump(**kwargs)
        return {"variable_class": self.__class__.__name__} | config


class ScalarVariable(Variable):
    """Variable for float values.

    Attributes:
        default_value: Default value for the variable. Note that the LUMEBaseModel requires this
          for input variables, but it is optional for output variables.
        value_range: Value range that is considered valid for the variable. If the value range is set to None,
          the variable is interpreted as a constant and values are validated against the default value.
        is_constant: Flag indicating whether the variable is constant.
        unit: Unit associated with the variable.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    default_value: Optional[float] = None
    is_constant: Optional[bool] = False
    value_range: Optional[tuple[float, float]] = None
    # tolerance for floating point errors, currently only used for constant variables
    value_range_tolerance: Optional[float] = 1e-8
    unit: Optional[str] = None

    @field_validator("value_range", mode="before")
    @classmethod
    def validate_value_range(cls, value):
        if value is not None:
            value = tuple(value)
            if not value[0] <= value[1]:
                raise ValueError(
                    f"Minimum value ({value[0]}) must be lower or equal than maximum ({value[1]})."
                )
        return value

    @model_validator(mode="after")
    def validate_default_value(self):
        if self.default_value is not None and self.value_range is not None:
            if not self._value_is_within_range(self.default_value):
                raise ValueError(
                    "Default value ({}) is out of valid range: ([{},{}]).".format(
                        self.default_value, *self.value_range
                    )
                )
        return self

    @model_validator(mode="after")
    def validate_constant_value_range(self):
        if self.is_constant and self.value_range is not None:
            # if the upper limit is not equal to the lower limit, raise an error
            if not self.value_range[0] == self.value_range[1]:
                error_message = (
                    f"Expected range to be constant for constant variable '{self.name}', "
                    f"but received a range of values. Set range to None or set the "
                    f"upper limit equal to the lower limit."
                )
                raise ValueError(error_message)
        return self

    @property
    def default_validation_config(self) -> ConfigEnum:
        return "warn"

    def validate_value(self, value: float, config: ConfigEnum = None):
        """
        Validates the given value.

        Args:
            value (float): The value to be validated.
            config (ConfigEnum, optional): The configuration for validation. Defaults to None.
              Allowed values are "none", "warn", and "error".

        Raises:
            TypeError: If the value is not of type float.
            ValueError: If the value is out of the valid range or does not match the default value
              for constant variables.
        """
        _config = self.default_validation_config if config is None else config
        # mandatory validation
        self._validate_value_type(value)
        value = self._validate_shape(value)
        # optional validation for each scalar / element in the array
        if config != "none":
            for v in value:
                self._validate_value_is_within_range(v, config=_config)

    @staticmethod
    def _validate_value_type(value: Union[float, np.ndarray, torch.Tensor]):
        if isinstance(value, (np.ndarray, torch.Tensor)):
            if isinstance(value, np.ndarray):
                if not np.issubdtype(value.dtype, np.floating):
                    raise TypeError(
                        f"Expected value to be of type {np.floating}, "
                        f"but received {value.dtype}."
                    )
            elif isinstance(value, torch.Tensor):
                if not torch.is_floating_point(value):
                    raise TypeError(
                        f"Expected value to be of type {torch.float64}, {torch.float32}, {torch.float16}, "
                        f"or {torch.bfloat16}, but received {value.dtype}."
                    )
        elif isinstance(value, float):
            pass
        else:
            raise TypeError(
                f"Expected value to be of type {float}, a torch tensor of floats, or a numpy "
                f"array of floats, but received {type(value)}."
            )

    @staticmethod
    def _validate_shape(value: Union[np.ndarray, torch.Tensor]):
        """Validates that the last dimension of the array is 0 or 1.

        Args:
            value: Numpy array or torch tensor to validate.
        """
        if isinstance(value, (np.ndarray, torch.Tensor)):
            if (
                value.ndim == 0
                or value.ndim == 1
                or (value.ndim > 1 and value.shape[-1] in (0, 1))
            ):
                # itemize and validate each element in the array
                return [v.item() if hasattr(v, "item") else v for v in value.flatten()]
            else:
                raise ValueError(
                    f"Expected scalar to have ndim 0 or 1, or array variable to have the last dimension "
                    f"be 0 or 1 for ScalarVariable type, but received {value.shape[-1]}."
                )
        else:  # float
            return [value]

    def _validate_value_is_within_range(self, value: float, config: ConfigEnum = None):
        if not self._value_is_within_range(value):
            if self.is_constant:
                error_message = "Expected value to be ({}) for constant variable ({}), but received ({}).".format(
                    self.default_value, self.name, value
                )
            else:
                error_message = (
                    "Value ({}) of '{}' is out of valid range: ([{},{}]).".format(
                        value, self.name, *self.value_range
                    )
                )
            range_warning_message = (
                error_message
                + " Executing the model outside of the training data range may result in"
                " unpredictable and invalid predictions."
            )
            if config == "warn":
                print("Warning: " + range_warning_message)
            else:
                raise ValueError(error_message)

    def _value_is_within_range(self, value) -> bool:
        self.value_range = self.value_range or (-np.inf, np.inf)
        tolerances = {"rel_tol": 0, "abs_tol": self.value_range_tolerance}
        is_within_range, is_within_tolerance = False, False
        # constant variables
        if self.value_range is None or self.is_constant:
            if self.default_value is None:
                is_within_tolerance = True
            else:
                is_within_tolerance = math.isclose(
                    value, self.default_value, **tolerances
                )
        # non-constant variables
        else:
            is_within_range = self.value_range[0] <= value <= self.value_range[1]
            is_within_tolerance = any(
                [math.isclose(value, ele, **tolerances) for ele in self.value_range]
            )
        return is_within_range or is_within_tolerance


class DistributionVariable(Variable):
    """Variable for distributions. Must be a subclass of torch.distributions.Distribution.

    Attributes:
        unit: Unit associated with the variable.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    unit: Optional[str] = None

    @property
    def default_validation_config(self) -> ConfigEnum:
        return "none"

    def validate_value(self, value: TDistribution, config: ConfigEnum = None):
        """
        Validates the given value.

        Args:
            value (Distribution): The value to be validated.
            config (ConfigEnum, optional): The configuration for validation. Defaults to None.
              Allowed values are "none", "warn", and "error".

        Raises:
            TypeError: If the value is not an instance of Distribution.
        """
        _config = self.default_validation_config if config is None else config
        # mandatory validation
        self._validate_value_type(value)
        # optional validation
        if config != "none":
            pass  # not implemented

    @staticmethod
    def _validate_value_type(value: TDistribution):
        if not isinstance(value, TDistribution):
            raise TypeError(
                f"Expected value to be of type {TDistribution}, "
                f"but received {type(value)}."
            )


class ArrayValidatorMixin:
    """Utility class providing common validation methods for array types.

    Provides shared validation logic for shape, dtype, and type checking.
    """

    @staticmethod
    def _validate_type(value: Any, base_type: Type, type_name: str) -> None:
        """Validates that value is of the expected base type."""
        if not isinstance(value, base_type):
            raise TypeError(f"Value must be a {type_name}, got {type(value)}")

    @staticmethod
    def _validate_shape(value: Any, expected_shape: Tuple[int, ...] = None) -> None:
        """Validates that value has the expected shape."""
        if expected_shape and tuple(value.shape) != expected_shape:
            raise ValueError(
                f"Expected shape {expected_shape}, got {tuple(value.shape)}"
            )

    @staticmethod
    def _validate_dtype(value: Any, expected_dtype: Any = None) -> None:
        """Validates that value has the expected dtype."""
        if expected_dtype and value.dtype != expected_dtype:
            raise ValueError(f"Expected dtype {expected_dtype}, got {value.dtype}")


class NumpyNDArray(np.ndarray):
    """Custom Pydantic-compatible type for numpy.ndarray."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: Callable[[Any], core_schema.CoreSchema]
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(cls.validate)

    @classmethod
    def validate(
        cls,
        value: Any,
        expected_shape: Tuple[int, ...] = None,
        expected_dtype: np.dtype = None,
    ) -> np.ndarray:
        ArrayValidatorMixin._validate_type(value, np.ndarray, "numpy.ndarray")
        ArrayValidatorMixin._validate_shape(value, expected_shape)
        ArrayValidatorMixin._validate_dtype(value, expected_dtype)
        return value


class TorchTensor(torch.Tensor):
    """Custom Pydantic-compatible type for torch.Tensor."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: Callable[[Any], core_schema.CoreSchema]
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(cls.validate)

    @classmethod
    def _validate_device(cls, value: torch.Tensor, expected_device: str = None) -> None:
        """Validates that tensor is on the expected device."""
        if expected_device and value.device.type != expected_device:
            raise ValueError(
                f"Expected device {expected_device}, got {value.device.type}"
            )

    @classmethod
    def validate(
        cls,
        value: Any,
        expected_shape: Tuple[int, ...] = None,
        expected_dtype: torch.dtype = None,
        expected_device: str = None,
    ) -> torch.Tensor:
        ArrayValidatorMixin._validate_type(value, torch.Tensor, "torch.Tensor")
        ArrayValidatorMixin._validate_shape(value, expected_shape)
        ArrayValidatorMixin._validate_dtype(value, expected_dtype)
        cls._validate_device(value, expected_device)
        return value


class ArrayVariable(Variable):
    """
    Variable for array data (NumpyNDArray or TorchTensor).

    Can also represent image data when num_channels is specified, which enforces
    2D or 3D shapes and validates channel dimensions.

    Attributes:
        default_value: Default value for the variable.
        name: Name of the variable.
        shape: Shape of the array.
        dtype: Data type of the array.
        unit: Unit associated with the variable.
        array_type: Type of array, either 'numpy' or 'torch'.
        device: Device for torch.Tensor ('cpu' or 'cuda'), optional.
        num_channels: Number of image channels (1 for grayscale, 3 for RGB).
            When set, enables image-specific validation requiring 2D or 3D shapes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    default_value: Optional[Union[NDArray, torch.Tensor]] = None
    shape: Tuple[int, ...]
    dtype: Any
    unit: Optional[str] = None
    array_type: Literal["numpy", "torch"] = "torch"
    device: Optional[str] = None
    num_channels: Optional[int] = (
        None  # When set, enables image validation (1=grayscale, 3=RGB)
    )

    @property
    def is_image(self) -> bool:
        """Returns True if this variable represents image data."""
        return self.num_channels is not None

    @property
    def default_validation_config(self):
        return "warn"

    @property
    def _validator_class(self) -> Type[ArrayValidatorMixin]:
        """Returns the appropriate validator class based on array_type."""
        return NumpyNDArray if self.array_type == "numpy" else TorchTensor

    @property
    def _expected_base_type(self) -> Type:
        """Returns the expected base type for the array."""
        return np.ndarray if self.array_type == "numpy" else torch.Tensor

    def _validate_array(
        self,
        value: Union[NDArray, torch.Tensor],
        expected_shape: Tuple[int, ...] = None,
        expected_dtype: Any = None,
    ) -> None:
        """Common validation logic for array values.

        Args:
            value: The array value to validate.
            expected_shape: Expected shape of the array.
            expected_dtype: Expected dtype of the array.

        Raises:
            ValueError: If value is not the expected array type, or shape/dtype don't match.
        """
        if not isinstance(value, self._expected_base_type):
            raise ValueError(
                f"Expected value to be a {self._expected_base_type.__name__}, "
                f"but received {type(value)}."
            )

        if self.array_type == "numpy":
            NumpyNDArray.validate(
                value,
                expected_shape=expected_shape,
                expected_dtype=expected_dtype,
            )
        elif self.array_type == "torch":
            TorchTensor.validate(
                value,
                expected_shape=expected_shape,
                expected_dtype=expected_dtype,
                expected_device=self.device,
            )

    def _validate_image_shape(self, value: Union[NDArray, torch.Tensor]) -> None:
        """Validates image-specific shape constraints.

        Args:
            value: The array value to validate.

        Raises:
            ValueError: If shape is not 2D or 3D, or channel count doesn't match.
        """
        if len(self.shape) not in (2, 3):
            raise ValueError(
                f"Image array expects shape to be 2D or 3D, got {self.shape}."
            )
        if len(value.shape) not in (2, 3):
            raise ValueError(
                f"Value for image array must be 2D or 3D, got {value.shape}."
            )

        # Validate channel count
        if self.array_type == "numpy":
            # NumPy images: (H, W) for grayscale or (H, W, C) for color
            if len(value.shape) == 2 and self.num_channels != 1:
                raise ValueError(
                    f"Expected 1 channel for grayscale image, got {self.num_channels}."
                )
            elif len(value.shape) == 3 and value.shape[2] != self.num_channels:
                raise ValueError(
                    f"Expected {self.num_channels} channels, got {value.shape[2]}."
                )
        elif self.array_type == "torch":
            # PyTorch images: (H, W) for grayscale or (C, H, W) for color
            if len(value.shape) == 2 and self.num_channels != 1:
                raise ValueError(
                    f"Expected 1 channel for grayscale image, got {self.num_channels}."
                )
            elif len(value.shape) == 3 and value.shape[0] != self.num_channels:
                raise ValueError(
                    f"Expected {self.num_channels} channels, got {value.shape[0]}."
                )

    @model_validator(mode="after")
    def validate_default_value(self):
        if self.default_value is not None:
            self._validate_array(
                self.default_value,
                expected_shape=self.shape,
                expected_dtype=self.dtype,
            )
            if self.is_image:
                self._validate_image_shape(self.default_value)
        return self

    @model_validator(mode="after")
    def validate_image_shape_config(self):
        """Validates that image configuration has valid 2D or 3D shape."""
        if self.is_image and len(self.shape) not in (2, 3):
            raise ValueError(
                f"Image array expects shape to be 2D or 3D, got {len(self.shape)}D shape {self.shape}."
            )
        return self

    def validate_value(
        self, value: Union[NumpyNDArray, TorchTensor], config: str = None
    ):
        _config = self.default_validation_config if config is None else config
        # mandatory validation
        print(value)
        self._validate_array(
            value,
            expected_shape=self.shape,
            expected_dtype=self.dtype,
        )
        # image-specific validation
        if self.is_image:
            self._validate_image_shape(value)
        # optional validation
        if config != "none":
            pass  # TODO: implement optional validation logic, like range checks, checking for NaNs, etc.


def get_variable(name: str) -> Type[Variable]:
    """Returns the Variable subclass with the given name.

    Args:
        name: Name of the Variable subclass.

    Returns:
        Variable subclass with the given name.
    """
    classes = [ScalarVariable, DistributionVariable, ArrayVariable]
    class_lookup = {c.__name__: c for c in classes}
    if name not in class_lookup.keys():
        raise KeyError(
            f"No variable named {name}, valid names are {list(class_lookup.keys())}"
        )
    return class_lookup[name]
