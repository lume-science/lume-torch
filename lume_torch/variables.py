"""
This module contains definitions of LUME-model variables for use with lume tools.
Variables are designed as pure descriptors and thus aren't intended to hold actual values,
but they can be used to validate encountered values.
"""

import logging
import warnings
from typing import Optional, Type, Union, ClassVar, List, Any

import torch
from torch import Tensor
from torch.distributions import Distribution as TDistribution
from pydantic import (
    field_validator,
    model_validator,
    ConfigDict,
    Field,
    field_serializer,
)

from lume.variables import Variable, ScalarVariable, NDVariable, ConfigEnum

logger = logging.getLogger(__name__)

# Rename base ScalarVariable for internal use
_BaseScalarVariable = ScalarVariable

# Re-export base classes for backward compatibility and clean API
# We alias TorchScalarVariable (defined below) as ScalarVariable for backwards compatibility
# with a deprecation warning. See end of this module for the aliasing.
# NOTE: ScalarVariable will be deprecated in the next release - use TorchScalarVariable instead.
__all__ = [
    "Variable",
    "ScalarVariable",
    "TorchScalarVariable",
    "NDVariable",
    "TorchNDVariable",
    "ConfigEnum",
    "DistributionVariable",
    "get_variable",
]

# Canonical map from string name → torch.dtype.
# Keys are what serialize_dtype produces (str(dtype).replace("torch.", "")).
# Aliases (e.g. "double" → float64) intentionally omitted so the map is
# bijective and round-trips cleanly.
_STR_TO_TORCH_DTYPE: dict[str, "torch.dtype"] = {
    str(d).replace("torch.", ""): d
    for d in [
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
        torch.bool,
        torch.complex64,
        torch.complex128,
    ]
}


def _normalize_legacy_read_only(data: Any) -> Any:
    """Map legacy `is_constant` input to canonical `read_only`."""
    if not isinstance(data, dict):
        return data

    normalized = dict(data)
    if "is_constant" in normalized:
        warnings.warn(
            "The 'is_constant' attribute is deprecated and will be removed in a future release. "
            "Use 'read_only' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        if "read_only" not in normalized:
            normalized["read_only"] = normalized["is_constant"]
    # Drop legacy key so stricter extra-field policies won't reject it.
    normalized.pop("is_constant", None)
    return normalized


class DistributionVariable(Variable):
    """Variable for distributions. Must be a subclass of torch.distributions.Distribution.

    Attributes
    ----------
    unit : str, optional
        Unit associated with the variable.

    """

    unit: Optional[str] = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _compat_is_constant(cls, data: Any) -> Any:
        data = _normalize_legacy_read_only(data)
        if isinstance(data, dict):
            data = {k: v for k, v in data.items() if k != "variable_class"}
        return data

    def validate_value(
        self, value: TDistribution, config: Optional[ConfigEnum] = None, **kwargs
    ):
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
        # mandatory validation
        self._validate_value_type(value)

        # optional validation
        config = self._validation_config_as_enum(config)
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

    This class also overrides validation to handle the tensor cases for batched and unbatched cases,
    including optional range checks and read-only checks that compare tensor values with tolerance.

    Attributes
    ----------
    default_value : Tensor | None
        Default value for the variable (must be 0D or 1D with size 1).
    dtype : torch.dtype | None
        Optional data type of the tensor. If specified, validates that tensor values
        match this exact dtype. If None (default), only validates that the dtype is
        a floating-point type without enforcing a specific precision.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    default_value: Optional[Union[Tensor, float]] = None
    dtype: Optional[torch.dtype] = None

    @model_validator(mode="before")
    @classmethod
    def _compat_is_constant(cls, data: Any) -> Any:
        data = _normalize_legacy_read_only(data)
        if isinstance(data, dict):
            data = {k: v for k, v in data.items() if k != "variable_class"}
        return data

    @field_serializer("default_value")
    def serialize_default_value(
        self, value: Optional[Union[Tensor, float]]
    ) -> Optional[float]:
        """Serialize default_value to a Python float when it is a torch.Tensor.

        Parameters
        ----------
        value : Tensor or float or None

        Returns
        -------
        float or None

        """
        if isinstance(value, Tensor):
            return value.item()
        return value

    @field_serializer("dtype")
    def serialize_dtype(self, value: Optional[torch.dtype]) -> Optional[str]:
        """Serialize torch.dtype to its string representation.

        Parameters
        ----------
        value : torch.dtype or None

        Returns
        -------
        str or None
            Dtype string, e.g. ``"torch.float32"``, or ``None``.

        """
        if value is None:
            return None
        return str(value)

    @field_validator("dtype", mode="before")
    @classmethod
    def validate_dtype(cls, value):
        """Validate (and coerce) dtype to a torch.dtype floating-point type.

        Accepts either a ``torch.dtype`` instance or a string (e.g. ``"float32"``
        or ``"torch.float32"``) so that JSON round-trips work correctly.
        """
        if value is None:
            return None

        # Coerce strings produced by serialize_dtype back to torch.dtype
        if isinstance(value, str):
            key = value.replace("torch.", "")
            if key not in _STR_TO_TORCH_DTYPE:
                raise TypeError(
                    f"dtype must be a known torch.dtype string, "
                    f"cannot coerce {value!r}. "
                    f"Known names: {sorted(_STR_TO_TORCH_DTYPE)}"
                )
            value = _STR_TO_TORCH_DTYPE[key]

        # Validate that value is a torch.dtype instance
        if not isinstance(value, torch.dtype):
            raise TypeError(
                f"dtype must be a torch.dtype instance, "
                f"got {type(value).__name__}. "
                f"Received value: {repr(value)}"
            )

        # Validate that the dtype is a floating-point type
        if not value.is_floating_point:
            raise ValueError(f"dtype must be a floating-point type, got {value}")
        return value

    @model_validator(mode="after")
    def validate_default_value(self):
        if self.default_value is not None:
            self.validate_value(self.default_value, ConfigEnum.ERROR)
        return self

    def _unbatch(self, value: Tensor) -> Tensor:
        """Unbatch a tensor with a batch dimension."""
        if value.ndim == 0:
            return (value,)
        elif value.ndim > 0 and value.shape[-1] == 1:
            return value.flatten()
        else:
            raise ValueError(
                f"Expected a 0D scalar tensor or a 1D tensor with a single element in the "
                f"last dimension, but got {value.ndim} dimensions with shape {value.shape}. "
            )

    def _validate_value_type(self, value):
        """Validates that value is a 0D/1D torch.Tensor or a regular float/int."""
        if isinstance(value, Tensor):
            if value.ndim != 0 and not (value.ndim == 1 and value.shape[0] == 1):
                raise ValueError(
                    f"Expected a 0D scalar tensor or a 1D tensor with a single element, "
                    f"but got {value.ndim} dimensions with shape {value.shape}. "
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

    def validate_value(
        self, value: Union[Tensor, float], config: Optional[ConfigEnum] = None
    ):
        """Validates the given tensor or float value, handling batches if present.

        Parameters
        ----------
        value : Tensor or float
            The value to be validated. Can be a 0D scalar tensor, a 1
            tensor with a single or batched elements, or a regular float.
        config : ConfigEnum, optional
            The configuration for validation. Defaults to None.
            Allowed values are "none", "warn", and "error". If not "none",
            performs optional range checks on the scalar values.

        Raises
        ------
        ValueError
            If the value fails type/dtype validation or range checks based on config.

        """
        # Unbatch if appropriate and get samples
        samples = self._unbatch(value) if isinstance(value, Tensor) else (value,)

        # Validate dtype/type strictly on first sample
        self._validate_value_type(samples[0])
        self._validate_dtype(samples[0])

        # Optional range validation for all samples if config != NULL
        config_enum = self._validation_config_as_enum(config)
        if config_enum != ConfigEnum.NULL:
            for sample in samples:
                scalar_value = sample.item() if isinstance(sample, Tensor) else sample
                self._validate_value_is_within_range(scalar_value, config=config_enum)

    def validate_read_only(
        self,
        value: Union[Tensor, float],
        config: Optional[ConfigEnum] = None,
        rtol: float = 1e-9,
        atol: float = 1e-9,
    ) -> list:
        """Validate that all (possibly batched) values equal the default value (with tolerance).
        Returns a list of indices where the check fails. Follows config for error/warn/none.

        Parameters
        ----------
        value : Tensor or float
            The value to be validated. Can be a 0D scalar tensor, a 1
            tensor with a single or batched elements, or a regular float.
        config : ConfigEnum, optional
            The configuration for validation. Defaults to None.
            Allowed values are "none", "warn", and "error".
        rtol : float, optional
            Relative tolerance for comparison. Defaults to 1e-9.
        atol : float, optional
            Absolute tolerance for comparison. Defaults to 1e-9.

        Returns
        -------
        list
            A list of indices (flattened) where the value does not match the default value within
            the specified tolerances.

        Raises
        ------
        ValueError
            If the variable is read-only but has no default value, or if config is invalid.

        """
        if self.default_value is None:
            raise ValueError(
                f"Variable '{self.name}' is read-only but has no default value."
            )

        # Unbatch if appropriate and get samples
        samples = self._unbatch(value) if isinstance(value, Tensor) else (value,)

        # Get expected scalar
        if not isinstance(self.default_value, Tensor):
            expected = torch.tensor(self.default_value)
        else:
            expected = self.default_value

        failed = []
        for idx, sample in enumerate(samples):
            actual = sample if isinstance(sample, Tensor) else torch.tensor(sample)
            if not torch.isclose(actual, expected, rtol=rtol, atol=atol):
                failed.append(idx)

        config_enum = self._validation_config_as_enum(config)
        if failed:
            msg = (
                f"Variable '{self.name}' is read-only and must equal its default value "
                f"({expected}), but found {len(failed)} mismatches at indices (flattened) {failed}."
            )
            if config_enum == ConfigEnum.ERROR:
                raise ValueError(msg)
            elif config_enum == ConfigEnum.WARN:
                warnings.warn(msg)
        return failed


class TorchNDVariable(NDVariable):  # noqa: E303
    """Variable for PyTorch tensor data with arbitrary shape and dtype.

    Attributes
    ----------
    default_value : Tensor | None
        Default value for the variable. Must match the expected
        shape and dtype if provided. Defaults to None.
    dtype : torch.dtype
        Expected element type. Defaults to torch.float64.
    default_value : torch.Tensor | None
        Optional default tensor. Validated against shape and dtype.

    Examples
    --------
    >>> import torch
    >>> from lume_torch.variables import TorchNDVariable
    >>>
    >>> var = TorchNDVariable(
    ...     name="my_tensor",
    ...     shape=(3, 4),
    ...     dtype=torch.float32
    ... )
    >>>
    >>> tensor = torch.rand(3, 4)
    >>> var.validate_value(tensor, config="error")  # Passes

    """

    default_value: Optional[Tensor] = None
    dtype: Union[torch.dtype | str] = torch.float32

    array_type: ClassVar[type] = Tensor

    # Internal flag: set to True when default_value was coerced from a
    # list so validate_default_value knows to cast rather than reject.
    # Excluded from serialization; populated via _mark_coerced_default_value.
    default_value_was_coerced: bool = Field(default=False, exclude=True, repr=False)

    @classmethod
    def _dtype_coerce(cls, value: Any) -> "torch.dtype":
        """Coerce a string (e.g. "float64") to the matching torch.dtype.

        Parameters
        ----------
        value : Any
            Typically a string produced by serialize_dtype.

        Returns
        -------
        torch.dtype
            The corresponding PyTorch dtype.

        Raises
        ------
        TypeError
            If value is not a string or does not map to a known dtype.

        """
        if not isinstance(value, str):
            raise TypeError(
                f"Cannot coerce {type(value).__name__!r} to torch.dtype; "
                "expected a string."
            )
        key = value.replace("torch.", "")  # tolerate "torch.float64" too
        try:
            return _STR_TO_TORCH_DTYPE[key]
        except KeyError:
            raise TypeError(
                f"Cannot coerce {value!r} to torch.dtype. "
                f"Known names: {sorted(_STR_TO_TORCH_DTYPE)}"
            )

    @field_serializer("dtype")
    def serialize_dtype(self, value: "torch.dtype") -> str:
        """Serialize torch.dtype to its full string representation.

        Parameters
        ----------
        value : torch.dtype
            The dtype to serialize.

        Returns
        -------
        str
            Dtype string, e.g. "torch.float32", "torch.int64".

        """
        return str(value)

    @field_serializer("default_value")
    def serialize_default_value(
        self, value: Optional["torch.Tensor"]
    ) -> Optional[List]:
        """Serialize a torch.Tensor to a nested Python list.

        Parameters
        ----------
        value : torch.Tensor or None

        Returns
        -------
        list or None

        """
        if value is None:
            return None
        return value.tolist()

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="forbid"
    )

    @model_validator(mode="before")
    @classmethod
    def _mark_coerced_default_value(cls, data: Any) -> Any:
        """Detect list default_value input before field validation.

        When default_value arrives as a list, tuple (e.g.
        after a JSON round-trip), Pydantic cannot infer the intended
        torch.dtype. This validator sets a _default_value_was_coerced
        key in the raw dict so that validate_default_value knows to
        cast the resulting tensor to self.dtype rather than reject the
        dtype mismatch.

        """
        data = _normalize_legacy_read_only(data)
        if isinstance(data, dict):
            data = {k: v for k, v in data.items() if k != "variable_class"}
            dv = data.get("default_value")
            if isinstance(dv, (list, tuple)):
                data = dict(data)
                data["default_value_was_coerced"] = True
        return data

    @field_validator("default_value", mode="before")
    @classmethod
    def coerce_default_value(cls, value: Any) -> Any:
        """Coerce a list/tuple back to a torch.Tensor on round-trip.

        The tensor is created with PyTorch's inferred dtype; the correct dtype
        is applied later in validate_default_value once self.dtype is known.

        Parameters
        ----------
        value : Any
            Raw input. Lists and tuples are converted via torch.tensor(value).

        Returns
        -------
        torch.Tensor or None or Any

        """
        if isinstance(value, (list, tuple)):
            return torch.tensor(value)
        return value

    @model_validator(mode="after")
    def validate_default_value(self):
        """Cast or validate default_value, then run base shape/type checks.

        * If default_value was coerced from a list (flagged by
          _mark_coerced_default_value), it is cast to self.dtype so the
          round-trip works without the caller having to specify the dtype twice.
        * If it was supplied as an explicit torch.Tensor with a mismatched
          dtype, the base-class validator raises ValueError as expected.

        Returns
        -------
        TorchNDVariable
            The validated instance.

        """
        if (
            self.default_value is not None
            and self.default_value_was_coerced
            and self.default_value.dtype != self.dtype
        ):
            object.__setattr__(self, "default_value", self.default_value.to(self.dtype))
        # Reset flag for subsequent validate_assignment calls
        object.__setattr__(self, "default_value_was_coerced", False)
        # Delegate shape / type checks to base
        return super().validate_default_value()

    def _unbatch(self, value: Tensor) -> Tensor:
        """Unbatch a tensor with a batch dimension."""
        if value.ndim < len(self.shape):
            raise ValueError(
                f"Expected tensor with at least {len(self.shape)} dimensions, "
                f"but got {value.ndim} dimensions with shape {value.shape}."
            )
        elif value.ndim == len(self.shape):
            return (value,)
        else:
            actual_shape = value.shape[-len(self.shape) :]
            if actual_shape != self.shape:
                raise ValueError(
                    f"Expected shape {self.shape}, got {actual_shape} "
                    f"from value with shape {value.shape}."
                )
            return value.reshape(-1, *self.shape)

    def validate_value(self, value: Tensor, config: Optional[ConfigEnum] = None):
        """Validates the given tensor value, handling batches if present.

        Parameters
        ----------
        value : Tensor
            The tensor value to be validated. Can have a batch dimension as long as the last dimensions
            match self.shape.
        config : ConfigEnum, optional
            The configuration for validation. Defaults to None.
            Allowed values are "none", "warn", and "error". If not "none",
            performs optional range checks on the tensor values.

        Raises
        ------
        ValueError
            If the value fails type/dtype/shape validation or range checks based on config.

        """
        # Validate type and dtype on full tensor
        self._validate_array_type(value)
        self._validate_dtype(value, self.dtype)

        # Unbatch if tensor with batch dimension
        samples = self._unbatch(value)

        # Validate shape strictly on first sample
        self._validate_shape(samples[0], expected_shape=self.shape)

        # No optional range validation for NDVariable (as in parent)

    def validate_read_only(
        self,
        value: Tensor,
        config: Optional[ConfigEnum] = None,
        rtol: float = 1e-9,
        atol: float = 1e-9,
    ) -> list:
        """Validate that all (possibly batched) tensors equal the default value (with tolerance).
        Returns a list of failed indices. Follows config for error/warn/none.

        Note that this expects the value to have already passed validate_value, so it assumes the
        shape and dtype are correct and does not re-check those.

        Parameters
        ----------
        value : Tensor
            The tensor value to be validated. Can have a batch dimension as long as the last dimensions
            match self.shape.
        config : ConfigEnum, optional
            The configuration for validation. Defaults to None.
            Allowed values are "none", "warn", and "error".
        rtol : float, optional
            Relative tolerance for comparison. Defaults to 1e-9.
        atol : float, optional
            Absolute tolerance for comparison. Defaults to 1e-9.

        Returns
        -------
        list
            A list of indices (flattened) where the value does not match the default value within
            the specified tolerances.

        Raises
        ------
        ValueError
            If the variable is read-only but has no default value, or if config is invalid.

        """
        if self.default_value is None:
            raise ValueError(
                f"Variable '{self.name}' is read-only but has no default value."
            )

        # Unbatch if tensor with batch dimension
        samples = self._unbatch(value)

        failed = []
        for idx, sample in enumerate(samples):
            if not torch.allclose(sample, self.default_value, rtol=rtol, atol=atol):
                failed.append(idx)

        config_enum = self._validation_config_as_enum(config)
        if failed:
            msg = (
                f"Variable '{self.name}' is read-only and must equal its default value, "
                f"but found {len(failed)} mismatches at indices {failed}."
            )
            if config_enum == ConfigEnum.ERROR:
                raise ValueError(msg)
            elif config_enum == ConfigEnum.WARN:
                warnings.warn(msg)
        return failed


# Alias TorchScalarVariable as ScalarVariable for backwards compatibility
# This will be deprecated in the next release
class ScalarVariable(TorchScalarVariable):
    """Deprecated alias for TorchScalarVariable.

    .. deprecated::
        ScalarVariable is deprecated and will be removed in the next release.
        Use TorchScalarVariable instead.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ScalarVariable is deprecated and will be removed in the next release. "
            "Please use TorchScalarVariable instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


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
    classes = [
        TorchScalarVariable,
        ScalarVariable,
        DistributionVariable,
        TorchNDVariable,
    ]
    class_lookup = {c.__name__: c for c in classes}
    if name not in class_lookup.keys():
        logger.error(
            f"Unknown variable type '{name}', valid names are {list(class_lookup.keys())}"
        )
        raise KeyError(
            f"No variable named {name}, valid names are {list(class_lookup.keys())}"
        )
    return class_lookup[name]
