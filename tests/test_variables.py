"""Test suite for lume_torch.variables module."""

import json
import pytest
import numpy as np
import torch
from pydantic import ValidationError
from torch.distributions import Normal

from lume_torch.variables import (
    ScalarVariable,
    TorchScalarVariable,
    TorchNDVariable,
    DistributionVariable,
    get_variable,
)


class TestGetVariable:
    """Tests for the get_variable function."""

    def test_get_scalar_variable(self):
        """Test getting ScalarVariable by name."""
        var_cls = get_variable("ScalarVariable")
        assert var_cls is ScalarVariable

    def test_get_torch_scalar_variable(self):
        """Test getting TorchScalarVariable by name."""
        var_cls = get_variable("TorchScalarVariable")
        assert var_cls is TorchScalarVariable

    def test_get_distribution_variable(self):
        """Test getting DistributionVariable by name."""
        var_cls = get_variable("DistributionVariable")
        assert var_cls is DistributionVariable

    def test_get_torch_nd_variable(self):
        """Test getting TorchNDVariable by name."""
        var_cls = get_variable("TorchNDVariable")
        assert var_cls is TorchNDVariable

    def test_get_variable_unknown_name_raises(self):
        """Test that unknown variable name raises KeyError."""
        with pytest.raises(KeyError, match="No variable named"):
            get_variable("UnknownVariable")

    def test_get_variable_creates_instance(self):
        """Test that get_variable returns a class that can be instantiated."""
        # Using ScalarVariable will trigger a deprecation warning
        with pytest.warns(DeprecationWarning, match="ScalarVariable is deprecated"):
            var = get_variable(ScalarVariable.__name__)(name="test")
        assert isinstance(var, ScalarVariable)


class TestScalarVariableAlias:
    """Tests to ensure ScalarVariable is a deprecated subclass of TorchScalarVariable."""

    def test_scalar_variable_is_torch_scalar_variable(self):
        """ScalarVariable should be a deprecated subclass of TorchScalarVariable."""
        # ScalarVariable is a subclass, not an alias
        assert issubclass(ScalarVariable, TorchScalarVariable)
        with pytest.warns(DeprecationWarning, match="ScalarVariable is deprecated"):
            var = ScalarVariable(name="test_var", default_value=1.0)
        assert isinstance(var, TorchScalarVariable)


class TestTorchScalarVariable:
    """Tests for TorchScalarVariable class."""

    def test_basic_creation(self):
        """Test basic variable creation with minimal parameters."""
        var = TorchScalarVariable(name="test_var")
        assert var.name == "test_var"
        assert var.default_value is None
        assert var.value_range is None
        assert var.read_only is False

    def test_creation_with_all_attributes(self):
        """Test variable creation with all attributes."""
        var = TorchScalarVariable(
            name="full_var",
            default_value=torch.tensor(1.5),
            value_range=(0.0, 10.0),
            unit="meters",
            read_only=True,
            dtype=torch.float32,
        )
        assert var.name == "full_var"
        assert torch.isclose(var.default_value, torch.tensor(1.5))
        assert var.value_range == (0.0, 10.0)
        assert var.unit == "meters"
        assert var.read_only is True
        assert var.dtype == torch.float32

    def test_creation_with_mismatched_dtype_raises(self):
        """Test variable creation with all attributes."""
        with pytest.raises(ValidationError):
            TorchScalarVariable(
                name="full_var",
                default_value=torch.tensor(1.5, dtype=torch.float32),
                value_range=(0.0, 10.0),
                unit="meters",
                read_only=True,
                dtype=torch.float64,
            )

    def test_default_value_as_float(self):
        """Test that float default values are accepted."""
        var = TorchScalarVariable(name="float_var", default_value=2.5)
        assert var.default_value == 2.5

    def test_default_value_as_tensor(self):
        """Test that tensor default values are accepted."""
        var = TorchScalarVariable(name="tensor_var", default_value=torch.tensor(3.0))
        assert torch.isclose(var.default_value, torch.tensor(3.0))

    def test_default_value_out_of_range_raises(self):
        """Test that default value out of range raises ValueError."""
        with pytest.raises(ValueError, match="out of valid range"):
            TorchScalarVariable(
                name="bad_var", default_value=15.0, value_range=(0.0, 10.0)
            )

    # Dtype validation tests
    def test_dtype_float64(self):
        """Test dtype with torch.float64."""
        var = TorchScalarVariable(name="test", dtype=torch.float64)
        assert var.dtype == torch.float64

    def test_dtype_double(self):
        """Test dtype with torch.double."""
        var = TorchScalarVariable(name="test", dtype=torch.double)
        assert var.dtype == torch.double

    def test_dtype_invalid_type_raises(self):
        """Test that invalid dtype type raises TypeError."""
        with pytest.raises(TypeError, match="dtype must be a"):
            TorchScalarVariable(name="test", dtype="invalid_dtype")

    def test_dtype_non_floating_raises(self):
        """Test that non-floating dtype raises ValueError."""
        with pytest.raises(ValueError, match="must be a floating-point type"):
            TorchScalarVariable(name="test", dtype=torch.int32)

    # Value validation tests
    def test_validate_value_float(self):
        """Test validation of float values."""
        var = TorchScalarVariable(name="test")
        var.validate_value(5.0)

    def test_validate_value_int(self):
        """Test validation of int values (allowed as scalars)."""
        var = TorchScalarVariable(name="test")
        var.validate_value(5)

    def test_validate_value_tensor_0d(self):
        """Test validation of 0D tensor."""
        var = TorchScalarVariable(name="test")
        var.validate_value(torch.tensor(5.0))

    def test_validate_value_tensor_1d(self):
        """Test that 1D scalar tensors are accepted."""
        var = TorchScalarVariable(name="test")
        var.validate_value(torch.tensor([5.0]))

    def test_validate_value_tensor_1d_batched(self):
        """Test that batched tensor with last dim = 1 is accepted."""
        var = TorchScalarVariable(name="test")
        var.validate_value(torch.tensor([[5.0], [6.0]]))

    def test_validate_value_tensor_nd_rejects_non_scalar(self):
        """Test that tensors with more than 1 element are rejected."""
        var = TorchScalarVariable(name="test")
        with pytest.raises(ValueError, match="Expected a 0D scalar"):
            var.validate_value(torch.tensor([5.0, 6.0]))

    def test_validate_batched_checks_every_sample_error(self):
        """Batched scalar validation should apply value-range checks to every sample."""
        var = TorchScalarVariable(name="test", value_range=(0.0, 10.0))
        with pytest.raises(ValueError, match="out of valid range"):
            var.validate_value(torch.tensor([[1.0], [99.0]]), config="error")

    def test_validate_value_tensor_invalid_shape_batched(self):
        """Test that batched tensors with singleton trailing dimension are accepted."""
        var = TorchScalarVariable(name="test")
        var.validate_value(torch.tensor([[[5.0]], [[6.0]], [[7.0]]]))

    def test_validate_value_tensor_higher_rank_batched(self):
        """Test that additional batched scalar layouts are accepted."""
        var = TorchScalarVariable(name="test")
        var.validate_value(torch.tensor([[[5.0], [6.0]], [[7.0], [8.0]]]))

    def test_validate_value_invalid_type_raises(self):
        """Test that invalid value type raises TypeError."""
        var = TorchScalarVariable(name="test")
        with pytest.raises(TypeError):
            var.validate_value("not_a_number")

    def test_validate_value_non_float_tensor_raises(self):
        """Test that non-float 0D tensor raises ValueError."""
        var = TorchScalarVariable(name="test")
        with pytest.raises(ValueError, match="floating-point type"):
            var.validate_value(torch.tensor(1))  # int64 0D tensor

    def test_validate_value_wrong_dtype_raises(self):
        """Test that tensor with wrong dtype raises ValueError."""
        var = TorchScalarVariable(name="test", dtype=torch.float64)
        with pytest.raises(ValueError, match="Expected dtype"):
            var.validate_value(torch.tensor(5.0, dtype=torch.float32))

    # Value range validation tests
    def test_validate_value_in_range(self):
        """Test validation of value within range."""
        var = TorchScalarVariable(
            name="test", value_range=(0.0, 10.0), default_validation_config="error"
        )
        var.validate_value(5.0)

    def test_validate_value_out_of_range_error(self):
        """Test that out-of-range value raises error with error config."""
        var = TorchScalarVariable(
            name="test", value_range=(0.0, 10.0), default_validation_config="error"
        )
        with pytest.raises(ValueError, match="out of valid range"):
            var.validate_value(15.0)

    def test_validate_value_out_of_range_warn(self):
        """Test that out-of-range value warns with warn config."""
        var = TorchScalarVariable(
            name="test", value_range=(0.0, 10.0), default_validation_config="warn"
        )
        with pytest.warns(UserWarning):
            var.validate_value(15.0)

    def test_validate_value_out_of_range_none(self):
        """Test that out-of-range value passes with none config."""
        var = TorchScalarVariable(
            name="test", value_range=(0.0, 10.0), default_validation_config="none"
        )
        var.validate_value(15.0)

    def test_validate_value_with_config_override(self):
        """Test that config parameter overrides default_validation_config."""
        var = TorchScalarVariable(
            name="test", value_range=(0.0, 10.0), default_validation_config="none"
        )
        # Default is "none" but we override with "error"
        with pytest.raises(ValueError, match="out of valid range"):
            var.validate_value(15.0, config="error")

    # Read-only validation tests
    # Note: read-only validation is handled by validate_read_only, not
    # validate_value. validate_value only checks type/dtype/range.
    # The model layer is responsible for calling validate_read_only separately.
    def test_read_only_matching_value(self):
        """Test read-only variable with matching value."""
        var = TorchScalarVariable(name="test", default_value=5.0, read_only=True)
        var.validate_read_only(5.0, config="error")

        # Test read-only variable with matching tensor value.
        var = TorchScalarVariable(name="test", default_value=5.0, read_only=True)
        var.validate_read_only(torch.tensor(5.0), config="error")

        # Test read-only variable with matching tensor value.
        var = TorchScalarVariable(
            name="test", default_value=torch.tensor(5.0), read_only=True
        )
        var.validate_read_only(5.0, config="error")

    def test_read_only_non_matching_value_raises(self):
        """Test read-only variable with non-matching value raises."""
        var = TorchScalarVariable(name="test", default_value=5.0, read_only=True)
        with pytest.raises(ValueError, match="read-only"):
            var.validate_read_only(10.0, config="error")

    def test_read_only_no_default_raises(self):
        """Test read-only variable without default raises."""
        var = TorchScalarVariable(name="test", read_only=True)
        with pytest.raises(ValueError, match="no default value"):
            var.validate_read_only(5.0, config="error")

    def test_read_only_tensor_near_default_within_tolerance(self):
        """Test read-only with values very close to default within tolerance."""
        var = TorchScalarVariable(name="test", default_value=5.0, read_only=True)
        with pytest.raises(
            ValueError, match="read-only and must equal its default value"
        ):
            var.validate_read_only(
                torch.tensor(5.0 + 1e-5), config="error", atol=1e-9, rtol=1e-9
            )

    # Serialization tests
    def test_model_dump(self):
        """Test model_dump includes variable_class."""
        var = TorchScalarVariable(name="test", default_value=1.0)
        dump = var.model_dump()
        assert "variable_class" in dump
        assert dump["variable_class"] == "TorchScalarVariable"
        assert dump["name"] == "test"

    def test_dtype_none_allows_any_float_dtype(self):
        """Test that dtype=None allows any floating-point dtype."""
        var = TorchScalarVariable(name="test", dtype=None)
        var.validate_value(torch.tensor(5.0, dtype=torch.float32))
        var.validate_value(torch.tensor(5.0, dtype=torch.float64))

    def test_model_dump_roundtrip(self):
        """Test model_dump includes all relevant attributes."""
        var = TorchScalarVariable(
            name="test",
            default_value=torch.tensor([5.0], dtype=torch.double),
            value_range=(0.0, 10.0),
            unit="meters",
            read_only=False,
            dtype=torch.double,
        )
        dumped = var.model_dump()
        json_str = json.dumps(dumped)
        parsed = json.loads(json_str)
        var2 = TorchScalarVariable(**parsed)
        assert var2.name == var.name
        assert var2.dtype == var.dtype
        assert var2.value_range == var.value_range
        assert var2.unit == var.unit
        # After JSON round-trip default_value is a plain float
        assert var2.default_value == 5.0

    def test_legacy_is_constant_maps_to_read_only(self):
        """Legacy `is_constant` should set `read_only`."""
        var = TorchScalarVariable(name="test", is_constant=True)
        assert var.read_only is True

    def test_read_only_overrides_legacy_is_constant(self):
        """Canonical field should win when both keys are provided."""
        var = TorchScalarVariable(name="test", read_only=False, is_constant=True)
        assert var.read_only is False

    def test_unknown_field_raises(self):
        """Misspelled or unknown fields should raise a ValidationError."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            TorchScalarVariable(name="test", range=(0.0, 1.0))  # should be value_range


class TestTorchNDVariable:
    """Tests for TorchNDVariable class."""

    def test_basic_creation(self):
        """Test basic ND variable creation."""
        var = TorchNDVariable(name="test_nd", shape=(10, 20))
        assert var.name == "test_nd"
        assert var.shape == (10, 20)
        assert var.dtype == torch.float32

    def test_unknown_field_raises(self):
        """Misspelled or unknown fields should raise a ValidationError."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            TorchNDVariable(name="test", shape=(4,), shpe=(4,))  # typo: shpe vs shape

    def test_missing_shape_raises_validation_error(self):
        """Test that missing shape raises ValidationError."""
        with pytest.raises(ValidationError):
            TorchNDVariable(name="test")

    def test_creation_with_default_value(self):
        """Test ND variable creation with default value."""
        default = torch.randn(10, 20)
        var = TorchNDVariable(name="test_nd", shape=(10, 20), default_value=default)
        assert torch.allclose(var.default_value, default)

    def test_invalid_dtype_type_raises(self):
        """Test invalid dtype type raises TypeError."""
        with pytest.raises(TypeError, match="dtype must be a"):
            TorchNDVariable(name="test", shape=(10,), dtype="invalid")

    def test_reject_ndarray(self):
        var = TorchNDVariable(name="t", shape=(3, 4))
        with pytest.raises(TypeError, match="Expected value to be a Tensor"):
            var.validate_value(np.zeros((3, 4)), config="error")

    def test_default_value_validated_on_creation(self):
        """Wrong dtype in default_value must be caught at construction time."""
        with pytest.raises(ValidationError, match="Expected dtype"):
            TorchNDVariable(
                name="t",
                shape=(2,),
                dtype=torch.float64,
                default_value=torch.zeros(2, dtype=torch.float32),
            )

    # Value validation tests
    def test_validate_value_correct_tensor(self):
        """Test validation of correct tensor value."""
        var = TorchNDVariable(name="test", shape=(10, 20))
        var.validate_value(torch.randn(10, 20))

    def test_validate_value_batched_tensor(self):
        """Test validation of batched tensor succeeds when trailing shape matches."""
        var = TorchNDVariable(name="test", shape=(10, 20))
        var.validate_value(torch.randn(5, 10, 20))

        with pytest.raises(ValueError, match="Expected shape"):
            var.validate_value(torch.randn(5, 10, 30))

    def test_validate_value_wrong_type_raises(self):
        """Test that non-tensor value raises TypeError."""
        var = TorchNDVariable(name="test", shape=(10, 20))
        with pytest.raises(TypeError, match="Expected value to be a Tensor"):
            var.validate_value([[1, 2], [3, 4]])

    # Read-only validation tests
    # Note: read-only validation is handled by validate_read_only, not
    # validate_value. validate_value only checks type/dtype/range/shape.
    # The model layer is responsible for calling validate_read_only separately
    # on an already validated value.
    def test_read_only_matching_value(self):
        """Test read-only ND variable with matching value."""
        default = torch.randn(10, 20)
        var = TorchNDVariable(
            name="test", shape=(10, 20), default_value=default, read_only=True
        )
        var.validate_read_only(default, config="error")

    def test_read_only_non_matching_value_raises(self):
        """Test read-only ND variable with non-matching value raises."""
        default = torch.randn(10, 20)
        var = TorchNDVariable(
            name="test", shape=(10, 20), default_value=default, read_only=True
        )
        with pytest.raises(ValueError, match="read-only"):
            var.validate_read_only(torch.randn(10, 20), config="error")

    def test_read_only_no_default_raises(self):
        """Test read-only ND variable without default raises."""
        var = TorchNDVariable(name="test", shape=(10, 20), read_only=True)
        with pytest.raises(ValueError, match="no default value"):
            var.validate_read_only(torch.randn(10, 20), config="error")

    def test_read_only_value_within_tolerance(self):
        """Test read-only ND variable with values within tolerance."""
        default = torch.randn(10, 20)
        var = TorchNDVariable(
            name="test", shape=(10, 20), default_value=default, read_only=True
        )
        # Value very close to default (within tolerance)
        close_value = default + 1e-10
        var.validate_read_only(close_value, config="error", atol=1e-9, rtol=1e-9)
        # Value not close to default
        not_close_value = default + 1e-5
        with pytest.raises(
            ValueError, match="read-only and must equal its default value"
        ):
            var.validate_read_only(
                not_close_value, config="error", atol=1e-9, rtol=1e-9
            )

    def test_legacy_is_constant_maps_to_read_only(self):
        """Legacy `is_constant` should set `read_only`."""
        var = TorchNDVariable(name="test", shape=(2,), is_constant=True)
        assert var.read_only is True

    # Test dtype coercion and validation
    def test_string_dtype_coerced(self):
        var = TorchNDVariable(name="t", shape=(2,), dtype="float32")
        assert var.dtype == torch.float32

    def test_string_dtype_with_prefix_coerced(self):
        """Strings like 'torch.float64' (from old serializations) are accepted."""
        var = TorchNDVariable(name="t", shape=(2,), dtype="torch.float64")
        assert var.dtype == torch.float64

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            TorchNDVariable(name="t", shape=(2,), dtype="double")

    # Test serialization of dtype and default_value
    def test_dtype_serialized_as_string(self):
        var = TorchNDVariable(name="t", shape=(3, 4))
        dumped = var.model_dump()
        assert dumped["dtype"] == "torch.float32"
        assert isinstance(dumped["dtype"], str)

    def test_dtype_float32_serialized(self):
        var = TorchNDVariable(name="t", shape=(3, 4), dtype=torch.float32)
        assert var.model_dump()["dtype"] == "torch.float32"

    def test_default_value_none_serialized(self):
        var = TorchNDVariable(name="t", shape=(3, 4))
        assert var.model_dump()["default_value"] is None

    def test_default_value_serialized_as_list(self):
        t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        var = TorchNDVariable(name="t", shape=(3, 4), default_value=t)
        dumped = var.model_dump()
        assert dumped["default_value"] == t.tolist()
        assert isinstance(dumped["default_value"], list)

    def test_variable_class_in_dump(self):
        var = TorchNDVariable(name="t", shape=(3, 4))
        assert var.model_dump()["variable_class"] == "TorchNDVariable"

    def test_model_dump_is_json_serializable(self):
        t = torch.ones(3, 4)
        var = TorchNDVariable(name="t", shape=(3, 4), default_value=t)
        dumped = var.model_dump()
        # Must not raise
        json_str = json.dumps(dumped)
        parsed = json.loads(json_str)
        assert parsed["dtype"] == "torch.float32"
        assert parsed["default_value"] == t.tolist()

    # Test round-trip serialization and deserialization
    def test_roundtrip_no_default_value(self):
        var = TorchNDVariable(name="weights", shape=(3, 4), dtype=torch.float32)
        dumped = var.model_dump()
        json_str = json.dumps(dumped)
        parsed = json.loads(json_str)
        parsed.pop("variable_class", None)

        var2 = TorchNDVariable(**parsed)
        assert var2.name == var.name
        assert var2.shape == var.shape
        assert var2.dtype == var.dtype
        assert var2.default_value is None

    def test_roundtrip_with_default_value(self):
        t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        var = TorchNDVariable(name="t", shape=(2, 3), default_value=t)
        dumped = var.model_dump()
        json_str = json.dumps(dumped)
        parsed = json.loads(json_str)

        var2 = TorchNDVariable(**parsed)
        assert var2.name == var.name
        assert var2.shape == var.shape
        assert var2.dtype == var.dtype
        assert torch.equal(var2.default_value, t)


class TestDistributionVariable:
    """Tests for DistributionVariable class."""

    def test_basic_creation(self):
        """Test basic distribution variable creation."""
        var = DistributionVariable(name="dist_var")
        assert var.name == "dist_var"
        assert var.unit is None

    def test_validate_normal_distribution(self):
        """Test validation of Normal distribution."""
        var = DistributionVariable(name="test")
        dist = Normal(loc=0.0, scale=1.0)
        var.validate_value(dist)

    def test_validation_error_wrong_type(self):
        """Test validation with wrong type raises TypeError."""
        var = DistributionVariable(name="dist")
        with pytest.raises(TypeError):
            var.validate_value("not_a_distribution")

    def test_legacy_is_constant_maps_to_read_only(self):
        """Legacy `is_constant` should set `read_only`."""
        var = DistributionVariable(name="dist", is_constant=True)
        assert var.read_only is True
