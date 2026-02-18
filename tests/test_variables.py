"""Test suite for lume_torch.variables module."""

import pytest
import torch
from pydantic import ValidationError
from torch.distributions import Normal, Uniform

from lume_torch.variables import (
    ScalarVariable,
    TorchScalarVariable,
    TorchNDVariable,
    DistributionVariable,
    Variable,
    ConfigEnum,
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
        # ScalarVariable is now a subclass, not an alias
        assert issubclass(ScalarVariable, TorchScalarVariable)
        # Verify it issues a deprecation warning when instantiated
        with pytest.warns(DeprecationWarning, match="ScalarVariable is deprecated"):
            var = ScalarVariable(name="test_var", default_value=1.0)
        # Verify it still works as expected
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
                default_value=torch.tensor(1.5),
                value_range=(0.0, 10.0),
                unit="meters",
                read_only=True,
                dtype=torch.float64,
            )

    def test_missing_name_raises_validation_error(self):
        """Test that missing name raises ValidationError."""
        with pytest.raises(ValidationError):
            TorchScalarVariable(default_value=0.1, value_range=(0, 1))

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
    def test_dtype_float32(self):
        """Test dtype with torch.float32."""
        var = TorchScalarVariable(name="test", dtype=torch.float32)
        assert var.dtype == torch.float32

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
        var.validate_value(5.0)  # Should not raise

    def test_validate_value_int(self):
        """Test validation of int values (allowed as scalars)."""
        var = TorchScalarVariable(name="test")
        var.validate_value(5)  # Should not raise

    def test_validate_value_tensor_0d(self):
        """Test validation of 0D tensor."""
        var = TorchScalarVariable(name="test")
        var.validate_value(torch.tensor(5.0))  # Should not raise

    def test_validate_value_tensor_1d(self):
        """Test validation of 1D tensor."""
        var = TorchScalarVariable(name="test")
        var.validate_value(torch.tensor([[5.0], [6.0], [7.0]]))  # Should not raise

    def test_validate_value_tensor_batched(self):
        """Test validation of batched tensor with last dim = 1."""
        var = TorchScalarVariable(name="test")
        var.validate_value(torch.tensor([[5.0], [6.0]]))  # Should not raise

    def test_validate_value_tensor_invalid_shape_raises(self):
        """Test that tensor with invalid shape raises ValueError."""
        var = TorchScalarVariable(name="test")
        with pytest.raises(ValueError, match="Expected tensor with 0 dimensions,"):
            var.validate_value(torch.tensor([[5.0, 6.0], [7.0, 8.0]]))

    def test_validate_value_invalid_type_raises(self):
        """Test that invalid value type raises TypeError."""
        var = TorchScalarVariable(name="test")
        with pytest.raises(TypeError):
            var.validate_value("not_a_number")

    def test_validate_value_non_float_tensor_raises(self):
        """Test that non-float tensor raises ValueError."""
        var = TorchScalarVariable(name="test")
        with pytest.raises(ValueError, match="floating-point type"):
            var.validate_value(torch.tensor([[1], [2], [3]]))  # int64 tensor

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
        var.validate_value(5.0)  # Should not raise

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
        var.validate_value(15.0)  # Should not raise

    def test_validate_value_with_config_override(self):
        """Test that config parameter overrides default_validation_config."""
        var = TorchScalarVariable(
            name="test", value_range=(0.0, 10.0), default_validation_config="none"
        )
        # Default is "none" but we override with "error"
        with pytest.raises(ValueError, match="out of valid range"):
            var.validate_value(15.0, config="error")

    def test_validate_value_config_enum_object(self):
        """Test validation with ConfigEnum object instead of string."""
        var = TorchScalarVariable(
            name="test",
            value_range=(0.0, 10.0),
            default_validation_config=ConfigEnum.NULL,
        )
        var.validate_value(15.0)  # Should not raise with NULL config

    def test_value_range_validation(self):
        """Test that value_range min must be <= max."""
        with pytest.raises(ValueError, match="Minimum value"):
            TorchScalarVariable(name="test", value_range=(10.0, 0.0))

    # Read-only validation tests
    def test_read_only_matching_value(self):
        """Test read-only variable with matching value."""
        var = TorchScalarVariable(name="test", default_value=5.0, read_only=True)
        var.validate_value(5.0)  # Should not raise

    def test_read_only_matching_tensor(self):
        """Test read-only variable with matching tensor value."""
        var = TorchScalarVariable(
            name="test", default_value=torch.tensor(5.0), read_only=True
        )
        var.validate_value(torch.tensor(5.0))  # Should not raise

    def test_read_only_non_matching_value_raises(self):
        """Test read-only variable with non-matching value raises."""
        var = TorchScalarVariable(name="test", default_value=5.0, read_only=True)
        with pytest.raises(ValueError, match="read-only"):
            var.validate_value(10.0)

    def test_read_only_no_default_raises(self):
        """Test read-only variable without default raises."""
        var = TorchScalarVariable(name="test", read_only=True)
        with pytest.raises(ValueError, match="no default value"):
            var.validate_value(5.0)

    def test_read_only_batched_tensor(self):
        """Test read-only validation with batched tensor."""
        var = TorchScalarVariable(name="test", default_value=5.0, read_only=True)
        # All values in batch equal to default
        var.validate_value(torch.tensor([[5.0], [5.0], [5.0]]))  # Should not raise

    def test_read_only_batched_tensor_mismatch_raises(self):
        """Test read-only validation with batched tensor containing mismatched values."""
        var = TorchScalarVariable(name="test", default_value=5.0, read_only=True)
        with pytest.raises(ValueError, match="read-only"):
            var.validate_value(torch.tensor([[5.0], [6.0], [5.0]]))

    def test_read_only_with_tensor_default_float_value(self):
        """Test read-only with tensor default but float value."""
        var = TorchScalarVariable(
            name="test", default_value=torch.tensor(5.0), read_only=True
        )
        var.validate_value(5.0)  # Float matching tensor default should work

    def test_read_only_with_float_default_tensor_value(self):
        """Test read-only with float default but tensor value."""
        var = TorchScalarVariable(name="test", default_value=5.0, read_only=True)
        var.validate_value(
            torch.tensor(5.0)
        )  # Tensor matching float default should work

    def test_read_only_with_multidim_batched_tensor(self):
        """Test read-only with multi-dimensional batched tensor."""
        var = TorchScalarVariable(name="test", default_value=5.0, read_only=True)
        # Shape (2, 3, 1) - multiple batch dimensions
        batched = torch.full((2, 3, 1), 5.0)
        var.validate_value(batched)  # Should not raise

    def test_read_only_tensor_near_default_within_tolerance(self):
        """Test read-only with values very close to default within tolerance."""
        var = TorchScalarVariable(name="test", default_value=5.0, read_only=True)
        # Value very close to default (within 1e-9 tolerance)
        var.validate_value(torch.tensor(5.0 + 1e-10))  # Should not raise

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
        var.validate_value(torch.tensor(5.0, dtype=torch.float32))  # Should not raise
        var.validate_value(torch.tensor(5.0, dtype=torch.float64))  # Should not raise

    def test_model_dump_includes_all_attributes(self):
        """Test model_dump includes all relevant attributes."""
        var = TorchScalarVariable(
            name="test",
            default_value=5.0,
            value_range=(0.0, 10.0),
            unit="meters",
            read_only=False,
        )
        dump = var.model_dump()
        assert dump["name"] == "test"
        assert dump["default_value"] == 5.0
        assert dump["value_range"] == (0.0, 10.0)
        assert dump["unit"] == "meters"
        assert dump["read_only"] is False

    def test_numpy_float_value(self):
        """Test validation of numpy float values."""
        import numpy as np

        var = TorchScalarVariable(name="test")
        var.validate_value(np.float64(5.0))  # Should not raise


class TestTorchNDVariable:
    """Tests for TorchNDVariable class."""

    def test_basic_creation(self):
        """Test basic ND variable creation."""
        var = TorchNDVariable(name="test_nd", shape=(10, 20))
        assert var.name == "test_nd"
        assert var.shape == (10, 20)
        assert var.dtype == torch.float32

    def test_missing_name_raises_validation_error(self):
        """Test that missing name raises ValidationError."""
        with pytest.raises(ValidationError):
            TorchNDVariable(shape=(10, 20))

    def test_missing_shape_raises_validation_error(self):
        """Test that missing shape raises ValidationError."""
        with pytest.raises(ValidationError):
            TorchNDVariable(name="test")

    def test_creation_with_default_value(self):
        """Test ND variable creation with default value."""
        default = torch.randn(10, 20)
        var = TorchNDVariable(name="test_nd", shape=(10, 20), default_value=default)
        assert torch.allclose(var.default_value, default)

    def test_creation_with_dtype(self):
        """Test dtype with torch dtype object."""
        var = TorchNDVariable(name="test", shape=(10,), dtype=torch.float64)
        assert var.dtype == torch.float64

    def test_creation_with_int_dtype(self):
        """Test int dtype is accepted."""
        var = TorchNDVariable(name="test", shape=(10,), dtype=torch.int32)
        assert var.dtype == torch.int32

    def test_invalid_dtype_type_raises(self):
        """Test invalid dtype type raises TypeError."""
        with pytest.raises(TypeError, match="dtype must be a"):
            TorchNDVariable(name="test", shape=(10,), dtype="invalid")

    # Value validation tests
    def test_validate_value_correct_tensor(self):
        """Test validation of correct tensor value."""
        var = TorchNDVariable(name="test", shape=(10, 20))
        var.validate_value(torch.randn(10, 20))  # Should not raise

    def test_validate_value_batched_tensor(self):
        """Test validation of batched tensor."""
        var = TorchNDVariable(name="test", shape=(10, 20))
        var.validate_value(torch.randn(5, 10, 20))  # Batch of 5

    def test_validate_value_wrong_type_raises(self):
        """Test that non-tensor value raises TypeError."""
        var = TorchNDVariable(name="test", shape=(10, 20))
        with pytest.raises(TypeError, match="Expected value to be a Tensor"):
            var.validate_value([[1, 2], [3, 4]])

    def test_validate_value_wrong_shape_raises(self):
        """Test that wrong shape raises ValueError."""
        var = TorchNDVariable(name="test", shape=(10, 20))
        with pytest.raises(ValueError, match="Expected last"):
            var.validate_value(torch.randn(10, 30))

    def test_validate_value_wrong_dtype_raises(self):
        """Test that wrong dtype raises ValueError."""
        var = TorchNDVariable(name="test", shape=(10,), dtype=torch.float32)
        with pytest.raises(ValueError, match="Expected dtype"):
            var.validate_value(torch.randn(10, dtype=torch.float64))

    def test_validate_value_insufficient_dims_raises(self):
        """Test that insufficient dimensions raise ValueError."""
        var = TorchNDVariable(name="test", shape=(10, 20, 30))
        with pytest.raises(ValueError, match="Expected last 3 dimension"):
            var.validate_value(torch.randn(20, 30))

    # Read-only validation tests
    def test_read_only_matching_value(self):
        """Test read-only ND variable with matching value."""
        default = torch.randn(10, 20)
        var = TorchNDVariable(
            name="test", shape=(10, 20), default_value=default, read_only=True
        )
        var.validate_value(default.clone())  # Should not raise

    def test_read_only_non_matching_value_raises(self):
        """Test read-only ND variable with non-matching value raises."""
        default = torch.randn(10, 20)
        var = TorchNDVariable(
            name="test", shape=(10, 20), default_value=default, read_only=True
        )
        with pytest.raises(ValueError, match="read-only"):
            var.validate_value(torch.randn(10, 20))

    def test_read_only_no_default_raises(self):
        """Test read-only ND variable without default raises."""
        var = TorchNDVariable(name="test", shape=(10, 20), read_only=True)
        with pytest.raises(ValueError, match="no default value"):
            var.validate_value(torch.randn(10, 20))

    def test_read_only_batched_tensor(self):
        """Test read-only ND variable with batched tensor matching default."""
        default = torch.randn(10, 20)
        var = TorchNDVariable(
            name="test", shape=(10, 20), default_value=default, read_only=True
        )
        # Batched input where all items match default
        batched = default.unsqueeze(0).repeat(3, 1, 1)  # (3, 10, 20)
        var.validate_value(batched)  # Should not raise

    def test_read_only_batched_tensor_mismatch_raises(self):
        """Test read-only ND variable with batched tensor not matching default raises."""
        default = torch.randn(10, 20)
        var = TorchNDVariable(
            name="test", shape=(10, 20), default_value=default, read_only=True
        )
        # Batched input where items don't match default
        batched = torch.randn(3, 10, 20)
        with pytest.raises(ValueError, match="read-only"):
            var.validate_value(batched)

    def test_read_only_value_within_tolerance(self):
        """Test read-only ND variable with values within tolerance."""
        default = torch.randn(10, 20)
        var = TorchNDVariable(
            name="test", shape=(10, 20), default_value=default, read_only=True
        )
        # Value very close to default (within tolerance)
        close_value = default + 1e-10
        var.validate_value(close_value)  # Should not raise


class TestDistributionVariable:
    """Tests for DistributionVariable class."""

    def test_basic_creation(self):
        """Test basic distribution variable creation."""
        var = DistributionVariable(name="dist_var")
        assert var.name == "dist_var"
        assert var.unit is None

    def test_missing_name_raises_validation_error(self):
        """Test that missing name raises ValidationError."""
        with pytest.raises(ValidationError):
            DistributionVariable(unit="meters")

    def test_creation_with_unit(self):
        """Test distribution variable with unit."""
        var = DistributionVariable(name="dist_var", unit="meters")
        assert var.unit == "meters"

    def test_validate_normal_distribution(self):
        """Test validation of Normal distribution."""
        var = DistributionVariable(name="test")
        dist = Normal(loc=0.0, scale=1.0)
        var.validate_value(dist)  # Should not raise

    def test_validate_uniform_distribution(self):
        """Test validation of Uniform distribution."""
        var = DistributionVariable(name="test")
        dist = Uniform(low=0.0, high=1.0)
        var.validate_value(dist)  # Should not raise

    def test_validate_non_distribution_raises(self):
        """Test that non-distribution value raises TypeError."""
        var = DistributionVariable(name="test")
        with pytest.raises(TypeError, match="Expected value to be of type"):
            var.validate_value(5.0)

    def test_validate_tensor_raises(self):
        """Test that tensor value raises TypeError."""
        var = DistributionVariable(name="test")
        with pytest.raises(TypeError, match="Expected value to be of type"):
            var.validate_value(torch.tensor([1.0, 2.0]))


class TestConfigEnum:
    """Tests for ConfigEnum."""

    def test_enum_values(self):
        """Test ConfigEnum values."""
        assert ConfigEnum.NULL.value == "none"
        assert ConfigEnum.WARN.value == "warn"
        assert ConfigEnum.ERROR.value == "error"

    def test_enum_from_string(self):
        """Test ConfigEnum creation from string."""
        assert ConfigEnum("none") == ConfigEnum.NULL
        assert ConfigEnum("warn") == ConfigEnum.WARN
        assert ConfigEnum("error") == ConfigEnum.ERROR


class TestVariableInheritance:
    """Tests for variable inheritance structure."""

    def test_torch_scalar_variable_is_variable(self):
        """Test TorchScalarVariable is a Variable."""
        var = TorchScalarVariable(name="test")
        assert isinstance(var, Variable)

    def test_torch_nd_variable_is_variable(self):
        """Test TorchNDVariable is a Variable."""
        var = TorchNDVariable(name="test", shape=(10,))
        assert isinstance(var, Variable)

    def test_distribution_variable_is_variable(self):
        """Test DistributionVariable is a Variable."""
        var = DistributionVariable(name="test")
        assert isinstance(var, Variable)


class TestTorchNDVariableEdgeCases:
    """Additional edge case tests for TorchNDVariable."""

    def test_1d_shape(self):
        """Test 1D shape variable."""
        var = TorchNDVariable(name="test", shape=(100,))
        var.validate_value(torch.randn(100))  # Should not raise

    def test_4d_shape(self):
        """Test 4D shape variable (e.g., video data)."""
        var = TorchNDVariable(name="test", shape=(10, 3, 64, 64))
        var.validate_value(torch.randn(10, 3, 64, 64))  # Should not raise

    def test_nested_batch_dimensions(self):
        """Test multiple batch dimensions."""
        var = TorchNDVariable(name="test", shape=(10, 20))
        # Shape (2, 3, 10, 20) means batch_size_1=2, batch_size_2=3
        var.validate_value(torch.randn(2, 3, 10, 20))  # Should not raise

    def test_model_dump_nd_variable(self):
        """Test model_dump for TorchNDVariable."""
        var = TorchNDVariable(name="test", shape=(10, 20), unit="pixels")
        dump = var.model_dump()
        assert dump["variable_class"] == "TorchNDVariable"
        assert dump["name"] == "test"
        assert dump["shape"] == (10, 20)
        assert dump["unit"] == "pixels"

    def test_default_value_wrong_shape_raises(self):
        """Test that default value with wrong shape raises error."""
        with pytest.raises(ValueError, match="Expected last"):
            TorchNDVariable(
                name="test", shape=(10, 20), default_value=torch.randn(10, 30)
            )

    def test_default_value_wrong_dtype_raises(self):
        """Test that default value with wrong dtype raises error."""
        with pytest.raises(ValueError, match="Expected dtype"):
            TorchNDVariable(
                name="test",
                shape=(10,),
                dtype=torch.float32,
                default_value=torch.randn(10, dtype=torch.float64),
            )

    def test_validate_value_config_parameter(self):
        """Test validate_value with config parameter."""
        var = TorchNDVariable(name="test", shape=(10, 20))
        # Should work with config parameter (even though optional validation is not implemented)
        var.validate_value(torch.randn(10, 20), config="error")
        var.validate_value(torch.randn(10, 20), config="warn")
        var.validate_value(torch.randn(10, 20), config="none")


class TestDistributionVariableEdgeCases:
    """Additional edge case tests for DistributionVariable."""

    def test_read_only_attribute(self):
        """Test read_only attribute on distribution variable."""
        var = DistributionVariable(name="test", read_only=True)
        assert var.read_only is True

    def test_default_validation_config(self):
        """Test default_validation_config attribute."""
        var = DistributionVariable(name="test", default_validation_config="warn")
        assert var.default_validation_config == ConfigEnum.WARN

    def test_model_dump(self):
        """Test model_dump for DistributionVariable."""
        var = DistributionVariable(name="test", unit="meters")
        dump = var.model_dump()
        assert dump["variable_class"] == "DistributionVariable"
        assert dump["name"] == "test"
        assert dump["unit"] == "meters"

    def test_validate_with_config_parameter(self):
        """Test validate_value with config parameter."""
        var = DistributionVariable(name="test")
        dist = Normal(loc=0.0, scale=1.0)
        var.validate_value(dist, config="error")  # Should not raise
        var.validate_value(dist, config=ConfigEnum.WARN)  # Should not raise

    def test_validate_batched_distribution(self):
        """Test validation of batched distribution."""
        var = DistributionVariable(name="test")
        # Batched normal distribution
        dist = Normal(loc=torch.zeros(5), scale=torch.ones(5))
        var.validate_value(dist)  # Should not raise

    def test_validate_multivariate_distribution(self):
        """Test validation of multivariate distribution."""
        from torch.distributions import MultivariateNormal

        var = DistributionVariable(name="test")
        dist = MultivariateNormal(
            loc=torch.zeros(3),
            covariance_matrix=torch.eye(3),
        )
        var.validate_value(dist)  # Should not raise
