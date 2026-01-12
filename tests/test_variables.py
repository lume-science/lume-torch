import pytest
import numpy as np
import torch
from pydantic import ValidationError

from lume_model.variables import (
    ScalarVariable,
    ArrayVariable,
    NumpyNDArray,
    TorchTensor,
    ArrayValidatorMixin,
    get_variable,
)


class TestScalarVariable:
    def test_init(self):
        ScalarVariable(
            name="test",
            default_value=0.1,
            value_range=(0, 1),
            unit="m",
        )
        # missing name
        with pytest.raises(ValidationError):
            ScalarVariable(default_value=0.1, value_range=(0, 1))
        # constant variable
        ScalarVariable(
            name="test",
            default_value=0.5,
            value_range=None,
        )

    def test_validate_value(self):
        var = ScalarVariable(
            name="test",
            default_value=0.8,
            value_range=(0.0, 10.0),
            value_range_tolerance=1e-8,
        )
        var.validate_value(5.0)
        with pytest.raises(TypeError):
            var.validate_value(int(5))
        # test validation config
        var.validate_value(11.0, config="none")
        # range check with strictness flag
        with pytest.raises(ValueError):
            var.validate_value(11.0, config="error")
        # constant variable
        constant_var = ScalarVariable(
            name="test",
            default_value=1.3,
            is_constant=True,
            value_range_tolerance=1e-5,
        )
        constant_var.validate_value(1.3, config="error")
        with pytest.raises(ValueError):
            constant_var.validate_value(1.4, config="error")
        # test tolerance
        var.validate_value(10.0 + 1e-9, config="error")
        with pytest.raises(ValueError):
            var.validate_value(10.0 + 1e-7, config="error")
        constant_var.validate_value(1.3 + 1e-6, config="error")
        with pytest.raises(ValueError):
            constant_var.validate_value(1.3 + 1e-4, config="error")
        # test constant range validation
        with pytest.raises(ValueError):
            constant_var = ScalarVariable(
                name="test",
                default_value=1.3,
                is_constant=True,
                value_range_tolerance=1e-5,
                value_range=(1.3, 1.5),
            )


class TestArrayValidatorMixin:
    """Tests for the ArrayValidatorMixin utility class."""

    def test_validate_type(self):
        """Should validate type correctly."""
        arr = np.array([1.0, 2.0])
        ArrayValidatorMixin._validate_type(arr, np.ndarray, "numpy.ndarray")

        with pytest.raises(TypeError, match="must be a numpy.ndarray"):
            ArrayValidatorMixin._validate_type([1.0, 2.0], np.ndarray, "numpy.ndarray")

    def test_validate_shape(self):
        """Should validate shape correctly."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        ArrayValidatorMixin._validate_shape(arr, (2, 2))

        with pytest.raises(ValueError, match="Expected shape"):
            ArrayValidatorMixin._validate_shape(arr, (3, 3))

    def test_validate_dtype(self):
        """Should validate dtype correctly."""
        arr = np.array([1.0, 2.0], dtype=np.float64)
        ArrayValidatorMixin._validate_dtype(arr, np.float64)

        with pytest.raises(ValueError, match="Expected dtype"):
            ArrayValidatorMixin._validate_dtype(arr, np.float32)


class TestNumpyNDArray:
    """Tests for NumpyNDArray validator."""

    def test_validate_type(self):
        """Should accept numpy arrays and reject other types."""
        arr = np.array([1.0, 2.0, 3.0])
        result = NumpyNDArray.validate(arr)
        assert isinstance(result, np.ndarray)

        with pytest.raises(TypeError, match="must be a numpy.ndarray"):
            NumpyNDArray.validate([1.0, 2.0, 3.0])

        with pytest.raises(TypeError, match="must be a numpy.ndarray"):
            NumpyNDArray.validate(torch.tensor([1.0, 2.0]))

    def test_validate_shape(self):
        """Should validate array shape correctly."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])

        # Matching shape should pass
        NumpyNDArray.validate(arr, expected_shape=(2, 2))

        # Wrong shape should fail
        with pytest.raises(ValueError, match="Expected shape"):
            NumpyNDArray.validate(arr, expected_shape=(3, 3))

        # None shape should skip validation
        NumpyNDArray.validate(arr, expected_shape=None)

    def test_validate_dtype(self):
        """Should validate array dtype correctly."""
        arr_float64 = np.array([1.0, 2.0], dtype=np.float64)
        arr_float32 = np.array([1.0, 2.0], dtype=np.float32)

        # Matching dtype should pass
        NumpyNDArray.validate(arr_float64, expected_dtype=np.float64)
        NumpyNDArray.validate(arr_float32, expected_dtype=np.float32)

        # Wrong dtype should fail
        with pytest.raises(ValueError, match="Expected dtype"):
            NumpyNDArray.validate(arr_float64, expected_dtype=np.float32)

        # None dtype should skip validation
        NumpyNDArray.validate(arr_float64, expected_dtype=None)

    def test_validate_combined(self):
        """Should validate type, shape, and dtype together."""
        arr = np.zeros((3, 4), dtype=np.float32)

        NumpyNDArray.validate(
            arr,
            expected_shape=(3, 4),
            expected_dtype=np.float32,
        )


class TestTorchTensor:
    """Tests for TorchTensor validator."""

    def test_validate_type(self):
        """Should accept torch tensors and reject other types."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = TorchTensor.validate(tensor)
        assert isinstance(result, torch.Tensor)

        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            TorchTensor.validate([1.0, 2.0, 3.0])

        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            TorchTensor.validate(np.array([1.0, 2.0]))

    def test_validate_shape(self):
        """Should validate tensor shape correctly."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        # Matching shape should pass
        TorchTensor.validate(tensor, expected_shape=(2, 2))

        # Wrong shape should fail
        with pytest.raises(ValueError, match="Expected shape"):
            TorchTensor.validate(tensor, expected_shape=(3, 3))

    def test_validate_dtype(self):
        """Should validate tensor dtype correctly."""
        tensor_float32 = torch.tensor([1.0, 2.0], dtype=torch.float32)
        tensor_float64 = torch.tensor([1.0, 2.0], dtype=torch.float64)

        # Matching dtype should pass
        TorchTensor.validate(tensor_float32, expected_dtype=torch.float32)
        TorchTensor.validate(tensor_float64, expected_dtype=torch.float64)

        # Wrong dtype should fail
        with pytest.raises(ValueError, match="Expected dtype"):
            TorchTensor.validate(tensor_float32, expected_dtype=torch.float64)

    def test_validate_device(self):
        """Should validate tensor device correctly."""
        tensor = torch.tensor([1.0, 2.0])  # defaults to CPU

        # CPU device should pass
        TorchTensor.validate(tensor, expected_device="cpu")

        # Wrong device should fail
        with pytest.raises(ValueError, match="Expected device"):
            TorchTensor.validate(tensor, expected_device="cuda")

        # None device should skip validation
        TorchTensor.validate(tensor, expected_device=None)

    def test_validate_combined(self):
        """Should validate type, shape, dtype, and device together."""
        tensor = torch.zeros((3, 4), dtype=torch.float32)

        TorchTensor.validate(
            tensor,
            expected_shape=(3, 4),
            expected_dtype=torch.float32,
            expected_device="cpu",
        )


class TestArrayVariable:
    """Tests for ArrayVariable."""

    def test_init_numpy(self):
        """Should initialize with numpy array type."""
        var = ArrayVariable(
            name="test_array",
            shape=(3, 4),
            dtype=np.float32,
            array_type="numpy",
        )
        assert var.name == "test_array"
        assert var.shape == (3, 4)
        assert var.dtype == np.float32
        assert var.array_type == "numpy"
        assert var.is_image is False

    def test_init_torch(self):
        """Should initialize with torch tensor type."""
        var = ArrayVariable(
            name="test_tensor",
            shape=(3, 4),
            dtype=torch.float32,
            array_type="torch",
            device="cpu",
        )
        assert var.name == "test_tensor"
        assert var.shape == (3, 4)
        assert var.dtype == torch.float32
        assert var.array_type == "torch"
        assert var.device == "cpu"

    def test_init_with_default_value_numpy(self):
        """Should validate default value for numpy arrays."""
        arr = np.zeros((2, 3), dtype=np.float64)
        var = ArrayVariable(
            name="test",
            shape=(2, 3),
            dtype=np.float64,
            array_type="numpy",
            default_value=arr,
        )
        assert var.default_value is not None

    def test_init_with_default_value_torch(self):
        """Should validate default value for torch tensors."""
        tensor = torch.zeros((2, 3), dtype=torch.float32)
        var = ArrayVariable(
            name="test",
            shape=(2, 3),
            dtype=torch.float32,
            array_type="torch",
            default_value=tensor,
        )
        assert var.default_value is not None

    def test_init_with_invalid_default_value_shape(self):
        """Should reject default value with wrong shape."""
        arr = np.zeros((3, 4), dtype=np.float64)
        with pytest.raises(ValidationError):
            ArrayVariable(
                name="test",
                shape=(2, 3),
                dtype=np.float64,
                array_type="numpy",
                default_value=arr,
            )

    def test_init_with_invalid_default_value_type(self):
        """Should reject default value with wrong type."""
        tensor = torch.zeros((2, 3))
        with pytest.raises(ValidationError):
            ArrayVariable(
                name="test",
                shape=(2, 3),
                dtype=np.float64,
                array_type="numpy",
                default_value=tensor,
            )

    def test_validate_value_numpy(self):
        """Should validate numpy array values."""
        var = ArrayVariable(
            name="test",
            shape=(2, 3),
            dtype=np.float64,
            array_type="numpy",
        )
        arr = np.zeros((2, 3), dtype=np.float64)
        var.validate_value(arr)

        # Wrong shape
        with pytest.raises(ValueError):
            var.validate_value(np.zeros((3, 2), dtype=np.float64))

        # Wrong dtype
        with pytest.raises(ValueError):
            var.validate_value(np.zeros((2, 3), dtype=np.float32))

        # Wrong type
        with pytest.raises(ValueError):
            var.validate_value(torch.zeros((2, 3)))

    def test_validate_value_torch(self):
        """Should validate torch tensor values."""
        var = ArrayVariable(
            name="test",
            shape=(2, 3),
            dtype=torch.float32,
            array_type="torch",
            device="cpu",
        )
        tensor = torch.zeros((2, 3), dtype=torch.float32)
        var.validate_value(tensor)

        # Wrong shape
        with pytest.raises(ValueError):
            var.validate_value(torch.zeros((3, 2), dtype=torch.float32))

        # Wrong dtype
        with pytest.raises(ValueError):
            var.validate_value(torch.zeros((2, 3), dtype=torch.float64))

        # Wrong type
        with pytest.raises(ValueError):
            var.validate_value(np.zeros((2, 3)))

    def test_validator_class_property(self):
        """Should return correct validator class."""
        var_numpy = ArrayVariable(
            name="test", shape=(2,), dtype=np.float32, array_type="numpy"
        )
        var_torch = ArrayVariable(
            name="test", shape=(2,), dtype=torch.float32, array_type="torch"
        )
        assert var_numpy._validator_class == NumpyNDArray
        assert var_torch._validator_class == TorchTensor

    def test_expected_base_type_property(self):
        """Should return correct base type."""
        var_numpy = ArrayVariable(
            name="test", shape=(2,), dtype=np.float32, array_type="numpy"
        )
        var_torch = ArrayVariable(
            name="test", shape=(2,), dtype=torch.float32, array_type="torch"
        )
        assert var_numpy._expected_base_type == np.ndarray
        assert var_torch._expected_base_type == torch.Tensor


class TestArrayVariableImage:
    """Tests for ArrayVariable with image functionality (num_channels set)."""

    def test_is_image_property(self):
        """Should correctly identify image variables."""
        regular_var = ArrayVariable(name="test", shape=(3, 64, 64), dtype=torch.float32)
        image_var = ArrayVariable(
            name="test", shape=(3, 64, 64), dtype=torch.float32, num_channels=3
        )
        assert regular_var.is_image is False
        assert image_var.is_image is True

    def test_image_validation_torch_rgb(self):
        """Should validate RGB torch image (C, H, W)."""
        var = ArrayVariable(
            name="image",
            shape=(3, 64, 64),
            dtype=torch.float32,
            array_type="torch",
            num_channels=3,
        )
        tensor = torch.zeros((3, 64, 64), dtype=torch.float32)
        var.validate_value(tensor)

    def test_image_validation_torch_grayscale(self):
        """Should validate grayscale torch image (H, W)."""
        var = ArrayVariable(
            name="image",
            shape=(64, 64),
            dtype=torch.float32,
            array_type="torch",
            num_channels=1,
        )
        tensor = torch.zeros((64, 64), dtype=torch.float32)
        var.validate_value(tensor)

    def test_image_validation_numpy_rgb(self):
        """Should validate RGB numpy image (H, W, C)."""
        var = ArrayVariable(
            name="image",
            shape=(64, 64, 3),
            dtype=np.float32,
            array_type="numpy",
            num_channels=3,
        )
        arr = np.zeros((64, 64, 3), dtype=np.float32)
        var.validate_value(arr)

    def test_image_validation_numpy_grayscale(self):
        """Should validate grayscale numpy image (H, W)."""
        var = ArrayVariable(
            name="image",
            shape=(64, 64),
            dtype=np.float32,
            array_type="numpy",
            num_channels=1,
        )
        arr = np.zeros((64, 64), dtype=np.float32)
        var.validate_value(arr)

    def test_image_invalid_shape_4d(self):
        """Should reject 4D shapes for images."""
        with pytest.raises(ValidationError):
            ArrayVariable(
                name="image",
                shape=(1, 3, 64, 64),
                dtype=torch.float32,
                array_type="torch",
                num_channels=3,
            )

    def test_image_wrong_channel_count_torch(self):
        """Should reject wrong channel count for torch images."""
        var = ArrayVariable(
            name="image",
            shape=(3, 64, 64),
            dtype=torch.float32,
            array_type="torch",
            num_channels=3,
        )
        # Wrong number of channels in first dimension
        tensor_wrong = torch.zeros((4, 64, 64), dtype=torch.float32)
        with pytest.raises(ValueError, match="Expected shape"):
            var.validate_value(tensor_wrong)

    def test_image_wrong_channel_count_numpy(self):
        """Should reject wrong channel count for numpy images."""
        var = ArrayVariable(
            name="image",
            shape=(64, 64, 3),
            dtype=np.float32,
            array_type="numpy",
            num_channels=3,
        )
        # Wrong number of channels in last dimension
        arr_wrong = np.zeros((64, 64, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected shape"):
            var.validate_value(arr_wrong)


def test_get_variable():
    var = get_variable(ScalarVariable.__name__)(name="test")
    assert isinstance(var, ScalarVariable)


def test_get_variable_array():
    """Should be able to get ArrayVariable by name."""
    var = get_variable("ArrayVariable")(
        name="test", shape=(2, 3), dtype=np.float32, array_type="numpy"
    )
    assert isinstance(var, ArrayVariable)
