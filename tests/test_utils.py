import os

import pytest
import torch

from lume_torch.models.utils import format_inputs
from lume_torch.utils import (
    verify_unique_variable_names,
    variables_as_yaml,
    variables_from_yaml,
)


def test_verify_unique_variable_names(simple_variables):
    input_variables = simple_variables["input_variables"]
    output_variables = simple_variables["output_variables"]
    # unique variables names
    verify_unique_variable_names(input_variables)
    verify_unique_variable_names(output_variables)
    # non-unique input names
    original_name = input_variables[1].name
    input_variables[1].name = input_variables[0].name
    with pytest.raises(ValueError):
        verify_unique_variable_names(input_variables)
    input_variables[1].name = original_name
    # non-unique output names
    original_name = output_variables[1].name
    output_variables[1].name = output_variables[0].name
    with pytest.raises(ValueError):
        verify_unique_variable_names(output_variables)
    output_variables[1].name = original_name


def test_variables_as_yaml(simple_variables):
    file = "test_variables.yml"
    variables_as_yaml(**simple_variables, file=file)
    os.remove(file)


def test_variables_as_and_from_yaml(simple_variables):
    file = "test_variables.yml"
    variables_as_yaml(**simple_variables, file=file)
    variables = variables_from_yaml(file)
    os.remove(file)
    assert simple_variables["input_variables"] == variables[0]
    assert simple_variables["output_variables"] == variables[1]


class TestFormatInputs:
    def test_converts_floats_to_tensors(self):
        result = format_inputs({"x": 1.0, "y": 2.0})
        assert isinstance(result["x"], torch.Tensor)
        assert isinstance(result["y"], torch.Tensor)
        assert result["x"].item() == pytest.approx(1.0)

    def test_passthrough_tensors(self):
        t = torch.tensor([3.0, 4.0])
        result = format_inputs({"x": t})
        assert torch.equal(result["x"], t)

    def test_none_tensor_kwargs_does_not_raise(self):
        # tensor_kwargs=None (the default) must not cause a TypeError
        result = format_inputs({"x": 1.0}, tensor_kwargs=None)
        assert isinstance(result["x"], torch.Tensor)

        result = format_inputs({"x": 1.0}, tensor_kwargs={"dtype": torch.float64})
        assert result["x"].dtype == torch.float64

    def test_squeeze_false_preserves_shape(self):
        t = torch.tensor([[1.0, 2.0]])  # shape (1, 2)
        result = format_inputs({"x": t}, squeeze=False)
        assert result["x"].shape == (1, 2)

    def test_squeeze_true_removes_last_dim(self):
        t = torch.tensor([[1.0], [2.0]])  # shape (2, 1)
        result = format_inputs({"x": t}, squeeze=True)
        assert result["x"].shape == (2,)
