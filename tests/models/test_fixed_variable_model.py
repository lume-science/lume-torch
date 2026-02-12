# test_fixed_variable_model.py

import pytest

try:
    import torch
    from lume_torch.variables import TorchScalarVariable
    from lume_torch.models.torch_module import (
        TorchModel,
        TorchModule,
        FixedVariableModel,
    )

    torch.manual_seed(42)
except ImportError:
    pass


@pytest.fixture
def synthetic_lume_torch():
    """
    Create synthetic LUME model: f(x, y) = 0.5*x^2 + y^2
    """

    class PriorTorchModel(torch.nn.Module):
        def forward(self, X):
            x = X[..., 0]
            y = X[..., 1]
            return 0.5 * x**2 + y**2

    input_variables = [
        TorchScalarVariable(name="x", default_value=0.0, value_range=[-3.0, 3.0]),
        TorchScalarVariable(name="y", default_value=1.0, value_range=[0.5, 1.5]),
    ]

    output_variables = [TorchScalarVariable(name="f")]

    torch_model = TorchModel(
        model=PriorTorchModel(),
        input_variables=input_variables,
        output_variables=output_variables,
    )

    return TorchModule(model=torch_model)


@pytest.fixture
def prior_model(synthetic_lume_torch):
    """Create FixedVariableModel with y fixed at 1.0"""
    return FixedVariableModel(model=synthetic_lume_torch, fixed_values={"y": 1.0})


def test_initialization(prior_model):
    """Test basic initialization."""
    assert len(prior_model.all_inputs) == 2
    assert prior_model.control_variables == ["x"]
    assert len(prior_model.fixed_indices) == 1
    assert prior_model.input_buffer.shape == (2,)


def test_forward_single_sample(prior_model):
    """Test forward pass with single sample."""
    x = torch.tensor([[2.0]], dtype=torch.float32)
    output = prior_model(x)

    # Expected: 0.5 * 2^2 + 1^2 = 3.0
    # Convert output to same dtype for comparison
    assert abs(output.item() - 3.0) < 1e-5


def test_forward_batch(prior_model):
    """Test forward pass with batch."""
    x = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32)
    output = prior_model(x)

    # Expected: [1.0, 1.5, 3.0]
    expected = [1.0, 1.5, 3.0]
    for i, exp_val in enumerate(expected):
        assert abs(output[i].item() - exp_val) < 1e-5


def test_update_fixed_values(prior_model):
    """Test updating fixed variables."""
    x = torch.tensor([[0.0]], dtype=torch.float32)

    # Before update: y=1.0, output = 0.5*0 + 1 = 1.0
    output_before = prior_model(x)
    assert abs(output_before.item() - 1.0) < 1e-5

    # Update y to 2.0
    prior_model.update_fixed_values({"y": 2.0})

    # After update: y=2.0, output = 0.5*0 + 4 = 4.0
    output_after = prior_model(x)
    assert abs(output_after.item() - 4.0) < 1e-5


def test_buffers_registered(prior_model):
    """Test that buffers are properly registered."""
    state_dict = prior_model.state_dict()
    assert "input_buffer" in state_dict
    assert "control_indices" in state_dict


def test_gradient_flow(prior_model):
    """Test gradients flow correctly (needed for GP training)."""
    x = torch.tensor([[1.0]], dtype=torch.float32, requires_grad=True)
    output = prior_model(x)
    output.backward()

    # Gradient of 0.5*x^2 at x=1 is x = 1.0
    assert abs(x.grad.item() - 1.0) < 1e-5


def test_negative_input(prior_model):
    """Test with negative values."""
    x = torch.tensor([[-2.0]], dtype=torch.float32)
    output = prior_model(x)

    # Expected: 0.5 * 4 + 1 = 3.0
    assert abs(output.item() - 3.0) < 1e-5


def test_zero_input(prior_model):
    """Test with zero input."""
    x = torch.tensor([[0.0]], dtype=torch.float32)
    output = prior_model(x)

    # Expected: 0.5 * 0 + 1 = 1.0
    assert abs(output.item() - 1.0) < 1e-5


def test_device_compatibility(prior_model):
    """Test model works on CPU (and GPU if available)."""
    prior_model = prior_model.cpu()
    x = torch.tensor([[1.0]], dtype=torch.float32)
    output = prior_model(x)
    assert output.device.type == "cpu"

    if torch.cuda.is_available():
        prior_model = prior_model.cuda()
        x = x.cuda()
        output = prior_model(x)
        assert output.device.type == "cuda"
