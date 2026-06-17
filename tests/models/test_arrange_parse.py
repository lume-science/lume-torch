"""Direct unit tests for TorchModel._arrange_inputs and _parse_outputs.

These test the internal methods directly with known tensors, covering
edge cases and error paths that aren't exercised by the integration
tests in test_torch_model.py.
"""

import pytest
import torch

from lume_torch.models import TorchModel
from lume_torch.variables import TorchScalarVariable, TorchNDVariable


# ---------------------------------------------------------------------------
# Helpers: minimal TorchModel instances with identity nn.Module
# ---------------------------------------------------------------------------


class _Identity(torch.nn.Module):
    def forward(self, x):
        return x


def _scalar_model(n_in=3, n_out=2, precision="double"):
    """TorchModel with n_in scalar inputs and n_out scalar outputs."""
    input_vars = [
        TorchScalarVariable(
            name=f"x{i}", default_value=float(i), value_range=(0.0, 10.0)
        )
        for i in range(n_in)
    ]
    output_vars = [TorchScalarVariable(name=f"y{i}") for i in range(n_out)]
    return TorchModel(
        model=_Identity(),
        input_variables=input_vars,
        output_variables=output_vars,
        precision=precision,
    )


def _nd_model(array_shape=(2, 3), n_in=1, n_out=1, precision="single"):
    """TorchModel with n_in ND inputs and n_out ND outputs."""
    input_vars = [
        TorchNDVariable(
            name=f"arr_in{i}",
            shape=array_shape,
            default_value=torch.zeros(array_shape),
        )
        for i in range(n_in)
    ]
    output_vars = [
        TorchNDVariable(name=f"arr_out{i}", shape=array_shape) for i in range(n_out)
    ]
    return TorchModel(
        model=_Identity(),
        input_variables=input_vars,
        output_variables=output_vars,
        precision=precision,
        fixed_model=False,
    )


# =========================================================================
# _arrange_inputs — Scalar path
# =========================================================================


class TestArrangeInputsScalar:
    """Tests for _arrange_inputs with TorchScalarVariable inputs."""

    def test_empty_dict_returns_defaults(self):
        """Empty formatted_inputs → (1, n_in) tensor of defaults."""
        model = _scalar_model(n_in=3)
        result = model._arrange_inputs({})
        assert result.shape == (1, 3)
        expected = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.double)
        assert torch.allclose(result, expected)

    def test_single_0d_input(self):
        """Single 0D tensor input → (1, n_in) with default fill."""
        model = _scalar_model(n_in=3)
        result = model._arrange_inputs({"x0": torch.tensor(5.0, dtype=torch.double)})
        assert result.shape == (1, 3)
        assert result[0, 0].item() == 5.0
        # x1 and x2 should be defaults
        assert result[0, 1].item() == 1.0
        assert result[0, 2].item() == 2.0

    def test_batched_1d_input(self):
        """1D tensor → (batch, n_in) with default fill for missing vars."""
        model = _scalar_model(n_in=3)
        result = model._arrange_inputs(
            {
                "x0": torch.tensor([10.0, 20.0, 30.0], dtype=torch.double),
            }
        )
        assert result.shape == (3, 3)
        assert torch.allclose(
            result[:, 0], torch.tensor([10.0, 20.0, 30.0], dtype=torch.double)
        )
        # defaults broadcast
        assert torch.all(result[:, 1] == 1.0)
        assert torch.all(result[:, 2] == 2.0)

    def test_trailing_feature_dim_squeezed(self):
        """Input shape (batch, 1) → (batch, n_in)."""
        model = _scalar_model(n_in=2)
        result = model._arrange_inputs(
            {
                "x0": torch.tensor([[5.0], [6.0]], dtype=torch.double),
                "x1": torch.tensor([[7.0], [8.0]], dtype=torch.double),
            }
        )
        assert result.shape == (2, 2)
        assert torch.allclose(
            result, torch.tensor([[5.0, 7.0], [6.0, 8.0]], dtype=torch.double)
        )

    def test_3d_batch_samples_feature(self):
        """Input shape (batch, samples, 1) → (batch, samples, n_in)."""
        model = _scalar_model(n_in=2)
        x0 = torch.ones(2, 3, 1, dtype=torch.double) * 5.0
        x1 = torch.ones(2, 3, 1, dtype=torch.double) * 9.0
        result = model._arrange_inputs({"x0": x0, "x1": x1})
        assert result.shape == (2, 3, 2)
        assert torch.all(result[..., 0] == 5.0)
        assert torch.all(result[..., 1] == 9.0)

    def test_inconsistent_batch_shapes_raises(self):
        """Different batch shapes across inputs → ValueError."""
        model = _scalar_model(n_in=2)
        with pytest.raises(ValueError, match="Inconsistent batch shapes"):
            model._arrange_inputs(
                {
                    "x0": torch.tensor([1.0, 2.0], dtype=torch.double),  # batch=2
                    "x1": torch.tensor([1.0, 2.0, 3.0], dtype=torch.double),  # batch=3
                }
            )

    def test_invalid_scalar_shape_raises(self):
        """Shape (3, 2) for a scalar input → ValueError."""
        model = _scalar_model(n_in=2)
        with pytest.raises(ValueError, match="unexpected shape"):
            model._arrange_inputs(
                {
                    "x0": torch.tensor(
                        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.double
                    ),
                }
            )

    def test_all_inputs_provided(self):
        """All inputs provided, no default fill needed."""
        model = _scalar_model(n_in=2)
        result = model._arrange_inputs(
            {
                "x0": torch.tensor([1.0, 2.0], dtype=torch.double),
                "x1": torch.tensor([3.0, 4.0], dtype=torch.double),
            }
        )
        assert result.shape == (2, 2)
        assert torch.allclose(
            result, torch.tensor([[1.0, 3.0], [2.0, 4.0]], dtype=torch.double)
        )

    def test_preserves_input_ordering(self):
        """Dict insertion order doesn't matter — variable list order is used."""
        model = _scalar_model(n_in=3)
        # Provide in reverse order
        result = model._arrange_inputs(
            {
                "x2": torch.tensor([30.0], dtype=torch.double),
                "x0": torch.tensor([10.0], dtype=torch.double),
            }
        )
        assert result.shape == (1, 3)
        assert result[0, 0].item() == 10.0  # x0
        assert result[0, 1].item() == 1.0  # x1 default
        assert result[0, 2].item() == 30.0  # x2


# =========================================================================
# _arrange_inputs — ND path
# =========================================================================


class TestArrangeInputsND:
    """Tests for _arrange_inputs with TorchNDVariable inputs."""

    def test_single_unbatched_nd(self):
        """Single unbatched array → (1, 1, *shape)."""
        model = _nd_model(array_shape=(2, 3), n_in=1)
        arr = torch.ones(2, 3)
        result = model._arrange_inputs({"arr_in0": arr})
        assert result.shape == (1, 1, 2, 3)

    def test_batched_nd(self):
        """Batched array (batch, *shape) → (batch, 1, *shape)."""
        model = _nd_model(array_shape=(2, 3), n_in=1)
        arr = torch.ones(5, 2, 3)
        result = model._arrange_inputs({"arr_in0": arr})
        assert result.shape == (5, 1, 2, 3)

    def test_multi_nd_inputs(self):
        """Two ND inputs → (batch, 2, *shape)."""
        model = _nd_model(array_shape=(4,), n_in=2)
        a0 = torch.ones(3, 4)
        a1 = torch.zeros(3, 4)
        result = model._arrange_inputs({"arr_in0": a0, "arr_in1": a1})
        assert result.shape == (3, 2, 4)
        assert torch.all(result[:, 0, :] == 1.0)
        assert torch.all(result[:, 1, :] == 0.0)

    def test_nd_wrong_sample_shape_raises(self):
        """Wrong trailing shape → ValueError."""
        model = _nd_model(array_shape=(2, 3), n_in=1)
        with pytest.raises(ValueError, match="expected sample shape"):
            model._arrange_inputs({"arr_in0": torch.ones(2, 4)})

    def test_nd_inconsistent_batch_raises(self):
        """Different batch sizes across ND inputs → ValueError."""
        model = _nd_model(array_shape=(4,), n_in=2)
        with pytest.raises(ValueError, match="inconsistent batch shapes"):
            model._arrange_inputs(
                {
                    "arr_in0": torch.ones(3, 4),
                    "arr_in1": torch.ones(5, 4),
                }
            )

    def test_nd_missing_input_uses_default_unbatched(self):
        """Missing ND input filled with default; unbatched → singleton batch."""
        model = _nd_model(array_shape=(4, 3), n_in=2)
        a0 = torch.ones(4, 3)
        result = model._arrange_inputs({"arr_in0": a0})
        assert result.shape == (1, 2, 4, 3)
        assert torch.all(result[:, 0, :, :] == 1.0)
        assert torch.all(result[:, 1, :, :] == 0.0)  # default

    def test_nd_missing_input_broadcasts_to_batch(self):
        """Missing ND input (singleton default) broadcasts to match batch."""
        model = _nd_model(array_shape=(4, 3), n_in=2)
        a0 = torch.ones(3, 4, 3)
        result = model._arrange_inputs({"arr_in0": a0})
        assert result.shape == (3, 2, 4, 3)
        assert torch.all(result[:, 0, :, :] == 1.0)
        assert torch.all(result[:, 1, :, :] == 0.0)  # default broadcast to batch=3


# =========================================================================
# _arrange_inputs — Mixed variable types
# =========================================================================


class TestArrangeInputsMixed:
    """Tests for the mixed scalar + ND error path."""

    def test_mixed_raises_not_implemented(self):
        input_vars = [
            TorchScalarVariable(name="s", default_value=1.0, value_range=(0.0, 5.0)),
            TorchNDVariable(name="a", shape=(3,), default_value=torch.zeros(3)),
        ]
        output_vars = [TorchScalarVariable(name="y")]
        model = TorchModel(
            model=_Identity(),
            input_variables=input_vars,
            output_variables=output_vars,
            fixed_model=False,
        )
        with pytest.raises(NotImplementedError, match="Mixing"):
            model._arrange_inputs({"s": torch.tensor(1.0)})


# =========================================================================
# _parse_outputs — Scalar
# =========================================================================


class TestParseOutputsScalar:
    """Tests for _parse_outputs with scalar output variables."""

    def test_0d_single_output(self):
        """0D tensor → dict with shape (1, 1)."""
        model = _scalar_model(n_out=1)
        result = model._parse_outputs(torch.tensor(5.0, dtype=torch.double))
        assert result["y0"].shape == (1, 1)
        assert result["y0"].item() == 5.0

    def test_1d_single_output_is_batch(self):
        """1D tensor (batch,) for single scalar output → (batch, 1)."""
        model = _scalar_model(n_out=1)
        result = model._parse_outputs(torch.tensor([1.0, 2.0, 3.0], dtype=torch.double))
        assert result["y0"].shape == (3, 1)

    def test_1d_multi_output_is_features(self):
        """1D tensor (features,) for multi scalar output → per-output (1, 1)."""
        model = _scalar_model(n_out=3)
        result = model._parse_outputs(
            torch.tensor([10.0, 20.0, 30.0], dtype=torch.double)
        )
        assert len(result) == 3
        for i, name in enumerate(["y0", "y1", "y2"]):
            assert result[name].shape == (1, 1)
            assert result[name].item() == (i + 1) * 10.0

    def test_2d_single_output(self):
        """2D tensor (batch, 1) for single scalar output → kept as-is."""
        model = _scalar_model(n_out=1)
        t = torch.tensor([[1.0], [2.0]], dtype=torch.double)
        result = model._parse_outputs(t)
        assert result["y0"].shape == (2, 1)

    def test_2d_multi_output(self):
        """2D tensor (batch, features) for multi scalar output → per-output (batch, 1)."""
        model = _scalar_model(n_out=2)
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.double)
        result = model._parse_outputs(t)
        assert result["y0"].shape == (3, 1)
        assert result["y1"].shape == (3, 1)
        assert torch.allclose(
            result["y0"], torch.tensor([[1.0], [3.0], [5.0]], dtype=torch.double)
        )
        assert torch.allclose(
            result["y1"], torch.tensor([[2.0], [4.0], [6.0]], dtype=torch.double)
        )

    def test_3d_single_output(self):
        """3D tensor (batch, samples, 1) for single scalar output."""
        model = _scalar_model(n_out=1)
        t = torch.ones(2, 3, 1, dtype=torch.double)
        result = model._parse_outputs(t)
        assert result["y0"].shape == (2, 3, 1)


# =========================================================================
# _parse_outputs — ND
# =========================================================================


class TestParseOutputsND:
    """Tests for _parse_outputs with NDVariable outputs."""

    def test_single_nd_output_squeezes_dim1(self):
        """(batch, 1, H, W) → squeezed to (batch, H, W)."""
        model = _nd_model(array_shape=(2, 3), n_out=1)
        t = torch.ones(4, 1, 2, 3)
        result = model._parse_outputs(t)
        assert result["arr_out0"].shape == (4, 2, 3)

    def test_single_nd_output_no_squeeze_needed(self):
        """(batch, H, W) with no extra dim 1 → unchanged."""
        model = _nd_model(array_shape=(2, 3), n_out=1)
        t = torch.ones(4, 2, 3)
        result = model._parse_outputs(t)
        # dim 1 is 2 (not 1), so squeeze(1) is a no-op
        assert result["arr_out0"].shape == (4, 2, 3)

    def test_single_nd_unbatched(self):
        """(1, 1, H, W) → squeeze dim 1 → (1, H, W)."""
        model = _nd_model(array_shape=(2, 3), n_out=1)
        t = torch.ones(1, 1, 2, 3)
        result = model._parse_outputs(t)
        assert result["arr_out0"].shape == (1, 2, 3)
