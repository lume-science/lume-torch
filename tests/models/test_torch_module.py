import os
import random
import warnings
from copy import deepcopy

import pytest

import torch
from botorch.models import SingleTaskGP
from lume_torch.models import TorchModel, TorchModule

torch.manual_seed(42)


def assert_california_module_result(result: torch.Tensor, idx=None):
    target = torch.tensor(
        [4.0636503726, 2.7774916915, 2.7928111793], dtype=result.dtype
    )
    if idx is None:
        assert all(torch.isclose(result, target))
    else:
        assert torch.isclose(result, target[idx])


def assert_model_equality(m1: TorchModel, m2: TorchModel):
    assert m1.input_variables == m2.input_variables
    assert m1.output_variables == m2.output_variables
    assert str(m1.model.state_dict()) == str(m2.model.state_dict())
    for attr in ["input_transformers", "output_transformers"]:
        m1_transformers, m2_transformers = getattr(m1, attr), getattr(m2, attr)
        assert len(m1_transformers) == len(m2_transformers)
        for i, t in enumerate(m1_transformers):
            if isinstance(t, torch.nn.Module):
                assert isinstance(m2_transformers[i], torch.nn.Module)
                assert str(t.state_dict()) == str(m2_transformers[i].state_dict())
    assert m1.output_format == m2.output_format
    assert m1.device == m2.device
    assert m1.fixed_model == m2.fixed_model


def assert_module_equality(m1: TorchModule, m2: TorchModule):
    assert_model_equality(m1.model, m2.model)
    assert m1.input_order == m2.input_order
    assert m1.output_order == m2.output_order


class TestTorchModule:
    def test_module_initialization(self, california_model):
        lume_module = TorchModule(model=california_model)
        parameters_with_requires_grad = [
            param for param in lume_module.parameters() if param.requires_grad
        ]

        # gradients should be deactivated and module in eval mode
        assert not parameters_with_requires_grad
        assert not lume_module.training

    def test_module_parameters_match_model(self, california_model):
        california_module = TorchModule(model=california_model)
        params_match = [
            a == b
            for a, b in zip(
                california_module.parameters(), california_model.model.parameters()
            )
        ]

        assert len(list(california_module.parameters())) == len(
            list(california_model.model.parameters())
        )
        for param_match in params_match:
            if isinstance(param_match, torch.Tensor):
                assert torch.all(param_match)
            else:
                assert param_match

    def test_module_train_and_eval_mode(self, california_module):
        assert california_module.training is False
        assert california_module._model.model.training is False
        california_module.train()
        assert california_module.training is True
        assert california_module._model.model.training is True
        california_module.eval()
        assert california_module.training is False
        assert california_module._model.model.training is False

    def test_module_differentiability(
        self, california_test_input_tensor, california_module
    ):
        lume_module = deepcopy(california_module)
        lume_module.train()
        lume_module.requires_grad_(True)
        parameters_with_requires_grad = [
            param for param in lume_module.parameters() if param.requires_grad
        ]
        criterion = torch.nn.MSELoss()

        assert lume_module.training
        for param in lume_module.parameters():
            assert param.requires_grad
        for param in lume_module._model.model.parameters():
            assert param.requires_grad
        assert len(parameters_with_requires_grad) == 8
        outputs = lume_module(california_test_input_tensor)
        loss = criterion(outputs, torch.zeros(outputs.shape, dtype=outputs.dtype))
        loss.backward()

    def test_module_from_yaml(self, rootdir: str, california_module):
        file = os.path.join(
            rootdir, "test_files", "california_regression", "torch_module.yml"
        )
        yaml_module = TorchModule(file)

        assert_module_equality(yaml_module, california_module)

    def test_module_as_yaml(self, rootdir: str, california_module):
        filename = "test_torch_module"
        file = f"{filename}.yml"
        california_module.dump(file)
        yaml_module = TorchModule(file)
        assert_module_equality(yaml_module, california_module)
        os.remove(file)
        os.remove(f"{filename}_model.pt")
        os.remove(f"{filename}_input_transformers_0.pt")
        os.remove(f"{filename}_output_transformers_0.pt")

    def test_module_as_jit(self, rootdir: str, california_module):
        filename = "test_torch_module"
        file = f"{filename}.yml"

        # Test saving with save_jit=True
        california_module.dump(file, save_jit=True)
        assert os.path.exists(f"{filename}_model.jit")

        # Test loading the JIT model
        try:
            loaded_model = torch.jit.load(f"{filename}_model.jit")
            assert loaded_model is not None
        except Exception as e:
            pytest.fail(f"Failed to load JIT model: {e}")

        os.remove(file)
        os.remove(f"{filename}_model.pt")
        os.remove(f"{filename}_model.jit")
        os.remove(f"{filename}_input_transformers_0.pt")
        os.remove(f"{filename}_output_transformers_0.pt")

    def test_module_call_single_input(
        self, california_test_input_tensor, california_model
    ):
        lume_module = TorchModule(
            model=california_model,
            input_order=[california_model.input_names[0]],
        )
        input_tensor = deepcopy(
            california_test_input_tensor[:, 0].unsqueeze(-1)
        )  # shape (3, 1)
        result = lume_module(input_tensor)
        target = torch.tensor(
            [3.5094612847, 1.7297480438, 2.7042855903], dtype=result.dtype
        )

        assert tuple(result.size()) == (3,)
        assert all(torch.isclose(result, target))

    def test_module_call_single_input_bad_shape(
        self, california_test_input_tensor, california_model
    ):
        lume_module = TorchModule(
            model=california_model,
            input_order=[california_model.input_names[0]],
        )
        # Create a 1D tensor by squeezing - this should fail validation
        input_tensor = deepcopy(
            california_test_input_tensor[:, 0].squeeze()
        )  # shape (3,)

        with pytest.raises(ValueError):
            lume_module(input_tensor)

    # TODO: check this test
    def test_tensor_to_dictionary_2d_single_feature_uses_old_format(
        self, california_test_input_tensor, california_model
    ):
        """Shape (B, 1) with 1 feature must use the old-format path (ndim < 3),
        not the new scalar format, and produce the same output as shape (B, 1, 1)."""
        lume_module = TorchModule(
            model=california_model,
            input_order=[california_model.input_names[0]],
        )
        # 2D old-format: (batch, n_features) = (3, 1) — ndim=2, last dim=1
        input_2d = california_test_input_tensor[:, 0, 0].unsqueeze(-1)  # (3, 1)
        # 3D new-format: (batch, n_features, 1) = (3, 1, 1) — ndim=3
        input_3d = input_2d.unsqueeze(-1)  # (3, 1, 1)

        result_2d = lume_module(input_2d)
        result_3d = lume_module(input_3d)

        assert tuple(result_2d.shape) == (3,)
        assert torch.all(torch.isclose(result_2d, result_3d))

    def test_module_call_single_sample(
        self, california_test_input_tensor, california_module
    ):
        idx = 0
        input_tensor = deepcopy(california_test_input_tensor[idx, :]).unsqueeze(
            0
        )  # shape (1,8)
        result = california_module(input_tensor)

        assert tuple(result.size()) == (1,)
        assert_california_module_result(result, idx=idx)

    def test_module_call_n_samples(
        self, california_test_input_tensor, california_module
    ):
        result = california_module(california_test_input_tensor)

        assert tuple(result.size()) == (3,)
        assert_california_module_result(result)

    def test_module_reordered_inputs(
        self, california_test_input_tensor, california_model
    ):
        # shuffle the input names and the values associated with them
        idx = 0
        shuffled_inputs = list(
            zip(california_model.input_names, california_test_input_tensor[idx])
        )
        random.shuffle(shuffled_inputs)
        input_names = [shuffled_inputs[i][0] for i in range(len(shuffled_inputs))]
        input_tensor = torch.tensor(
            [shuffled_inputs[i][1] for i in range(len(shuffled_inputs))]
        ).reshape(1, -1)
        output_names = deepcopy(california_model.output_names)
        lume_module = TorchModule(
            model=california_model,
            input_order=input_names,
            output_order=output_names,
        )
        result = lume_module(input_tensor)

        assert tuple(result.size()) == (1,)
        assert_california_module_result(result, idx=idx)

    def test_module_call_manipulate_output(
        self, california_test_input_tensor, california_model
    ):
        n = 2
        output_order = deepcopy(california_model.output_names)
        output_order.append(f"MedHouseVal_x{n}")

        class ExampleTorchModule(TorchModule):
            def __init__(self, factor: int, **kwargs):
                super().__init__(**kwargs)
                self.factor = factor

            def manipulate_output(self, y_model: dict[str, torch.Tensor]):
                y_model[self.output_order[-1]] = y_model["MedHouseVal"] * self.factor
                return y_model

        lume_module = ExampleTorchModule(
            factor=n,
            model=california_model,
            input_order=california_model.input_names,
            output_order=output_order,
        )
        input_tensor = deepcopy(california_test_input_tensor)
        result = lume_module(input_tensor)

        assert tuple(result.size()) == (3, 2)
        assert_california_module_result(result[:, 0])
        assert_california_module_result(result[:, 1] / n)

    def test_module_call_batch_n_samples(
        self, california_test_input_tensor, california_module
    ):
        # module should be able to handle input of shape [n_batch, n_samples, n_features, 1]
        n_batch = 5
        input_tensor = california_test_input_tensor.unsqueeze(0).repeat(
            (n_batch, 1, 1, 1)
        )
        result = california_module(input_tensor)

        assert tuple(result.shape) == (n_batch, 3)
        for i in range(n_batch):
            assert_california_module_result(result[i])

    # TODO: check dimensions
    def test_module_as_gp_prior_mean(
        self, california_test_input_tensor, california_module
    ):
        # Squeeze trailing dimension for BoTorch compatibility: (3, 8, 1) -> (3, 8)
        train_x = california_test_input_tensor.squeeze(-1).double()
        train_y = california_module(train_x).unsqueeze(-1)
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore"
            )  # ignore warning that input data is not standardized
            gp = SingleTaskGP(
                train_x, train_y, mean_module=california_module, outcome_transform=None
            )
        x_lim = torch.stack(
            [torch.min(train_x, dim=0).values, torch.max(train_x, dim=0).values]
        )
        x_i = [
            x_lim[0, i] + (x_lim[1, i] - x_lim[0, i]) * torch.rand(size=(2,))
            for i in range(x_lim.shape[-1])
        ]
        test_x = torch.cartesian_prod(*x_i)
        test_y = california_module(test_x)
        post = gp.posterior(test_x)
        mean = post.mean.squeeze()

        assert tuple(mean.shape) == (test_x.shape[0],)
        assert tuple(test_y.shape) == (test_x.shape[0],)
        # GP should just predict the prior mean values for the posterior
        assert all(torch.isclose(mean, test_y))


class TestTorchModuleND:
    def test_nd_module_forward_single_batch(self, nd_model_and_data):
        model, batch_input = nd_model_and_data
        module = TorchModule(model=model)
        # Input: (batch, num_vars, *array_shape) = (3, 1, 2, 3)
        x = batch_input.unsqueeze(1)  # (3, 1, 2, 3)
        result = module(x)
        # Output: (batch, num_outputs, *array_shape) = (3, 1, 2, 3)
        assert tuple(result.shape) == (3, 1, 2, 3)
        # Identity model: output should equal input
        assert torch.allclose(result[:, 0, :, :], batch_input, atol=1e-5)

    def test_nd_module_validate_input_too_few_dims(self, nd_model_and_data):
        model, batch_input = nd_model_and_data
        module = TorchModule(model=model)
        # array_shape=(2,3) so min_ndim=4; (3,2,3) is 3D — too few
        with pytest.raises(ValueError, match="at least 4"):
            module(batch_input)  # shape (3, 2, 3) — 3D, too few
        with pytest.raises(ValueError, match="at least 4"):
            module(batch_input[0])  # shape (2, 3) — 2D, too few

    def test_nd_all_nd_property(self, nd_model_and_data):
        model, _ = nd_model_and_data
        module = TorchModule(model=model)
        assert module._nd_inputs is True
        assert module._nd_outputs is True
        assert module._scalar_inputs is False
        assert module._scalar_outputs is False

    def test_scalar_all_scalars_property(self, california_model):
        module = TorchModule(model=california_model)
        assert module._scalar_inputs is True
        assert module._scalar_outputs is True
        assert module._nd_inputs is False
        assert module._nd_outputs is False
