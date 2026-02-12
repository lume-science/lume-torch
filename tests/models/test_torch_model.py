import os
import random
from typing import Any, Union
from copy import deepcopy

import pytest

try:
    import torch
    from botorch.models.transforms.input import (
        AffineInputTransform,
        ReversibleInputTransform,
    )
    from lume_torch.models import TorchModel
    from lume_torch.variables import TorchScalarVariable

    torch.manual_seed(42)
except ImportError:
    pass

random.seed(42)


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


class TestTorchModel:
    def test_model_from_objects(
        self,
        california_model_info: dict[str, str],
        california_model_kwargs: dict[str, Union[list, dict, str]],
        california_variables: tuple[
            list[TorchScalarVariable], list[TorchScalarVariable]
        ],
        california_transformers: tuple[list, list],
        california_model,
    ):
        input_variables, output_variables = california_variables
        input_transformer, output_transformer = california_transformers

        assert isinstance(california_model, TorchModel)
        assert california_model.input_names == california_model_info["model_in_list"]
        assert california_model.output_names == california_model_info["model_out_list"]
        assert california_model.input_variables == input_variables
        assert california_model.output_variables == output_variables
        assert california_model.input_transformers == [input_transformer]
        assert california_model.output_transformers == [output_transformer]

    def test_model_from_yaml(self, rootdir: str, california_model):
        file = os.path.join(
            rootdir, "test_files", "california_regression", "torch_model.yml"
        )
        yaml_model = TorchModel.from_file(file)

        assert_model_equality(yaml_model, california_model)

    def test_model_as_yaml(self, rootdir: str, california_model):
        filename = "test_torch_model"
        file = f"{filename}.yml"
        california_model.dump(file)
        yaml_model = TorchModel(file)
        assert_model_equality(yaml_model, california_model)
        os.remove(file)
        os.remove(f"{filename}_model.pt")
        os.remove(f"{filename}_input_transformers_0.pt")
        os.remove(f"{filename}_output_transformers_0.pt")

    def test_input_validation(self, california_test_input_dict: dict, california_model):
        california_model.input_validation(california_test_input_dict)

    def test_output_validation(self, california_model):
        output_dict = {"MedHouseVal": torch.tensor([5.0, 3.1])}
        california_model.output_validation(output_dict)

    def test_precision(self, california_model):
        assert california_model.precision == "double"
        assert california_model.dtype == torch.double
        california_model.precision = "single"
        assert california_model.dtype == torch.float
        # set back to double
        california_model.precision = "double"

    def test_model_evaluate_single_sample(
        self, california_test_input_dict: dict, california_model
    ):
        results = california_model.evaluate(california_test_input_dict)

        assert isinstance(results["MedHouseVal"], torch.Tensor)
        assert torch.isclose(
            results["MedHouseVal"],
            torch.tensor(4.063651, dtype=results["MedHouseVal"].dtype),
        )

    def test_model_evaluate_n_samples(
        self, california_test_input_tensor, california_model
    ):
        test_dict = {
            key: california_test_input_tensor[:, idx]
            for idx, key in enumerate(california_model.input_names)
        }
        results = california_model.evaluate(test_dict)
        target_tensor = torch.tensor(
            [4.063651, 2.7774928, 2.792812], dtype=results["MedHouseVal"].dtype
        )

        assert torch.all(torch.isclose(results["MedHouseVal"], target_tensor))

    def test_model_evaluate_batch_n_samples(
        self,
        california_test_input_tensor,
        california_model,
    ):
        # model should be able to handle input of shape [n_batch, n_samples, n_dim]
        input_dict = {
            key: california_test_input_tensor[:, idx]
            .unsqueeze(-1)
            .unsqueeze(1)
            .repeat((1, 3, 1))
            for idx, key in enumerate(california_model.input_names)
        }
        results = california_model.evaluate(input_dict)

        # output shape should be [n_batch, n_samples]
        assert tuple(results["MedHouseVal"].shape) == (3, 3)

    def test_model_evaluate_raw(
        self,
        california_test_input_dict: dict,
        california_model_kwargs: dict[str, Union[list, dict, str]],
    ):
        kwargs = deepcopy(california_model_kwargs)
        kwargs["output_format"] = "raw"
        california_model = TorchModel(**kwargs)
        float_dict = {
            key: value.item() for key, value in california_test_input_dict.items()
        }
        results = california_model.evaluate(float_dict)

        assert isinstance(results["MedHouseVal"], float)
        assert results["MedHouseVal"] == pytest.approx(4.063651)

    def test_model_evaluate_shuffled_input(
        self, california_test_input_dict: dict, california_model
    ):
        shuffled_input = deepcopy(california_test_input_dict)
        item_list = list(shuffled_input.items())
        random.shuffle(item_list)
        shuffled_input = dict(item_list)
        results = california_model.evaluate(shuffled_input)

        assert isinstance(results["MedHouseVal"], torch.Tensor)
        assert torch.isclose(
            results["MedHouseVal"],
            torch.tensor(4.063651, dtype=results["MedHouseVal"].dtype),
        )

    @pytest.mark.parametrize(
        "test_idx,expected", [(0, 4.063651), (1, 2.7774928), (2, 2.792812)]
    )
    def test_model_evaluate_different_values(
        self,
        test_idx: int,
        expected: float,
        california_test_input_tensor,
        california_model,
    ):
        input_dict = {
            key: california_test_input_tensor[test_idx][idx]
            for idx, key in enumerate(california_model.input_names)
        }
        results = california_model.evaluate(input_dict)

        assert results["MedHouseVal"].item() == pytest.approx(expected)

    def test_model_evaluate_with_no_output_transformers(
        self,
        california_test_input_dict: dict,
        california_model_kwargs: dict[str, Union[list, dict, str]],
    ):
        kwargs = deepcopy(california_model_kwargs)
        kwargs["output_transformers"] = []
        model = TorchModel(**kwargs)
        results = model.evaluate(california_test_input_dict)

        assert torch.isclose(
            results["MedHouseVal"],
            torch.tensor(1.8523695, dtype=results["MedHouseVal"].dtype),
        )

    def test_differentiability(
        self,
        california_test_input_dict: dict,
        california_model_kwargs: dict[str, Any],
    ):
        kwargs = deepcopy(california_model_kwargs)
        kwargs["model"].train().requires_grad_(True)
        model = TorchModel(**kwargs, fixed_model=False)
        parameters_with_requires_grad = []
        for name, param in model.model.named_parameters():
            if param.requires_grad:
                parameters_with_requires_grad.append(name)
        criterion = torch.nn.MSELoss()

        assert model.model.training
        assert len(parameters_with_requires_grad) == 8
        output_dict = model.evaluate(california_test_input_dict)
        outputs = torch.stack([v for k, v in output_dict.items()])
        loss = criterion(outputs, torch.zeros(outputs.shape, dtype=outputs.dtype))
        loss.backward()

    def test_update_input_variables_to_transformer(self, california_model):
        model = deepcopy(california_model)
        input_variables = model.input_variables

        def get_x_limits(v):
            x_limits = {
                "min": torch.tensor(
                    [var.value_range[0] for var in v], dtype=california_model.dtype
                ),
                "max": torch.tensor(
                    [var.value_range[1] for var in v], dtype=california_model.dtype
                ),
                "default": torch.tensor(
                    [var.default_value for var in v], dtype=california_model.dtype
                ),
            }
            return x_limits

        x_lim = get_x_limits(input_variables)
        x_lim_nn = {key: model._transform_inputs(x_lim[key]) for key in x_lim.keys()}
        # add new transformer
        d = len(model.input_variables)
        new_transformer = AffineInputTransform(
            d=d, offset=torch.rand(d), coefficient=1.0 + torch.rand(d)
        )
        model.insert_input_transformer(
            new_transformer, loc=len(model.input_transformers)
        )
        updated_input_variables = model.update_input_variables_to_transformer(
            len(model.input_transformers) - 1
        )
        model.input_variables = updated_input_variables
        # compute new NN limits
        x_lim_updated = get_x_limits(updated_input_variables)
        x_lim_nn_updated = {
            key: model._transform_inputs(x_lim_updated[key])
            for key in x_lim_updated.keys()
        }

        for key in x_lim_nn.keys():
            assert torch.all(
                torch.isclose(x_lim_nn[key], x_lim_nn_updated[key], atol=1e-6)
            )

    def test_jit_serialization(self, california_model):
        filename = "test_torch_model"
        file = f"{filename}.yml"

        # Test saving with save_jit=True
        california_model.dump(file, save_jit=True)
        assert os.path.exists(file)
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


class DivideByFiveCallable(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / 5.0


def divide_by_five(x: torch.Tensor):
    return x / 5.0


class MultiplyByFiveCallable(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 5.0


def multiply_by_five(x: torch.Tensor):
    return x * 5.0


class DivideByFiveReversibleTransform(ReversibleInputTransform):
    reverse = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _transform(self, x):
        return x / 5.0

    def _untransform(self, x):
        return x * 5.0


class TestTransformers:
    @pytest.mark.parametrize(
        "input_transformer_type", ["module", "function", "reversible", "linear"]
    )
    def test_callable_input_transformers(
        self, input_transformer_type, california_model, california_test_input_dict
    ):
        # for the purpose of the test, we multiply the contents of the input_dictionary by 5
        # so we can add an input transform that needs to divide by 5
        input_dict = {
            key: value * 5 for key, value in california_test_input_dict.items()
        }
        if input_transformer_type == "module":
            input_transformer = DivideByFiveCallable()
        elif input_transformer_type == "function":
            input_transformer = divide_by_five
        elif input_transformer_type == "reversible":
            input_transformer = DivideByFiveReversibleTransform()
        elif input_transformer_type == "linear":
            input_transformer = torch.nn.Linear(
                len(california_model.input_names), len(california_model.input_names)
            )
        else:
            raise ValueError(f"Unknown input transformer type {input_transformer_type}")

        california_model.insert_input_transformer(input_transformer, 0)
        results = california_model.evaluate(input_dict)
        assert len(results) == len(california_model.output_names)

    @pytest.mark.parametrize(
        "output_transformer_type", ["module", "function", "reversible", "linear"]
    )
    def test_callable_output_transformers(
        self, output_transformer_type, california_model, california_test_input_dict
    ):
        # for the purpose of the test, we multiply the contents of the input_dictionary by 5
        # so we can add an input transform that needs to divide by 5
        if output_transformer_type == "module":
            output_transformer = MultiplyByFiveCallable()
        elif output_transformer_type == "function":
            output_transformer = multiply_by_five
        elif output_transformer_type == "reversible":
            output_transformer = DivideByFiveReversibleTransform()
        elif output_transformer_type == "linear":
            output_transformer = torch.nn.Linear(
                len(california_model.output_names), len(california_model.output_names)
            )
        else:
            raise ValueError(
                f"Unknown input transformer type {output_transformer_type}"
            )

        california_model.insert_output_transformer(output_transformer, 0)
        results = california_model.evaluate(california_test_input_dict)
        assert len(results) == len(california_model.output_names)

    @pytest.mark.parametrize("input_transformer_type", ["function", "module"])
    def test_update_input_variables_to_transformer_not_implemented(
        self, input_transformer_type, california_model
    ):
        if input_transformer_type == "module":
            input_transformer = DivideByFiveCallable()
        elif input_transformer_type == "function":
            input_transformer = divide_by_five
        else:
            raise ValueError(f"Unknown input transformer type {input_transformer_type}")

        california_model.insert_input_transformer(input_transformer, 0)
        with pytest.raises(NotImplementedError):
            california_model.update_input_variables_to_transformer(0)
