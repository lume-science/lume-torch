import os
import json
from typing import Any, Union

import pytest

from lume_torch.utils import variables_from_yaml
from lume_torch.variables import ScalarVariable, DistributionVariable

try:
    import torch
    from botorch.models.transforms.input import AffineInputTransform  # noqa: F401
    from botorch.models.transforms.outcome import Standardize  # noqa: F401
    from botorch.models import MultiTaskGP, SingleTaskGP
    from lume_torch.models import TorchModel, TorchModule
except (ModuleNotFoundError, ImportError):
    pass


@pytest.fixture(scope="session")
def rootdir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


# TorchModel fixtures
@pytest.fixture(scope="session")
def simple_variables() -> dict[str, Union[list[ScalarVariable], list[ScalarVariable]]]:
    input_variables = [
        ScalarVariable(name="input1", default_value=1.0, value_range=(0.0, 5.0)),
        ScalarVariable(name="input2", default_value=2.0, value_range=(1.0, 3.0)),
    ]
    output_variables = [ScalarVariable(name="output1"), ScalarVariable(name="output2")]
    return {"input_variables": input_variables, "output_variables": output_variables}


@pytest.fixture(scope="module")
def california_model_info(rootdir) -> dict[str, str]:
    try:
        with open(
            f"{rootdir}/test_files/california_regression/model_info.json", "r"
        ) as f:
            model_info = json.load(f)
        return model_info
    except FileNotFoundError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def california_variables(rootdir) -> tuple[list[ScalarVariable], list[ScalarVariable]]:
    try:
        file = f"{rootdir}/test_files/california_regression/variables.yml"
        input_variables, output_variables = variables_from_yaml(file)
        return input_variables, output_variables
    except FileNotFoundError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def california_transformers(rootdir):
    botorch = pytest.importorskip("botorch")

    try:
        with open(
            f"{rootdir}/test_files/california_regression/normalization.json", "r"
        ) as f:
            normalizations = json.load(f)
    except FileNotFoundError as e:
        pytest.skip(str(e))

    input_transformer = botorch.models.transforms.input.AffineInputTransform(
        len(normalizations["x_mean"]),
        coefficient=torch.tensor(normalizations["x_scale"]),
        offset=torch.tensor(normalizations["x_mean"]),
    )
    output_transformer = botorch.models.transforms.input.AffineInputTransform(
        len(normalizations["y_mean"]),
        coefficient=torch.tensor(normalizations["y_scale"]),
        offset=torch.tensor(normalizations["y_mean"]),
    )
    return input_transformer, output_transformer


@pytest.fixture(scope="module")
def california_model_kwargs(
    rootdir,
    california_model_info,
    california_variables,
    california_transformers,
) -> dict[str, Any]:
    _ = pytest.importorskip("botorch")

    input_variables, output_variables = california_variables
    input_transformer, output_transformer = california_transformers
    model_kwargs = {
        "model": torch.load(
            f"{rootdir}/test_files/california_regression/model.pt", weights_only=False
        ),
        "input_variables": input_variables,
        "output_variables": output_variables,
        "input_transformers": [input_transformer],
        "output_transformers": [output_transformer],
        "output_format": "tensor",
    }
    return model_kwargs


@pytest.fixture(scope="module")
def california_test_input_tensor(rootdir: str):
    torch = pytest.importorskip("torch")

    try:
        test_input_tensor = torch.load(
            f"{rootdir}/test_files/california_regression/test_input_tensor.pt",
            weights_only=False,
        )
    except FileNotFoundError as e:
        pytest.skip(str(e))
    return test_input_tensor


@pytest.fixture(scope="module")
def california_test_input_dict(
    california_test_input_tensor, california_model_info
) -> dict:
    pytest.importorskip("botorch")

    test_input_dict = {
        key: california_test_input_tensor[0, idx]
        for idx, key in enumerate(california_model_info["model_in_list"])
    }
    return test_input_dict


@pytest.fixture(scope="module")
def california_model(california_model_kwargs):
    _ = pytest.importorskip("botorch")

    return TorchModel(**california_model_kwargs)


@pytest.fixture(scope="module")
def california_module(california_model):
    _ = pytest.importorskip("botorch")

    return TorchModule(model=california_model)


# GPModel fixtures
@pytest.fixture(scope="session")
def gp_variables() -> dict[
    str, Union[list[ScalarVariable], list[DistributionVariable]]
]:
    input_variables = [ScalarVariable(name="input")]
    output_variables = [
        DistributionVariable(name="output1"),
        DistributionVariable(name="output2"),
    ]
    return input_variables, output_variables


# SingleTask GP
@pytest.fixture(scope="module")
def single_task_gp_transformers(rootdir):
    input_transformer = torch.load(
        f"{rootdir}/test_files/single_task_gp/input_transformers.pt", weights_only=False
    )
    output_transformer = torch.load(
        f"{rootdir}/test_files/single_task_gp/output_transformers.pt",
        weights_only=False,
    )
    return input_transformer, output_transformer


@pytest.fixture(scope="module")
def single_task_gp_model_kwargs(
    rootdir,
    gp_variables,
    single_task_gp_transformers,
) -> dict[str, Any]:
    _ = pytest.importorskip("botorch")

    input_variables, output_variables = gp_variables
    input_transformer, output_transformer = single_task_gp_transformers
    model_kwargs = {
        "model": torch.load(
            f"{rootdir}/test_files/single_task_gp/model.pt", weights_only=False
        ),
        "input_variables": input_variables,
        "output_variables": output_variables,
        "input_transformers": [input_transformer],
        "output_transformers": [output_transformer],
    }
    return model_kwargs


# MultiTask GP
@pytest.fixture(scope="module")
def multi_task_gp_transformers(rootdir):
    input_transformer = torch.load(
        f"{rootdir}/test_files/multi_task_gp/input_transformers.pt", weights_only=False
    )
    output_transformer = torch.load(
        f"{rootdir}/test_files/multi_task_gp/output_transformers.pt", weights_only=False
    )
    return input_transformer, output_transformer


@pytest.fixture(scope="module")
def multi_task_gp_model_kwargs(
    rootdir,
    gp_variables,
    multi_task_gp_transformers,
) -> dict[str, Any]:
    _ = pytest.importorskip("botorch")

    input_variables, output_variables = gp_variables
    input_transformer, output_transformer = multi_task_gp_transformers
    model_kwargs = {
        "model": torch.load(
            f"{rootdir}/test_files/multi_task_gp/model.pt", weights_only=False
        ),
        "input_variables": input_variables,
        "output_variables": output_variables,
        "input_transformers": [input_transformer],
        "output_transformers": [output_transformer],
    }
    return model_kwargs


# ModelListGP
@pytest.fixture(scope="module")
def create_multi_task_gp():
    _ = pytest.importorskip("botorch")
    tkwargs = {"dtype": torch.double}
    train_x_raw, train_y = get_random_data(
        batch_shape=torch.Size(), m=1, n=10, **tkwargs
    )
    task_idx = torch.cat(
        [torch.ones(5, 1, **tkwargs), torch.zeros(5, 1, **tkwargs)], dim=0
    )
    train_x = torch.cat([train_x_raw, task_idx], dim=-1)
    # single output
    model = MultiTaskGP(
        train_X=train_x,
        train_Y=train_y,
        task_feature=-1,
        output_tasks=[0],
    )
    # multi output
    model2 = MultiTaskGP(
        train_X=train_x,
        train_Y=train_y,
        task_feature=-1,
    )
    return model, model2, train_x_raw


@pytest.fixture(scope="module")
def create_single_task_gp():
    tkwargs = {"dtype": torch.double}
    train_x1, train_y1 = get_random_data(batch_shape=torch.Size(), m=1, n=10, **tkwargs)
    model1 = SingleTaskGP(train_X=train_x1, train_Y=train_y1, outcome_transform=None)
    model1.to(**tkwargs)
    test_x = torch.tensor([[0.25], [0.75]], **tkwargs)
    return model1, test_x


@pytest.fixture(scope="module")
def create_single_task_gp_w_transform():
    tkwargs = {"dtype": torch.double}
    train_x1, train_y1 = get_random_data(batch_shape=torch.Size(), m=1, n=10, **tkwargs)
    input_transform = AffineInputTransform(
        1,
        coefficient=train_x1.std(dim=0),
        offset=train_y1.mean(dim=0),
    )
    output_transform = Standardize(m=1)
    model1 = SingleTaskGP(
        train_X=train_x1,
        train_Y=train_y1,
        input_transform=input_transform,
        outcome_transform=output_transform,
    )
    model1.to(**tkwargs)
    test_x = torch.tensor([[0.25], [0.75]], **tkwargs)
    return model1, test_x, input_transform, output_transform


def get_random_data(
    batch_shape: torch.Size, m: int, d: int = 1, n: int = 10, **tkwargs
):
    r"""Generate random data for testing purposes.

    Args:
        batch_shape: The batch shape of the data.
        m: The number of outputs.
        d: The dimension of the input.
        n: The number of data points.
        tkwargs: `device` and `dtype` tensor constructor kwargs.

    Returns:
        A tuple `(train_X, train_Y)` with randomly generated training data.
    """
    rep_shape = batch_shape + torch.Size([1, 1])
    train_x = torch.stack(
        [torch.linspace(0, 0.95, n, **tkwargs) for _ in range(d)], dim=-1
    )
    train_x = train_x + 0.05 * torch.rand_like(train_x).repeat(rep_shape)
    train_x[0] += 0.02  # modify the first batch
    train_y = torch.sin(train_x[..., :1] * (2 * torch.pi))
    train_y = train_y + 0.2 * torch.randn(n, m, **tkwargs).repeat(rep_shape)
    return train_x, train_y
