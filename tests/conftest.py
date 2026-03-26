import os
import json
from typing import Any, Union

import pytest

from lume_torch.utils import variables_from_yaml
from lume_torch.variables import (
    TorchScalarVariable,
    TorchNDVariable,
    DistributionVariable,
)

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
def simple_variables() -> dict[
    str, Union[list[TorchScalarVariable], list[TorchScalarVariable]]
]:
    input_variables = [
        TorchScalarVariable(name="input1", default_value=1.0, value_range=(0.0, 5.0)),
        TorchScalarVariable(name="input2", default_value=2.0, value_range=(1.0, 3.0)),
    ]
    output_variables = [
        TorchScalarVariable(name="output1"),
        TorchScalarVariable(name="output2"),
    ]
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
def california_variables(
    rootdir,
) -> tuple[list[TorchScalarVariable], list[TorchScalarVariable]]:
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
        ).unsqueeze(-1)
    except FileNotFoundError as e:
        pytest.skip(str(e))
    return test_input_tensor


@pytest.fixture(scope="module")
def california_test_input_dict(
    california_test_input_tensor, california_model_info
) -> dict:
    pytest.importorskip("botorch")

    test_input_dict = {
        key: california_test_input_tensor[:, idx]
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


@pytest.fixture(scope="module")
def nd_model_and_data():
    """A minimal TorchModel with a single TorchNDVariable input and output.

    The model is a single-layer linear network mapping (C, H, W) -> (C, H, W)
    via a learned weight on the flattened input = identity-like transform.
    For test simplicity we use shape (2, 3) arrays.
    """
    array_shape = (2, 3)
    n_elements = array_shape[0] * array_shape[1]  # 6

    default_array = torch.zeros(array_shape, dtype=torch.float32)

    input_var = TorchNDVariable(
        name="array_in",
        shape=array_shape,
        default_value=default_array,
    )
    output_var = TorchNDVariable(
        name="array_out",
        shape=array_shape,
    )

    # Simple linear model: flatten -> linear -> reshape
    class FlatLinear(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.linear = torch.nn.Linear(n, n, bias=False)
            # Initialise to identity so output == input (easy to verify)
            torch.nn.init.eye_(self.linear.weight)

        def forward(self, x):
            batch = x.shape[0]
            n_arrays = x.shape[1]
            flat = x.reshape(batch, n_arrays, -1)  # (batch, 1, 6)
            out = self.linear(flat)  # (batch, 1, 6)
            return out.reshape(batch, n_arrays, *array_shape)  # (batch, 1, 2, 3)

    nn_model = FlatLinear(n_elements)

    model = TorchModel(
        model=nn_model,
        input_variables=[input_var],
        output_variables=[output_var],
        precision="single",
        fixed_model=False,  # avoid grad-deactivation for the identity check
    )

    # A batch of 3 test arrays, each shape (2, 3)
    test_input = torch.arange(n_elements, dtype=torch.float32).reshape(array_shape)
    batch_input = test_input.unsqueeze(0).repeat(3, 1, 1)  # (3, 2, 3)

    return model, batch_input


@pytest.fixture(scope="module")
def mixed_model_and_data():
    """A TorchModel with mixed TorchScalarVariable and TorchNDVariable inputs/outputs.

    Inputs:  2 TorchScalarVariables (x1, x2) + 1 TorchNDVariable (image, shape (2, 3))
    Outputs: 1 TorchScalarVariable (y_scalar) + 1 TorchNDVariable (y_image, shape (2, 3))

    The underlying nn.Module:
    - scalar_head: Linear(2, 1, bias=False) with ones weights (output = sum of inputs)
    - img_linear:  Linear(6, 6, bias=False) with eye weights (identity on flat image)
    """
    array_shape = (2, 3)
    n_elements = array_shape[0] * array_shape[1]

    input_variables = [
        TorchScalarVariable(name="x1", default_value=0.0, value_range=(-1.0, 1.0)),
        TorchScalarVariable(name="x2", default_value=0.0, value_range=(-1.0, 1.0)),
        TorchNDVariable(
            name="image",
            shape=array_shape,
            default_value=torch.zeros(array_shape, dtype=torch.float32),
        ),
    ]
    output_variables = [
        TorchScalarVariable(name="y_scalar"),
        TorchNDVariable(name="y_image", shape=array_shape),
    ]

    class MixedModel(torch.nn.Module):
        def __init__(self, n_scalars, n_elements):
            super().__init__()
            self.scalar_head = torch.nn.Linear(n_scalars, 1, bias=False)
            torch.nn.init.ones_(
                self.scalar_head.weight
            )  # output = sum of scalar inputs
            self.img_linear = torch.nn.Linear(n_elements, n_elements, bias=False)
            torch.nn.init.eye_(self.img_linear.weight)  # identity on images

        def forward(self, scalars, images):
            # scalars: (batch, n_scalars), images: (batch, n_nd_vars, H, W)
            scalar_out = self.scalar_head(scalars)  # (batch, 1)
            batch, n_imgs = images.shape[0], images.shape[1]
            rest = images.shape[2:]
            flat = images.reshape(batch, n_imgs, -1)
            img_out = self.img_linear(flat).reshape(batch, n_imgs, *rest)
            return scalar_out, img_out

    nn_model = MixedModel(n_scalars=2, n_elements=n_elements)
    model = TorchModel(
        model=nn_model,
        input_variables=input_variables,
        output_variables=output_variables,
        precision="single",
        fixed_model=False,
    )
    return model, 3  # batch_size=3


@pytest.fixture(scope="session")
def gp_variables() -> dict[
    str, Union[list[TorchScalarVariable], list[DistributionVariable]]
]:
    input_variables = [TorchScalarVariable(name="input")]
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
