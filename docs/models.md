# Models

LUME-torch provides several model classes for different use cases, from simple custom models to advanced probabilistic models.

## Overview

All LUME-torch models inherit from `LUMETorch` and provide:

- Consistent input/output interface using dictionaries
- Variable validation
- Configuration file support (YAML)
- Serialization/deserialization

## Base Model

### LUMETorch

The foundation for all LUME models.

::: lume_torch.base.LUMETorch
    options:
        show_root_heading: true
        show_source: true
        members:
            - __init__
            - __call__
            - evaluate
            - _evaluate
            - input_validation
            - output_validation
            - dump
            - set

### Creating Custom Models

To create a custom model, inherit from `LUMETorch` and implement `_evaluate`:

```python
from lume_torch.base import LUMETorch
from lume_torch.variables import ScalarVariable


class MyModel(LUMETorch):
    """Custom model implementing specific logic."""

    def _evaluate(self, input_dict):
        """Implement your model logic here.

        Parameters
        ----------
        input_dict : dict
            Dictionary mapping input variable names to values

        Returns
        -------
        dict
            Dictionary mapping output variable names to values
        """
        x = input_dict["x"]
        y = input_dict["y"]
        return {
            "sum": x + y,
            "product": x * y
        }


# Create model with variables
model = MyModel(
    input_variables=[
        ScalarVariable(name="x", value_range=[0, 10]),
        ScalarVariable(name="y", value_range=[0, 10]),
    ],
    output_variables=[
        ScalarVariable(name="sum"),
        ScalarVariable(name="product"),
    ]
)

# Use the model
result = model({"x": 3.0, "y": 4.0})
```

## PyTorch Models

### TorchModel

Wrapper for PyTorch neural networks.

::: lume_torch.models.torch_model.TorchModel
    options:
        show_root_heading: true
        show_source: true
        members:
            - __init__
            - _evaluate
            - get_device
            - get_dtype

::: lume_torch.models.torch_model.InputDictModel
    options:
        show_root_heading: true
        show_source: true

#### Usage Example

```python
from lume_torch.models.torch_model import TorchModel
from lume_torch.variables import ScalarVariable
import torch.nn as nn

# Create a neural network
network = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

# Wrap in TorchModel
model = TorchModel(
    model=network,
    input_variables=[
        ScalarVariable(name="x1", value_range=[-5, 5]),
        ScalarVariable(name="x2", value_range=[-5, 5]),
    ],
    output_variables=[
        ScalarVariable(name="y"),
    ],
    device="cpu",
    precision="double"
)

# Evaluate
result = model({"x1": 1.0, "x2": 2.0})
```

### TorchModule

PyTorch-compatible interface for LUME models.

::: lume_torch.models.torch_module.TorchModule
    options:
        show_root_heading: true
        show_source: true
        members:
            - __init__
            - forward
            - to

#### Usage Example

```python
from lume_torch.models.torch_module import TorchModule
import torch

# Wrap TorchModel in TorchModule
torch_module = TorchModule(model=model)

# Use like a PyTorch module
input_tensor = torch.tensor([[1.0, 2.0]])
output_tensor = torch_module(input_tensor)

# Integrate with PyTorch pipelines
optimizer = torch.optim.Adam(torch_module.parameters())
```

## Probabilistic Models

For models that output distributions, see [Probabilistic Models](probabilistic-models.md).

## Serialization Functions

::: lume_torch.base.process_torch_module
    options:
        show_root_heading: true
        show_source: true

::: lume_torch.base.model_kwargs_from_dict
    options:
        show_root_heading: true
        show_source: true

::: lume_torch.base.parse_config
    options:
        show_root_heading: true
        show_source: true

::: lume_torch.base.recursive_serialize
    options:
        show_root_heading: true
        show_source: true

::: lume_torch.base.recursive_deserialize
    options:
        show_root_heading: true
        show_source: true

::: lume_torch.base.json_dumps
    options:
        show_root_heading: true
        show_source: true

::: lume_torch.base.json_loads
    options:
        show_root_heading: true
        show_source: true

## Configuration Files

Models can be saved and loaded using YAML configuration files:

```python
# Save model
model.dump("my_model.yml")

# Load model
loaded_model = MyModel("my_model.yml")
```

Example configuration file:

```yaml
model_class: TorchModel
input_variables:
  x1:
    variable_class: ScalarVariable
    default_value: 0.0
    value_range: [-5.0, 5.0]
  x2:
    variable_class: ScalarVariable
    default_value: 0.0
    value_range: [-5.0, 5.0]
output_variables:
  y:
    variable_class: ScalarVariable
model: model.pt
device: cpu
precision: double
```

## See Also

- [Variables](variables.md) - Defining inputs and outputs
- [Probabilistic Models](probabilistic-models.md) - GP and ensemble models
- [Examples](examples.md) - Example notebooks
