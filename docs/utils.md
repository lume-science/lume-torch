# Utilities

LUME-torch provides utility functions for working with variables, paths, imports, and model data.

## Variable Utilities

Functions for serializing, deserializing, and managing variables.

### Serialization

::: lume_torch.utils.variables_as_yaml
    options:
        show_root_heading: true
        show_source: true

::: lume_torch.utils.serialize_variables
    options:
        show_root_heading: true
        show_source: true

### Deserialization

::: lume_torch.utils.variables_from_yaml
    options:
        show_root_heading: true
        show_source: true

::: lume_torch.utils.variables_from_dict
    options:
        show_root_heading: true
        show_source: true

::: lume_torch.utils.deserialize_variables
    options:
        show_root_heading: true
        show_source: true

### Validation

::: lume_torch.utils.verify_unique_variable_names
    options:
        show_root_heading: true
        show_source: true

## Path Utilities

Functions for handling file paths and path resolution.

::: lume_torch.utils.get_valid_path
    options:
        show_root_heading: true
        show_source: true

::: lume_torch.utils.replace_relative_paths
    options:
        show_root_heading: true
        show_source: true

## Import Utilities

::: lume_torch.utils.try_import_module
    options:
        show_root_heading: true
        show_source: true

## Model Utilities

Utilities for working with model inputs and outputs.

::: lume_torch.models.utils.itemize_dict
    options:
        show_root_heading: true
        show_source: true

::: lume_torch.models.utils.format_inputs
    options:
        show_root_heading: true
        show_source: true

::: lume_torch.models.utils.InputDictModel
    options:
        show_root_heading: true
        show_source: true

## Usage Examples

### Working with Variables

```python
from lume_torch.variables import ScalarVariable
from lume_torch.utils import (
    variables_as_yaml,
    variables_from_yaml,
    serialize_variables,
    deserialize_variables
)

# Create variables
variables = [
    ScalarVariable(name="x", value_range=[0, 10]),
    ScalarVariable(name="y", value_range=[0, 10]),
]

# Serialize to YAML string
yaml_str = variables_as_yaml(variables)
print(yaml_str)

# Deserialize from YAML
loaded_vars = variables_from_yaml(yaml_str)

# Serialize to dict
var_dict = serialize_variables(variables)

# Deserialize from dict
restored_vars = deserialize_variables(var_dict)
```

### Path Resolution

```python
from lume_torch.utils import get_valid_path, replace_relative_paths
import os

# Get valid path (resolves relative paths)
config_path = get_valid_path("models/model.yml")

# Replace relative paths in a dictionary
config = {
    "model": "model.pt",
    "transformers": ["transform1.pt", "transform2.pt"]
}

# Convert relative to absolute paths
base_path = "/path/to/models"
absolute_config = replace_relative_paths(config, base_path)
```

### Dynamic Imports

```python
from lume_torch.utils import try_import_module

# Safely import optional dependencies
mlflow = try_import_module("mlflow")
if mlflow is not None:
    # Use mlflow
    mlflow.log_metric("loss", 0.05)
else:
    print("MLflow not installed")

# Import custom model class
model_module = try_import_module("my_models.custom_model")
if model_module:
    ModelClass = getattr(model_module, "CustomModel")
```

### Itemizing Dictionaries

```python
from lume_torch.models.utils import itemize_dict
import torch

# Batched input dictionary
batch_input = {
    "x": torch.tensor([1.0, 2.0, 3.0]),
    "y": torch.tensor([4.0, 5.0, 6.0])
}

# Convert to list of individual dictionaries
individual_inputs = itemize_dict(batch_input)
# Result: [{"x": 1.0, "y": 4.0}, {"x": 2.0, "y": 5.0}, {"x": 3.0, "y": 6.0}]
```

### Formatting Inputs

```python
from lume_torch.models.utils import format_inputs
import torch

# Convert dictionary to tensor
input_dict = {"x": 1.0, "y": 2.0}
input_order = ["x", "y"]

input_tensor = format_inputs(input_dict, input_order)
# Result: torch.tensor([[1.0, 2.0]])
```

## See Also

- [Variables](variables.md) - Variable types and usage
- [Models](models.md) - Using utilities with models
