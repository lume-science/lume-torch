# Variables

Variables in LUME-torch define the inputs and outputs of models. They provide validation, type checking, and metadata about model parameters.

## Overview

LUME-torch uses variables to:

- Define input and output specifications for models
- Validate values at runtime
- Store metadata like units, ranges, and default values
- Support configuration file serialization

## Variable Types

### Base Variable Classes

These are imported from the [lume-base](https://github.com/roussel-ryan/lume-base) package:

**`Variable`** - Abstract base class for all variables
- Attributes: `name`, `read_only`, `default_validation_config`
- Methods: `validate_value()`, `model_dump()`

**`ScalarVariable`** - Variable for scalar floating-point values
- Attributes: `name`, `default_value`, `read_only`, `value_range`, `unit`
- Methods: `validate_value()`, validates type and range

**`ConfigEnum`** - Validation configuration options
- `"none"`: No validation
- `"warn"`: Emit warnings for validation failures
- `"error"`: Raise errors for validation failures

For complete API documentation of these classes, see the [lume-base documentation](https://github.com/roussel-ryan/lume-base).

### LUME-torch Specific Variables

::: lume_torch.variables.DistributionVariable
    options:
        show_root_heading: true
        show_source: true
        members:
            - validate_value

## Utilities

::: lume_torch.variables.get_variable
    options:
        show_root_heading: true
        show_source: true

## Usage Examples

### Creating Scalar Variables

```python
from lume_torch.variables import ScalarVariable

# Basic variable
var = ScalarVariable(name="temperature")

# Variable with range
var = ScalarVariable(
    name="pressure",
    default_value=1.0,
    value_range=[0.0, 10.0],
    unit="atm"
)

# Read-only variable
constant = ScalarVariable(
    name="speed_of_light",
    default_value=299792458.0,
    read_only=True,
    unit="m/s"
)
```

### Creating Distribution Variables

```python
from lume_torch.variables import DistributionVariable

# For probabilistic model outputs
dist_var = DistributionVariable(
    name="output_distribution",
    unit="GeV"
)
```

### Validation

```python
from lume_torch.variables import ScalarVariable, ConfigEnum

var = ScalarVariable(
    name="energy",
    value_range=[0.0, 100.0],
    default_validation_config=ConfigEnum.ERROR
)

# This will raise an error
try:
    var.validate_value(150.0)
except ValueError as e:
    print(f"Validation failed: {e}")

# Configure validation behavior
var.validate_value(50.0, config=ConfigEnum.WARN)  # Valid, no warning
var.validate_value(150.0, config=ConfigEnum.WARN)  # Warning but no error
```

### Using with Models

```python
from lume_torch.base import LUMETorch
from lume_torch.variables import ScalarVariable


class PhysicsModel(LUMETorch):
    def _evaluate(self, input_dict):
        return {"force": input_dict["mass"] * input_dict["acceleration"]}


input_vars = [
    ScalarVariable(name="mass", value_range=[0.1, 1000.0], unit="kg"),
    ScalarVariable(name="acceleration", value_range=[0.0, 100.0], unit="m/s^2"),
]
output_vars = [
    ScalarVariable(name="force", unit="N"),
]

model = PhysicsModel(
    input_variables=input_vars,
    output_variables=output_vars
)
```

## See Also

- [Models](models.md) - Using variables with models
- [Getting Started](getting-started.md) - Basic usage examples
