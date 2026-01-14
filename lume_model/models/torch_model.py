import os
import logging
from typing import Union, Callable
from copy import deepcopy

import torch
from pydantic import field_validator
from botorch.models.transforms.input import ReversibleInputTransform

from lume_model.base import LUMEBaseModel
from lume_model.variables import ScalarVariable, ArrayVariable
from lume_model.models.utils import itemize_dict, format_inputs

logger = logging.getLogger(__name__)


class TorchModel(LUMEBaseModel):
    """LUME-model class for torch models.

    By default, the models are assumed to be fixed, so all gradient computation is deactivated and the model and
    transformers are put in evaluation mode.

    Attributes:
        model: The torch base model.
        input_variables: List defining the input variables and their order.
        output_variables: List defining the output variables and their order.
        input_transformers: List of transformer objects to apply to input before passing to model.
        output_transformers: List of transformer objects to apply to output of model.
        output_format: Determines format of outputs: "tensor" or "raw".
        device: Device on which the model will be evaluated. Defaults to "cpu".
        fixed_model: If true, the model and transformers are put in evaluation mode and all gradient
          computation is deactivated.
        precision: Precision of the model, either "double" or "single".
    """

    model: torch.nn.Module
    input_transformers: list[
        Union[ReversibleInputTransform, torch.nn.Linear, Callable]
    ] = None
    output_transformers: list[
        Union[ReversibleInputTransform, torch.nn.Linear, Callable]
    ] = None
    output_format: str = "tensor"
    device: Union[torch.device, str] = "cpu"
    fixed_model: bool = True
    precision: str = "double"

    def __init__(self, *args, **kwargs):
        """Initializes TorchModel.

        Args:
            *args: Accepts a single argument which is the model configuration as dictionary, YAML or JSON
              formatted string or file path.
            **kwargs: See class attributes.
        """
        super().__init__(*args, **kwargs)
        self.input_transformers = (
            [] if self.input_transformers is None else self.input_transformers
        )
        self.output_transformers = (
            [] if self.output_transformers is None else self.output_transformers
        )

        # dtype property sets precision across model and transformers
        self.dtype

        # fixed model: set full model in eval mode and deactivate all gradients
        if self.fixed_model:
            is_scripted = isinstance(self.model, torch.jit.ScriptModule)
            self.model.eval().requires_grad_(False) if not is_scripted else None
            for t in self.input_transformers + self.output_transformers:
                if isinstance(t, torch.nn.Module):
                    t.eval().requires_grad_(False)

        # ensure consistent device
        self.to(self.device)

    @property
    def dtype(self):
        if self.precision == "double":
            self._dtype = torch.double
        elif self.precision == "single":
            self._dtype = torch.float
        else:
            raise ValueError(
                f"Unknown precision {self.precision}, "
                f"expected one of ['double', 'single']."
            )
        self._set_precision(self._dtype)
        return self._dtype

    @property
    def _tkwargs(self):
        return {"device": self.device, "dtype": self.dtype}

    @field_validator("model", mode="before")
    def validate_torch_model(cls, v):
        if isinstance(v, (str, os.PathLike)):
            if os.path.exists(v):
                fname = v
                try:
                    v = torch.jit.load(v)
                    print(f"Loaded TorchScript (JIT) model from file: {fname}")
                except RuntimeError:
                    v = torch.load(v, weights_only=False)
                    print(f"Loaded PyTorch model from file: {fname}")
            else:
                raise OSError(f"File {v} is not found.")
        return v

    @field_validator("input_variables")
    def verify_input_default_value(cls, value):
        """Verifies that input variables have the required default values."""
        for var in value:
            if var.default_value is None:
                raise ValueError(
                    f"Input variable {var.name} must have a default value."
                )
        return value

    @field_validator("input_transformers", "output_transformers", mode="before")
    def validate_transformers(cls, v):
        if not isinstance(v, list):
            raise ValueError("Transformers must be passed as list.")
        loaded_transformers = []
        for t in v:
            if isinstance(t, (str, os.PathLike)):
                if os.path.exists(t):
                    t = torch.load(t, weights_only=False)
                else:
                    raise OSError(f"File {t} is not found.")
            loaded_transformers.append(t)
        v = loaded_transformers
        return v

    @field_validator("output_format")
    def validate_output_format(cls, v):
        supported_formats = ["tensor", "variable", "raw"]
        if v not in supported_formats:
            raise ValueError(
                f"Unknown output format {v}, expected one of {supported_formats}."
            )
        return v

    def _set_precision(self, value: torch.dtype):
        """Sets the precision of the model."""
        self.model.to(dtype=value)
        for t in self.input_transformers + self.output_transformers:
            if isinstance(t, torch.nn.Module):
                t.to(dtype=value)

    def _evaluate(
        self,
        input_dict: dict[str, Union[float, torch.Tensor]],
    ) -> dict[str, Union[float, torch.Tensor]]:
        """Evaluates model on the given input dictionary.

        Args:
            input_dict: Input dictionary on which to evaluate the model.

        Returns:
            Dictionary of output variable names to values.
        """
        formatted_inputs = format_inputs(input_dict)
        input_tensor = self._arrange_inputs(formatted_inputs)
        input_tensor = self._transform_inputs(input_tensor)
        output_tensor = self.model(input_tensor)
        output_tensor = self._transform_outputs(output_tensor)
        parsed_outputs = self._parse_outputs(output_tensor)
        output_dict = self._prepare_outputs(parsed_outputs)
        return output_dict

    def input_validation(self, input_dict: dict[str, Union[float, torch.Tensor]]):
        """Validates input dictionary before evaluation.

        Args:
            input_dict: Input dictionary to validate.

        Returns:
            Validated input dictionary.
        """
        # validate original inputs (catches dtype mismatches)
        for name, value in input_dict.items():
            if name in self.input_names:
                var = self.input_variables[self.input_names.index(name)]
                _config = (
                    None
                    if self.input_validation_config is None
                    else self.input_validation_config.get(name)
                )
                var.validate_value(value, config=_config)

        # format inputs as tensors w/o changing the dtype
        formatted_inputs = format_inputs(input_dict)

        # cast tensors to expected dtype and device
        formatted_inputs = {
            k: v.to(**self._tkwargs) for k, v in formatted_inputs.items()
        }

        # check default values for missing inputs
        filled_inputs = self._fill_default_inputs(formatted_inputs)

        return filled_inputs

    def output_validation(self, output_dict: dict[str, Union[float, torch.Tensor]]):
        """Itemizes tensors before performing output validation."""
        for i, var in enumerate(self.output_variables):
            if isinstance(var, (ArrayVariable)):
                # run the validation for ArrayVariable
                super().output_validation({var.name: output_dict[var.name]})
            elif isinstance(var, ScalarVariable):
                itemized_outputs = itemize_dict({var.name: output_dict[var.name]})
                for ele in itemized_outputs:
                    super().output_validation(ele)

    def random_input(self, n_samples: int = 1) -> dict[str, torch.Tensor]:
        """Generates random input(s) for the model.

        Args:
            n_samples: Number of random samples to generate.

        Returns:
            Dictionary of input variable names to tensors.
        """
        input_dict = {}
        for var in self.input_variables:
            if isinstance(var, ScalarVariable):
                input_dict[var.name] = var.value_range[0] + torch.rand(
                    size=(n_samples,)
                ) * (var.value_range[1] - var.value_range[0])
            else:
                var.default_value.detach().clone().repeat((n_samples, 1))
        return input_dict

    def random_evaluate(
        self, n_samples: int = 1
    ) -> dict[str, Union[float, torch.Tensor]]:
        """Returns random evaluation(s) of the model.

        Args:
            n_samples: Number of random samples to evaluate.

        Returns:
            Dictionary of variable names to outputs.
        """
        random_input = self.random_input(n_samples)
        return self.evaluate(random_input)

    def to(self, device: Union[torch.device, str]):
        """Updates the device for the model, transformers and default values.

        Args:
            device: Device on which the model will be evaluated.
        """
        self.model.to(device)
        for t in self.input_transformers + self.output_transformers:
            if isinstance(t, torch.nn.Module):
                t.to(device)
        self.device = device

    def insert_input_transformer(
        self, new_transformer: ReversibleInputTransform, loc: int
    ):
        """Inserts an additional input transformer at the given location.

        Args:
            new_transformer: New transformer to add.
            loc: Location where the new transformer shall be added to the transformer list.
        """
        self.input_transformers = (
            self.input_transformers[:loc]
            + [new_transformer]
            + self.input_transformers[loc:]
        )

    def insert_output_transformer(
        self, new_transformer: ReversibleInputTransform, loc: int
    ):
        """Inserts an additional output transformer at the given location.

        Args:
            new_transformer: New transformer to add.
            loc: Location where the new transformer shall be added to the transformer list.
        """
        self.output_transformers = (
            self.output_transformers[:loc]
            + [new_transformer]
            + self.output_transformers[loc:]
        )

    def update_input_variables_to_transformer(
        self, transformer_loc: int
    ) -> list[ScalarVariable]:
        """Returns input variables updated to the transformer at the given location.

        Updated are the value ranges and default of the input variables. This allows, e.g., to add a
        calibration transformer and to update the input variable specification accordingly.

        Args:
            transformer_loc: The location of the input transformer to adjust for.

        Returns:
            The updated input variables.
        """
        x_old = {
            "min": torch.tensor(
                [var.value_range[0] for var in self.input_variables], dtype=self.dtype
            ),
            "max": torch.tensor(
                [var.value_range[1] for var in self.input_variables], dtype=self.dtype
            ),
            "default": torch.tensor(
                [var.default_value for var in self.input_variables], dtype=self.dtype
            ),
        }
        x_new = {}
        for key, x in x_old.items():
            # Make at least 2D
            if x.ndim == 0:
                x = x.unsqueeze(0)
            if x.ndim == 1:
                x = x.unsqueeze(0)

            # compute previous limits at transformer location
            for i in range(transformer_loc):
                if isinstance(self.input_transformers[i], ReversibleInputTransform):
                    x = self.input_transformers[i].transform(x)
                else:
                    x = self.input_transformers[i](x)
            # untransform of transformer to adjust for
            if isinstance(
                self.input_transformers[transformer_loc], ReversibleInputTransform
            ):
                x = self.input_transformers[transformer_loc].untransform(x)
            elif isinstance(self.input_transformers[transformer_loc], torch.nn.Linear):
                w = self.input_transformers[transformer_loc].weight
                b = self.input_transformers[transformer_loc].bias
                x = torch.matmul((x - b), torch.linalg.inv(w.T))
            else:
                raise NotImplementedError(
                    f"Reverse transformation for type {type(self.input_transformers[transformer_loc])} is not supported."
                )
            # backtrack through transformers
            for transformer in self.input_transformers[:transformer_loc][::-1]:
                if isinstance(
                    self.input_transformers[transformer_loc], ReversibleInputTransform
                ):
                    x = transformer.untransform(x)
                elif isinstance(
                    self.input_transformers[transformer_loc], torch.nn.Linear
                ):
                    w, b = transformer.weight, transformer.bias
                    x = torch.matmul((x - b), torch.linalg.inv(w.T))
                else:
                    raise NotImplementedError(
                        f"Reverse transformation for type {type(self.input_transformers[transformer_loc])} is not supported."
                    )

            x_new[key] = x
        updated_variables = deepcopy(self.input_variables)
        for i, var in enumerate(updated_variables):
            var.value_range = [x_new["min"][0][i].item(), x_new["max"][0][i].item()]
            var.default_value = x_new["default"][0][i].item()
        return updated_variables

    def _fill_default_inputs(
        self, input_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Fills missing input variables with default values.

        Args:
            input_dict: Dictionary of input variable names to tensors.

        Returns:
            Dictionary of input variable names to tensors with default values for missing inputs.
        """
        for var in self.input_variables:
            if var.name not in input_dict.keys():
                if isinstance(var.default_value, torch.Tensor):
                    input_dict[var.name] = var.default_value.detach().clone()
                else:
                    # Handle float default values for ScalarVariable
                    input_dict[var.name] = torch.tensor(
                        var.default_value, **self._tkwargs
                    )
                return input_dict
        return input_dict

    def _arrange_inputs(
        self, formatted_inputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Enforces order of input variables.

        Enforces the order of the input variables to be passed to the transformers and model and updates the
        returned tensor with default values for any inputs that are missing.

        Args:
            formatted_inputs: Dictionary of input variable names to tensors.

        Returns:
            Ordered input tensor to be passed to the transformers.
        """
        # Build default values tensor, handling both scalar and array variables
        default_list = []
        for var in self.input_variables:
            if isinstance(var, ArrayVariable):
                # Array variables: keep as-is (already a tensor)
                default_list.append(var.default_value)
            else:  # ScalarVariable
                # Scalar variables: handle both tensor and float default values
                if isinstance(var.default_value, torch.Tensor):
                    default_list.append(var.default_value.detach().clone())
                else:
                    # Handle float default values for ScalarVariable
                    default_list.append(
                        torch.tensor(var.default_value, **self._tkwargs)
                    )

        # Concatenate along the last dimension to create [total_features] tensor
        default_tensor = torch.cat([d.flatten() for d in default_list]).to(
            **self._tkwargs
        )

        # Determine batch size from inputs
        if formatted_inputs:
            # Infer batch shape from first provided input
            batch_shape = None
            for var in self.input_variables:
                if var.name in formatted_inputs:
                    value = formatted_inputs[var.name]
                    if isinstance(var, ArrayVariable):
                        # Array: batch dims are all except last len(shape) dims
                        expected_ndim = len(var.shape)
                        batch_shape = (
                            value.shape[:-expected_ndim]
                            if expected_ndim > 0
                            else value.shape
                        )
                    else:  # ScalarVariable
                        # Scalar: batch dims are all except last 1 dim (if last dim is 0 or 1)
                        if value.ndim > 0 and value.shape[-1] in (0, 1):
                            batch_shape = value.shape[:-1]
                        else:
                            batch_shape = value.shape
                    break

            # Build input tensor with correct batch shape
            if batch_shape and len(batch_shape) > 0:
                # Has batch dimensions - expand default tensor
                expanded_shape = (*batch_shape, default_tensor.shape[0])
                input_tensor = (
                    default_tensor.unsqueeze(0).expand(expanded_shape).clone()
                )
            else:
                # No batch dimensions, use default with single batch dim
                input_tensor = default_tensor.unsqueeze(0)

            # Fill in provided values
            current_idx = 0
            for var in self.input_variables:
                if var.name in formatted_inputs:
                    value = formatted_inputs[var.name]
                    if isinstance(var, ArrayVariable):
                        # Array variable: assign the full array
                        num_features = var.default_value.numel()
                        # Flatten only the feature dimensions, keep batch dims
                        expected_ndim = len(var.shape)
                        if value.ndim > expected_ndim:
                            # Has batch dimensions
                            batch_dims = value.ndim - expected_ndim
                            flat_value = value.reshape(*value.shape[:batch_dims], -1)
                        else:
                            flat_value = value.flatten()
                        input_tensor[..., current_idx : current_idx + num_features] = (
                            flat_value
                        )
                    else:  # ScalarVariable
                        # Scalar variable: squeeze last dim if it's 1 (batch format)
                        if value.ndim > 0 and value.shape[-1] == 1:
                            input_tensor[..., current_idx] = value.squeeze(-1)
                        else:
                            input_tensor[..., current_idx] = value

                # Update index
                if isinstance(var, ArrayVariable):
                    current_idx += var.default_value.numel()
                else:
                    current_idx += 1
        else:
            # No inputs provided, use default tensor with batch dimension
            input_tensor = default_tensor.unsqueeze(0)

        # Validate total feature dimension
        expected_features = sum(
            var.default_value.numel() if isinstance(var, ArrayVariable) else 1
            for var in self.input_variables
        )
        if input_tensor.shape[-1] != expected_features:
            raise ValueError(
                f"Last dimension of input tensor doesn't match the expected number of features\n"
                f"received: {input_tensor.shape}, expected {expected_features} as the last dimension"
            )
        return input_tensor

    def _transform_inputs(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Applies transformations to the inputs.

        Args:
            input_tensor: Ordered input tensor to be passed to the transformers.

        Returns:
            Tensor of transformed inputs to be passed to the model.
        """
        # Make at least 2D
        if input_tensor.ndim == 0:
            input_tensor = input_tensor.unsqueeze(0)
        if input_tensor.ndim == 1:
            input_tensor = input_tensor.unsqueeze(0)

        for transformer in self.input_transformers:
            if isinstance(transformer, ReversibleInputTransform):
                input_tensor = transformer.transform(input_tensor)
            else:
                input_tensor = transformer(input_tensor)
        return input_tensor

    def _transform_outputs(self, output_tensor: torch.Tensor) -> torch.Tensor:
        """(Un-)Transforms the model output tensor.

        Args:
            output_tensor: Output tensor from the model.

        Returns:
            (Un-)Transformed output tensor.
        """
        for transformer in self.output_transformers:
            if isinstance(transformer, ReversibleInputTransform):
                output_tensor = transformer.untransform(output_tensor)
            elif isinstance(transformer, torch.nn.Linear):
                w, b = transformer.weight, transformer.bias
                output_tensor = torch.matmul((output_tensor - b), torch.linalg.inv(w.T))
            else:
                # we assume anything else is provided as a callable
                output_tensor = transformer(output_tensor)
        return output_tensor

    def _parse_outputs(self, output_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        """Constructs dictionary from model output tensor.

        Args:
            output_tensor: (Un-)transformed output tensor from the model.

        Returns:
            Dictionary of output variable names to (un-)transformed tensors.
        """
        parsed_outputs = {}
        if output_tensor.dim() in [0, 1]:
            output_tensor = output_tensor.unsqueeze(0)
        if len(self.output_names) == 1:
            parsed_outputs[self.output_names[0]] = output_tensor.squeeze()
        else:
            for idx, output_name in enumerate(self.output_names):
                parsed_outputs[output_name] = output_tensor[..., idx].squeeze()
        return parsed_outputs

    def _prepare_outputs(
        self,
        parsed_outputs: dict[str, torch.Tensor],
    ) -> dict[str, Union[float, torch.Tensor]]:
        """Updates and returns outputs according to output_format.

        Updates the output variables within the model to reflect the new values.

        Args:
            parsed_outputs: Dictionary of output variable names to transformed tensors.

        Returns:
            Dictionary of output variable names to values depending on output_format.
        """
        if self.output_format.lower() == "tensor":
            return parsed_outputs
        else:
            return {
                key: value.item() if value.squeeze().dim() == 0 else value
                for key, value in parsed_outputs.items()
            }
