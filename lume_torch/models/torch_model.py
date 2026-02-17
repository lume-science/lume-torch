import os
import logging
from typing import Union, Callable
from copy import deepcopy

import torch
from pydantic import field_validator
from botorch.models.transforms.input import ReversibleInputTransform

from lume_torch.base import LUMETorch
from lume_torch.variables import TorchScalarVariable, TorchNDVariable
from lume_torch.models.utils import itemize_dict, format_inputs


logger = logging.getLogger(__name__)


class TorchModel(LUMETorch):
    """LUME-model class for torch models.

    By default, the models are assumed to be fixed, so all gradient computation
    is deactivated and the model and transformers are put in evaluation mode.

    Attributes
    ----------
    model : torch.nn.Module
        The underlying torch model.
    input_variables : list of TorchScalarVariable or TorchNDVariable
        List defining the input variables and their order. Supports both scalar
        variables and multi-dimensional array variables.
    output_variables : list of TorchScalarVariable or TorchNDVariable
        List defining the output variables and their order.
    input_transformers : list of ReversibleInputTransform, torch.nn.Linear, or Callable
        Transformer objects applied to the inputs before passing to the model.
    output_transformers : list of ReversibleInputTransform, torch.nn.Linear, or Callable
        Transformer objects applied to the outputs of the model.
    output_format : {"tensor", "variable", "raw"}
        Determines format of outputs. "tensor" returns tensors, "variable" and
        "raw" return scalars where possible.
    device : torch.device or str
        Device on which the model will be evaluated. Defaults to ``"cpu"``.
    fixed_model : bool
        If ``True``, the model and transformers are put in evaluation mode and
        all gradient computation is deactivated.
    precision : {"double", "single"}
        Precision of the model, either ``"double"`` or ``"single"``.

    Methods
    -------
    evaluate(input_dict, **kwargs)
        Evaluate the model on a dictionary of inputs and return outputs.
    input_validation(input_dict)
        Validate and normalize the input dictionary before evaluation.
    output_validation(output_dict)
        Validate the output dictionary after evaluation.
    random_input(n_samples=1)
        Generate random inputs consistent with the input variable ranges.
    random_evaluate(n_samples=1)
        Evaluate the model on random inputs.
    to(device)
        Move the model, transformers, and default values to a given device.

    Notes
    -----
    When using TorchNDVariable inputs, all inputs must be TorchNDVariable.
    Mixing TorchScalarVariable and TorchNDVariable is not currently supported.

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

        Parameters
        ----------
        *args : dict, str, or Path
            Accepts a single argument which is the model configuration as dictionary, YAML or JSON
            formatted string or file path.
        **kwargs
            See class attributes.

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
    @classmethod
    def validate_torch_model(
        cls, v: Union[str, os.PathLike, torch.nn.Module]
    ) -> torch.nn.Module:
        """Validate and load the torch model from file if needed.

        Parameters
        ----------
        v : str, os.PathLike, or torch.nn.Module
            Model or path to model file.

        Returns
        -------
        torch.nn.Module
            Loaded or validated torch model.

        Raises
        ------
        OSError
            If the model file does not exist.
        """
        if isinstance(v, (str, os.PathLike)):
            if os.path.exists(v):
                fname = v
                try:
                    v = torch.jit.load(v)
                    logger.info(f"Loaded TorchScript (JIT) model from file: {fname}")
                except RuntimeError:
                    v = torch.load(v, weights_only=False)
                    logger.info(f"Loaded PyTorch model from file: {fname}")
            else:
                logger.error(f"File {v} not found")
                raise OSError(f"File {v} is not found.")
        return v

    @field_validator("input_variables")
    @classmethod
    def verify_input_default_value(
        cls, value: list[Union[TorchScalarVariable, TorchNDVariable]]
    ) -> list[Union[TorchScalarVariable, TorchNDVariable]]:
        """Verify that input variables have the required default values.

        Parameters
        ----------
        value : list of TorchScalarVariable or TorchNDVariable
            Input variables to validate.

        Returns
        -------
        list of TorchScalarVariable or TorchNDVariable
            Validated input variables.

        Raises
        ------
        ValueError
            If any input variable is missing a default value.
        """
        for var in value:
            if var.default_value is None:
                logger.error(
                    f"Input variable {var.name} is missing required default value"
                )
                raise ValueError(
                    f"Input variable {var.name} must have a default value."
                )
        return value

    @field_validator("input_transformers", "output_transformers", mode="before")
    @classmethod
    def validate_transformers(cls, v: Union[list, str, os.PathLike]) -> list:
        """Validate and load transformers from files if needed.

        Parameters
        ----------
        v : list, str, or os.PathLike
            List of transformers or paths to transformer files.

        Returns
        -------
        list
            List of loaded transformers.

        Raises
        ------
        ValueError
            If transformers are not provided as a list.
        OSError
            If a transformer file does not exist.
        """
        if not isinstance(v, list):
            logger.error(f"Transformers must be a list, got {type(v)}")
            raise ValueError("Transformers must be passed as list.")
        loaded_transformers = []
        for t in v:
            if isinstance(t, (str, os.PathLike)):
                if os.path.exists(t):
                    t = torch.load(t, weights_only=False)
                    logger.debug(f"Loaded transformer from file: {t}")
                else:
                    logger.error(f"Transformer file {t} not found")
                    raise OSError(f"File {t} is not found.")
            loaded_transformers.append(t)
        v = loaded_transformers
        return v

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        """Validate the output format.

        Parameters
        ----------
        v : str
            Output format to validate.

        Returns
        -------
        str
            Validated output format.

        Raises
        ------
        ValueError
            If output format is not one of the supported formats.
        """
        supported_formats = ["tensor", "variable", "raw"]
        if v not in supported_formats:
            logger.error(
                f"Invalid output format {v}, expected one of {supported_formats}"
            )
            raise ValueError(
                f"Unknown output format {v}, expected one of {supported_formats}."
            )
        return v

    def _set_precision(self, value: torch.dtype):
        """Sets the precision of the model and transformers.

        Parameters
        ----------
        value : torch.dtype
            Dtype to set for the model and transformers.
        """
        self.model.to(dtype=value)
        for t in self.input_transformers + self.output_transformers:
            if isinstance(t, torch.nn.Module):
                t.to(dtype=value)

    def _default_to_tensor(
        self, default_value: Union[torch.Tensor, float]
    ) -> torch.Tensor:
        """Convert a default value to a tensor with proper dtype and device.

        Parameters
        ----------
        default_value : torch.Tensor or float
            Default value to convert.

        Returns
        -------
        torch.Tensor
            Default value as a tensor with proper dtype and device.
        """
        if isinstance(default_value, torch.Tensor):
            return default_value.detach().clone().to(**self._tkwargs)
        else:
            return torch.tensor(default_value, **self._tkwargs)

    def _evaluate(
        self,
        input_dict: dict[str, Union[float, torch.Tensor]],
    ) -> dict[str, Union[float, torch.Tensor]]:
        """Evaluate the model on the given input dictionary.

        Parameters
        ----------
        input_dict : dict of str to float or torch.Tensor
            Input dictionary on which to evaluate the model.

        Returns
        -------
        dict of str to float or torch.Tensor
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
        """Validate the input dictionary before evaluation.

        Parameters
        ----------
        input_dict : dict of str to float or torch.Tensor
            Input dictionary to validate.

        Returns
        -------
        dict of str to float or torch.Tensor
            Validated input dictionary.

        """
        # validate original inputs (catches dtype mismatches)
        super().input_validation(input_dict)

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
        """Itemize tensors before performing output validation.

        Parameters
        ----------
        output_dict : dict of str to float or torch.Tensor
            Output dictionary to validate.

        """
        for var in self.output_variables:
            if isinstance(var, TorchNDVariable):
                # run the validation for TorchNDVariable (arrays/images)
                super().output_validation({var.name: output_dict[var.name]})
            elif isinstance(var, TorchScalarVariable):
                # itemize scalar tensors for element-wise validation
                itemized_outputs = itemize_dict({var.name: output_dict[var.name]})
                for ele in itemized_outputs:
                    super().output_validation(ele)

    def random_input(self, n_samples: int = 1) -> dict[str, torch.Tensor]:
        """Generates random input(s) for the model.

        Parameters
        ----------
        n_samples : int, optional
            Number of random samples to generate.

        Returns
        -------
        dict of str to torch.Tensor
            Dictionary of input variable names to tensors.

        Notes
        -----
        For TorchScalarVariable inputs, generates random values within the variable's
        value_range. For TorchNDVariable inputs, repeats the default value for
        the requested number of samples.

        """
        input_dict = {}
        for var in self.input_variables:
            if isinstance(var, TorchScalarVariable):
                input_dict[var.name] = var.value_range[0] + torch.rand(
                    size=(n_samples,), **self._tkwargs
                ) * (var.value_range[1] - var.value_range[0])
            elif isinstance(var, TorchNDVariable):
                # For ND variables, repeat the default value for n_samples
                # Works for any dimensionality: 1D arrays, 2D matrices, 3D images, etc.
                default = self._default_to_tensor(var.default_value)
                # Add batch dim and repeat n_samples times (keeping original shape)
                # e.g., (3, 64, 64) -> (1, 3, 64, 64) -> (n_samples, 3, 64, 64)
                input_dict[var.name] = default.unsqueeze(0).repeat(
                    n_samples, *([1] * default.ndim)
                )
        return input_dict

    def random_evaluate(
        self, n_samples: int = 1
    ) -> dict[str, Union[float, torch.Tensor]]:
        """Return random evaluations of the model.

        Parameters
        ----------
        n_samples : int, optional
            Number of random samples to evaluate.

        Returns
        -------
        dict of str to float or torch.Tensor
            Dictionary of variable names to outputs.

        """
        random_input = self.random_input(n_samples)
        return self.evaluate(random_input)

    def to(self, device: Union[torch.device, str]):
        """Update the device for the model, transformers and default values.

        Parameters
        ----------
        device : torch.device or str
            Device on which the model will be evaluated.

        """
        self.model.to(device)
        for t in self.input_transformers + self.output_transformers:
            if isinstance(t, torch.nn.Module):
                t.to(device)
        self.device = device

    def insert_input_transformer(
        self, new_transformer: ReversibleInputTransform, loc: int
    ):
        """Insert an additional input transformer at the given location.

        Parameters
        ----------
        new_transformer : ReversibleInputTransform
            New transformer to add.
        loc : int
            Location where the new transformer shall be added to the
            transformer list.

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

        Parameters
        ----------
        new_transformer : ReversibleInputTransform
            New transformer to add.
        loc : int
            Location where the new transformer shall be added to the transformer list.

        """
        self.output_transformers = (
            self.output_transformers[:loc]
            + [new_transformer]
            + self.output_transformers[loc:]
        )

    def update_input_variables_to_transformer(
        self, transformer_loc: int
    ) -> list[TorchScalarVariable]:
        """Return input variables updated to the transformer at the given location.

        Updated are the value ranges and defaults of the input variables. This
        allows, for example, adding a calibration transformer and updating the
        input variable specification accordingly.

        Parameters
        ----------
        transformer_loc : int
            Index of the input transformer to adjust for.

        Returns
        -------
        list of TorchScalarVariable
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
                if isinstance(transformer, ReversibleInputTransform):
                    x = transformer.untransform(x)
                elif isinstance(transformer, torch.nn.Linear):
                    w, b = transformer.weight, transformer.bias
                    x = torch.matmul((x - b), torch.linalg.inv(w.T))
                else:
                    raise NotImplementedError(
                        f"Reverse transformation for type {type(transformer)} is not supported."
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
        """Fill missing input variables with default values.

        Parameters
        ----------
        input_dict : dict of str to torch.Tensor
            Dictionary of input variable names to tensors.

        Returns
        -------
        dict of str to torch.Tensor
            Dictionary of input variable names to tensors with default values
            for missing inputs.

        """
        for var in self.input_variables:
            if var.name not in input_dict.keys():
                input_dict[var.name] = self._default_to_tensor(var.default_value)

        return input_dict

    def _arrange_inputs(
        self, formatted_inputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Enforces ordering, batching, and default filling of inputs.

        * If all inputs are `TorchNDVariable`, stacks them into shape `(batch, num_arrays, *array_shape)`.
        * If all inputs are `TorchScalarVariable`, concatenates them so the last dimension matches the number of inputs,
         broadcasting defaults as needed.
        * If a mix of array and scalar inputs is provided, raises `NotImplementedError`.
        * Missing inputs are filled with their default values before arranging.

        Parameters
        ----------
        formatted_inputs : dict of str to torch.Tensor
            Dictionary of input variable names to tensors.

        Returns
        -------
        torch.Tensor
            Ordered input tensor to be passed to the transformers.

        """
        contains_array = any(
            isinstance(v, TorchNDVariable) for v in self.input_variables
        )
        contains_scalar = any(
            isinstance(v, TorchScalarVariable) for v in self.input_variables
        )

        if contains_array and contains_scalar:
            raise NotImplementedError(
                "Mixing TorchScalarVariable and TorchNDVariable inputs is not supported."
            )

        # All TorchNDVariable
        if contains_array:
            tensor_list = []
            batch_shape = None
            for var in self.input_variables:
                if var.name in formatted_inputs:
                    value = formatted_inputs[var.name]
                else:
                    value = self._default_to_tensor(var.default_value)

                expected_sample_shape = tuple(var.shape)
                sample_ndim = len(expected_sample_shape)
                if value.shape[-sample_ndim:] != expected_sample_shape:
                    raise ValueError(
                        f"Input {var.name} has shape {value.shape}, "
                        f"expected sample shape {expected_sample_shape}"
                    )

                current_batch = value.shape[:-sample_ndim]
                if current_batch == torch.Size():
                    # No batch dim provided -> add singleton batch
                    value = value.unsqueeze(0)
                    current_batch = torch.Size([1])

                if batch_shape is None:
                    batch_shape = current_batch
                elif current_batch != batch_shape:
                    raise ValueError(
                        f"Inputs have inconsistent batch shapes: "
                        f"{batch_shape} vs {current_batch}"
                    )

                tensor_list.append(value.to(**self._tkwargs))

            stacked = torch.stack(tensor_list, dim=1)  # (batch, num_arrays, ...)
            logger.debug(f"Arranged ND inputs into tensor shape: {stacked.shape}")
            return stacked

        # All TorchScalarVariables
        default_list = []
        for var in self.input_variables:
            default_list.append(self._default_to_tensor(var.default_value))

        default_tensor = torch.cat([d.flatten() for d in default_list]).to(
            **self._tkwargs
        )

        if formatted_inputs:
            batch_shape = None
            for var in self.input_variables:
                if var.name in formatted_inputs:
                    value = formatted_inputs[var.name]
                    if value.ndim > 0 and value.shape[-1] in (0, 1):
                        batch_shape = value.shape[:-1]
                    else:
                        batch_shape = value.shape
                    break

            if batch_shape and len(batch_shape) > 0:
                expanded_shape = (*batch_shape, default_tensor.shape[0])
                input_tensor = (
                    default_tensor.unsqueeze(0).expand(expanded_shape).clone()
                )
            else:
                input_tensor = default_tensor.unsqueeze(0)

            current_idx = 0
            for var in self.input_variables:
                if var.name in formatted_inputs:
                    value = formatted_inputs[var.name]
                    if value.ndim > 0 and value.shape[-1] == 1:
                        input_tensor[..., current_idx] = value.squeeze(-1)
                    else:
                        input_tensor[..., current_idx] = value
                current_idx += 1
        else:
            input_tensor = default_tensor.unsqueeze(0)

        expected_features = len(self.input_variables)
        if input_tensor.shape[-1] != expected_features:
            raise ValueError(
                "Last dimension of input tensor doesn't match the expected number of features\n"
                f"received: {input_tensor.shape}, expected {expected_features} as the last dimension"
            )

        return input_tensor

    def _transform_inputs(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Applies transformations to the inputs.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Ordered input tensor to be passed to the transformers.

        Returns
        -------
        torch.Tensor
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
        """(Un-)transform the model output tensor.

        Parameters
        ----------
        output_tensor : torch.Tensor
            Output tensor from the model.

        Returns
        -------
        torch.Tensor
            (Un-)transformed output tensor.

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
        """Construct a dictionary from the model output tensor.

        Parameters
        ----------
        output_tensor : torch.Tensor
            (Un-)transformed output tensor from the model.

        Returns
        -------
        dict of str to torch.Tensor
            Dictionary of output variable names to (un-)transformed tensors.

        """
        parsed_outputs = {}

        # Check if all outputs are scalar variables
        all_scalars = all(
            isinstance(v, TorchScalarVariable) for v in self.output_variables
        )

        # Handle 0D and 1D tensors
        if output_tensor.dim() == 0:
            # 0D tensor - always add batch dimension at start
            output_tensor = output_tensor.unsqueeze(0)
        elif output_tensor.dim() == 1:
            # 1D tensor - interpretation depends on variable types
            if all_scalars and len(self.output_names) == 1:
                # For single scalar output, 1D means (batch,) -> should become (batch, 1)
                output_tensor = output_tensor.unsqueeze(-1)
            elif all_scalars and len(self.output_names) > 1:
                # For multiple scalar outputs, 1D means (features,) -> should become (1, features)
                output_tensor = output_tensor.unsqueeze(0)
            else:
                # For non-scalar outputs, default to adding batch dimension at start
                output_tensor = output_tensor.unsqueeze(0)

        if len(self.output_names) == 1:
            output = output_tensor
            # For scalar variables, ensure shape is (batch, 1) for single-sample batched outputs
            # or (batch, samples) for multi-sample outputs
            if all_scalars:
                if output.dim() == 2:
                    # Already 2D: could be (batch, 1) or (batch, samples) - keep as is
                    parsed_outputs[self.output_names[0]] = output
                elif output.dim() == 1:
                    # Shape is (batch,), reshape to (batch, 1)
                    parsed_outputs[self.output_names[0]] = output.unsqueeze(-1)
                else:
                    # 3D or higher dimensional - squeeze last dim if it's 1
                    # This handles multi-sample cases: (batch, samples, 1) -> (batch, samples)
                    if output.shape[-1] == 1:
                        parsed_outputs[self.output_names[0]] = output.squeeze(-1)
                    else:
                        # Shouldn't happen, but handle by squeezing all and adding feature dim
                        parsed_outputs[self.output_names[0]] = (
                            output.squeeze().unsqueeze(-1)
                            if output.squeeze().dim() > 0
                            else output.squeeze().unsqueeze(0).unsqueeze(-1)
                        )
            else:
                # For non-scalar outputs (NDVariable), keep original behavior
                parsed_outputs[self.output_names[0]] = output.squeeze()
        else:
            for idx, output_name in enumerate(self.output_names):
                output = output_tensor[..., idx]
                var = self.output_variables[idx]

                # For scalar variables, ensure shape is (batch, 1) for batched outputs
                if isinstance(var, TorchScalarVariable):
                    if output.dim() == 1:
                        # Shape is (batch,), reshape to (batch, 1)
                        parsed_outputs[output_name] = output.unsqueeze(-1)
                    elif output.dim() == 0:
                        # Scalar output, reshape to (1, 1)
                        parsed_outputs[output_name] = output.unsqueeze(0).unsqueeze(-1)
                    else:
                        # Already has proper dimensions or higher, ensure last dim is 1
                        parsed_outputs[output_name] = (
                            output.squeeze().unsqueeze(-1)
                            if output.squeeze().dim() > 0
                            else output.squeeze().unsqueeze(0).unsqueeze(-1)
                        )
                else:
                    # For non-scalar outputs (NDVariable), keep original behavior
                    parsed_outputs[output_name] = output.squeeze()

        return parsed_outputs

    def _prepare_outputs(
        self,
        parsed_outputs: dict[str, torch.Tensor],
    ) -> dict[str, Union[float, torch.Tensor]]:
        """Update and return outputs according to ``output_format``.

        Updates the output variables within the model to reflect the new
        values.

        Parameters
        ----------
        parsed_outputs : dict of str to torch.Tensor
            Dictionary of output variable names to transformed tensors.

        Returns
        -------
        dict of str to float or torch.Tensor
            Dictionary of output variable names to values depending on
            ``output_format``.

        """
        if self.output_format.lower() == "tensor":
            return parsed_outputs
        else:
            return {
                key: value.item() if value.squeeze().dim() == 0 else value
                for key, value in parsed_outputs.items()
            }
