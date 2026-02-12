import os
import logging
from typing import Union, Callable
from copy import deepcopy

import torch
from pydantic import field_validator
from botorch.models.transforms.input import ReversibleInputTransform

from lume_torch.base import LUMETorch
from lume_torch.variables import ScalarVariable
from lume_torch.models.utils import itemize_dict, format_inputs, InputDictModel

logger = logging.getLogger(__name__)


class TorchModel(LUMETorch):
    """LUME-model class for torch models.

    By default, the models are assumed to be fixed, so all gradient computation
    is deactivated and the model and transformers are put in evaluation mode.

    Attributes
    ----------
    model : torch.nn.Module
        The underlying torch model.
    input_variables : list of ScalarVariable
        List defining the input variables and their order.
    output_variables : list of ScalarVariable
        List defining the output variables and their order.
    input_transformers : list of callable or modules
        Transformer objects applied to the inputs before passing to the model.
    output_transformers : list of callable or modules
        Transformer objects applied to the outputs of the model.
    output_format : {"tensor", "variable", "raw"}
        Determines format of outputs.
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
    def validate_torch_model(cls, v):
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
    def verify_input_default_value(cls, value):
        """Verifies that input variables have the required default values."""
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
    def validate_transformers(cls, v):
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
    def validate_output_format(cls, v):
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
        """Sets the precision of the model."""
        self.model.to(dtype=value)
        for t in self.input_transformers + self.output_transformers:
            if isinstance(t, torch.nn.Module):
                t.to(dtype=value)

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
        # validate input type (ints only are cast to floats for scalars)
        validated_input = InputDictModel(input_dict=input_dict).input_dict
        # format inputs as tensors w/o changing the dtype
        formatted_inputs = format_inputs(validated_input)
        # check default values for missing inputs
        filled_inputs = self._fill_default_inputs(formatted_inputs)
        # itemize inputs for validation
        itemized_inputs = itemize_dict(filled_inputs)

        for ele in itemized_inputs:
            # validate values that were in the torch tensor
            # any ints in the torch tensor will be cast to floats by Pydantic
            # but others will be caught, e.g. booleans
            ele = InputDictModel(input_dict=ele).input_dict
            # validate each value based on its var class and config
            super().input_validation(ele)

        # return the validated input dict for consistency w/ casting ints to floats
        if any([isinstance(value, torch.Tensor) for value in validated_input.values()]):
            validated_input = {
                k: v.to(**self._tkwargs) for k, v in validated_input.items()
            }

        return validated_input

    def output_validation(self, output_dict: dict[str, Union[float, torch.Tensor]]):
        """Itemize tensors before performing output validation.

        Parameters
        ----------
        output_dict : dict of str to float or torch.Tensor
            Output dictionary to validate.

        """
        itemized_outputs = itemize_dict(output_dict)
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

        """
        input_dict = {}
        for var in self.input_variables:
            if isinstance(var, ScalarVariable):
                input_dict[var.name] = var.value_range[0] + torch.rand(
                    size=(n_samples,)
                ) * (var.value_range[1] - var.value_range[0])
            else:
                torch.tensor(var.default_value, **self._tkwargs).repeat((n_samples, 1))
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
    ) -> list[ScalarVariable]:
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
        list of ScalarVariable
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
                input_dict[var.name] = torch.tensor(var.default_value, **self._tkwargs)
        return input_dict

    def _arrange_inputs(
        self, formatted_inputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Enforce the order of input variables.

        Enforces the order of the input variables to be passed to the
        transformers and model and updates the returned tensor with default
        values for any inputs that are missing.

        Parameters
        ----------
        formatted_inputs : dict of str to torch.Tensor
            Dictionary of input variable names to tensors.

        Returns
        -------
        torch.Tensor
            Ordered input tensor to be passed to the transformers.

        """
        default_tensor = torch.tensor(
            [var.default_value for var in self.input_variables], **self._tkwargs
        )

        # determine input shape
        input_shapes = [formatted_inputs[k].shape for k in formatted_inputs.keys()]
        if not all(ele == input_shapes[0] for ele in input_shapes):
            raise ValueError("Inputs have inconsistent shapes.")

        input_tensor = torch.tile(default_tensor, dims=(*input_shapes[0], 1))
        for key, value in formatted_inputs.items():
            input_tensor[..., self.input_names.index(key)] = value

        if input_tensor.shape[-1] != len(self.input_names):
            raise ValueError(
                f"""
                Last dimension of input tensor doesn't match the expected number of inputs\n
                received: {default_tensor.shape}, expected {len(self.input_names)} as the last dimension
                """
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
