import os
import json
import yaml
import logging
import inspect
from typing import Union, Any

import torch

from lume_torch.base import parse_config, recursive_serialize
from lume_torch.models.torch_model import TorchModel
from lume_torch.mlflow_utils import register_model
from lume_torch.variables import TorchScalarVariable, TorchNDVariable

logger = logging.getLogger(__name__)


class TorchModule(torch.nn.Module):
    """Wrapper to allow a LUME TorchModel to be used like a torch.nn.Module.

    As the base model within the TorchModel is assumed to be fixed during instantiation,
    so is the TorchModule.

    """

    def __init__(
        self,
        *args,
        model: TorchModel = None,
        input_order: list[str] = None,
        output_order: list[str] = None,
    ):
        """Initializes TorchModule.

        Parameters
        ----------
        *args : dict, str, or Path
            Accepts a single argument which is the model configuration as dictionary, YAML or JSON
            formatted string or file path.
        model : TorchModel, optional
            The TorchModel instance to wrap around. If config is None, this has to be defined.
        input_order : list of str, optional
            Input names in the order they are passed to the model. If None, the input order of the
            TorchModel is used.
        output_order : list of str, optional
            Output names in the order they are returned by the model. If None, the output order of
            the TorchModel is used.

        """
        if all(arg is None for arg in [*args, model]):
            logger.error("TorchModule requires either a YAML config or model argument")
            raise ValueError(
                "Either a YAML string has to be given or model has to be defined."
            )
        super().__init__()
        if len(args) == 1:
            if not all(v is None for v in [model, input_order, output_order]):
                logger.error(
                    "Cannot specify both YAML config and keyword arguments for TorchModule"
                )
                raise ValueError(
                    "Cannot specify YAML string and keyword arguments for TorchModule init."
                )
            logger.debug("Initializing TorchModule from configuration file")
            model_fields = {f"model.{k}": v for k, v in TorchModel.model_fields.items()}
            kwargs = parse_config(args[0], model_fields)
            kwargs["model"] = TorchModel(kwargs["model"])
            self.__init__(**kwargs)
        elif len(args) > 1:
            logger.error(f"Too many positional arguments to TorchModule: {len(args)}")
            raise ValueError(
                "Arguments to TorchModule must be either a single YAML string or keyword arguments."
            )
        else:
            logger.debug(f"Initializing TorchModule with model: {type(model).__name__}")
            if model.output_format != "tensor":
                logger.warning(
                    f"TorchModule requires output_format='tensor', "
                    f"but got '{model.output_format}'. Switching to 'tensor'."
                )
                model.output_format = "tensor"
            self._model = model
            self._input_order = input_order
            self._output_order = output_order
            self.register_module("base_model", self._model.model)
            logger.debug(
                f"Registered {len(self._model.input_transformers)} input transformers"
            )
            for i, input_transformer in enumerate(self._model.input_transformers):
                self.register_module(f"input_transformers_{i}", input_transformer)
            logger.debug(
                f"Registered {len(self._model.output_transformers)} output transformers"
            )
            for i, output_transformer in enumerate(self._model.output_transformers):
                self.register_module(f"output_transformers_{i}", output_transformer)
            if not model.model.training:  # TorchModel defines train/eval mode
                self.eval()
            logger.info(
                f"Initialized TorchModule with {len(self.input_order)} inputs and {len(self.output_order)} outputs"
            )

    @property
    def model(self):
        return self._model

    @property
    def input_order(self):
        if self._input_order is None:
            return self._model.input_names
        else:
            return self._input_order

    @property
    def output_order(self):
        if self._output_order is None:
            return self._model.output_names
        else:
            return self._output_order

    @property
    def _nd_inputs(self) -> bool:
        """True when every input variable is a TorchNDVariable."""
        return all(isinstance(v, TorchNDVariable) for v in self._model.input_variables)

    @property
    def _nd_outputs(self) -> bool:
        """True when every output variable is a TorchNDVariable."""
        return all(isinstance(v, TorchNDVariable) for v in self._model.output_variables)

    @property
    def _scalar_inputs(self) -> bool:
        """True when every input variable is a TorchScalarVariable."""
        return all(
            isinstance(v, TorchScalarVariable) for v in self._model.input_variables
        )

    @property
    def _scalar_outputs(self) -> bool:
        """True when every output variable is a TorchScalarVariable."""
        return all(
            isinstance(v, TorchScalarVariable) for v in self._model.output_variables
        )

    def forward(self, x: torch.Tensor):
        x = self._validate_input(x)
        model_input = self._tensor_to_dictionary(x)
        y_model = self.evaluate_model(model_input)
        y_model = self.manipulate_output(y_model)
        y = self._dictionary_to_tensor(y_model)
        if self._scalar_outputs:
            # Remove trailing output-feature dim for BoTorch Mean compatibility
            # (K=1: (batch, 1) -> (batch,); K>1: (batch, K) unchanged)
            y = y.squeeze(-1)
        return y

    def yaml(
        self,
        base_key: str = "",
        file_prefix: str = "",
        save_models: bool = False,
        save_jit: bool = False,
    ) -> str:
        """Serializes the object and returns a YAML formatted string defining the TorchModule instance.

        Parameters
        ----------
        base_key : str, optional
            Base key for serialization.
        file_prefix : str, optional
            Prefix for generated filenames.
        save_models : bool, optional
            Determines whether models are saved to file.
        save_jit : bool, optional
            Determines whether the structure of the model is saved as TorchScript

        Returns
        -------
        str
            YAML formatted string defining the TorchModule instance.

        """
        d = {}
        for k, v in inspect.signature(TorchModule.__init__).parameters.items():
            if k not in ["self", "args", "model"]:
                d[k] = getattr(self, k)
        output = json.loads(
            json.dumps(
                recursive_serialize(d, base_key, file_prefix, save_models, save_jit)
            )
        )
        model_output = json.loads(
            self._model.to_json(
                base_key=base_key,
                file_prefix=file_prefix,
                save_models=save_models,
                save_jit=save_jit,
            )
        )
        output["model"] = model_output
        # create YAML formatted string
        s = yaml.dump(
            {"model_class": self.__class__.__name__} | output,
            default_flow_style=None,
            sort_keys=False,
        )
        return s

    def dump(
        self,
        file: Union[str, os.PathLike],
        save_models: bool = True,
        base_key: str = "",
        save_jit: bool = False,
    ):
        """Returns and optionally saves YAML formatted string defining the model.

        Parameters
        ----------
        file : str or Path
            File path to which the YAML formatted string and corresponding files are saved.
        save_models : bool, optional
            Determines whether models are saved to file.
        base_key : str, optional
            Base key for serialization.
        save_jit : bool, optional
            Whether the model is saved using just in time pytorch method

        """
        logger.info(f"Dumping TorchModule configuration to: {file}")
        if save_models:
            logger.debug("Saving model files alongside configuration")
        if save_jit:
            logger.debug("Saving TorchModule as TorchScript (JIT)")
        file_prefix = os.path.splitext(file)[0]
        with open(file, "w") as f:
            f.write(
                self.yaml(
                    save_models=save_models,
                    base_key=base_key,
                    file_prefix=file_prefix,
                    save_jit=save_jit,
                )
            )

    def evaluate_model(self, x: dict[str, torch.Tensor]):
        """Placeholder method to modify model calls.

        Parameters
        ----------
        x : dict of str to torch.Tensor
            Input dictionary to evaluate.

        Returns
        -------
        dict of str to torch.Tensor
            Model evaluation results.

        """
        return self._model.evaluate(x)

    def manipulate_output(self, y_model: dict[str, torch.Tensor]):
        """Placeholder method to modify the model output.

        Parameters
        ----------
        y_model : dict of str to torch.Tensor
            Model output dictionary.

        Returns
        -------
        dict of str to torch.Tensor
            Modified model output.

        """
        return y_model

    def _tensor_to_dictionary(self, x: torch.Tensor):
        input_dict = {}
        if self._nd_inputs:
            # ND format: (batch, num_vars, *array_shape) — slice dim 1 per variable
            for idx, input_name in enumerate(self.input_order):
                input_dict[input_name] = x[:, idx, ...]
        elif x.ndim >= 3 and x.shape[-1] == 1:
            # Scalar new format: (..., n_features, 1) — index second-to-last dim
            for idx, input_name in enumerate(self.input_order):
                input_dict[input_name] = x[..., idx, :]
        else:
            # Scalar old format: (..., n_features) — index last dim, add trailing 1
            for idx, input_name in enumerate(self.input_order):
                input_dict[input_name] = x[..., idx].unsqueeze(-1)
        return input_dict

    def _dictionary_to_tensor(self, y_model: dict[str, torch.Tensor]):
        output_list = []
        for output_name in self.output_order:
            output_list.append(y_model[output_name])

        if self._nd_outputs:
            # ND outputs: stack along dim 1 -> (batch, num_outputs, *array_shape)
            return torch.stack(output_list, dim=1)
        else:
            # Scalar outputs: squeeze trailing feature dim then stack along last dim
            # -> (batch, K) for K outputs
            squeezed = [o.squeeze(-1) if o.shape[-1] == 1 else o for o in output_list]
            return torch.stack(squeezed, dim=-1)

    def _validate_input(self, x: torch.Tensor) -> torch.Tensor:
        if self._nd_inputs:
            # ND inputs: (batch, num_vars, *array_shape)
            # Minimum ndim = 1 (batch) + 1 (num_vars) + len(array_shape of first var)
            first_var = self._model.input_variables[0]
            min_ndim = 2 + len(first_var.shape)
            if x.dim() < min_ndim:
                logger.error(
                    f"Invalid input dimensions for ND variables: expected at least {min_ndim}D "
                    f"([batch, num_vars, *array_shape] where array_shape={first_var.shape}), "
                    f"got {tuple(x.shape)}"
                )
                raise ValueError(
                    f"Expected input dim to be at least {min_ndim} "
                    f"([batch, num_vars, *array_shape] where array_shape={first_var.shape}) "
                    f"for TorchNDVariable inputs, received: {tuple(x.shape)}"
                )
        else:
            # Scalar inputs: expect at least 2D ([batch, n_features] or [batch, n_features, 1])
            if x.dim() <= 1:
                logger.error(
                    f"Invalid input dimensions: expected at least 2D ([n_samples, n_features]), "
                    f"got {tuple(x.shape)}"
                )
                raise ValueError(
                    f"Expected input dim to be at least 2 ([n_samples, n_features]), "
                    f"received: {tuple(x.shape)}"
                )
        return x

    def register_to_mlflow(
        self,
        artifact_path: str,
        registered_model_name: str | None = None,
        tags: dict[str, Any] | None = None,
        version_tags: dict[str, Any] | None = None,
        alias: str | None = None,
        run_name: str | None = None,
        log_model_dump: bool = True,
        save_jit: bool = False,
        **kwargs,
    ):
        """Registers the model to MLflow if mlflow is installed.

        Each time this function is called, a new version of the model is created. The model is saved to the
        tracking server or local directory, depending on the MLFLOW_TRACKING_URI.

        If no tracking server is set up, data and artifacts are saved directly under your current directory. To set up
        a tracking server, set the environment variable MLFLOW_TRACKING_URI, e.g. a local port/path. See
        https://mlflow.org/docs/latest/getting-started/intro-quickstart/ for more info.

        Parameters
        ----------
        artifact_path : str
            Path to store the model in MLflow.
        registered_model_name : str or None, optional
            Name of the registered model in MLflow.
        tags : dict of str to Any or None, optional
            Tags to add to the MLflow model.
        version_tags : dict of str to Any or None, optional
            Tags to add to this MLflow model version.
        alias : str or None, optional
            Alias to add to this MLflow model version.
        run_name : str or None, optional
            Name of the MLflow run.
        log_model_dump : bool, optional
            Whether to log the model dump files as artifacts.
        save_jit : bool, optional
            Whether to save the model as TorchScript when calling model.dump, if log_model_dump=True.
        **kwargs
            Additional arguments for mlflow.pyfunc.log_model.

        Returns
        -------
        mlflow.models.model.ModelInfo
            Model info metadata.

        """
        return register_model(
            self,
            artifact_path,
            registered_model_name,
            tags,
            version_tags,
            alias,
            run_name,
            log_model_dump,
            save_jit,
            **kwargs,
        )


class FixedVariableModel(torch.nn.Module):
    """Prior model for Bayesian optimization.

    This module wraps a LUME model and manages the separation between control variables
    and fixed variables (measured from the machine). It also maintains
    an efficient buffer of fixed variables that is updated periodically.
    The prior model is used as a mean function in Gaussian process models to incorporate
    physics knowledge from the LUME surrogate model into the Bayesian optimization process.

    Parameters
    ----------
    model : TorchModule
        LUME model that takes all input variables and produces outputs.
        The model's input order is obtained via model.input_variables.
    fixed_values : dict
        Dictionary mapping PV names to their initial measured values
        for all non-control variables. Keys should be PV names (str), values should be
        floats. These represent the initial state of variables not being optimized.

    Attributes
    ----------
    model : TorchModule
        The LUME surrogate model.
    all_inputs : list
        Ordered list of all input variable names from the LUME model.
    control_variables : list
        List of control variable names, derived as all_inputs - fixed_values.
    input_buffer : torch.Tensor
        1D tensor storing the current values of all inputs.
        Shape: (n_total_inputs,). This is updated when fixed variables change.
    control_indices : torch.Tensor
        1D tensor of indices for control variables in the full input tensor.
        Shape: (n_control_vars,). Used for fast indexing.
    fixed_indices : list
        List of indices for fixed variables in the full input tensor.

    """

    def __init__(self, model: TorchModule, fixed_values, control_variables):
        """Initialize the FixedVariableModel class.

        This constructor sets up the model wrapper that separates control variables (to be optimized) from fixed
        variables (measured from machine state). It creates an efficient buffer structure for fast forward passes
        during optimization.

        Parameters
        ----------
        model : TorchModule
            The LUME surrogate model
        fixed_values : dict of str to float
            Dictionary mapping PV (process variable) names to their initial measured values
            for all non-control variables
        control_variable : list of control variables (to be vocs.variable_names to preserve the
        variable ordering in xopt vocs)

        """
        super(FixedVariableModel, self).__init__()
        self.model = model
        self.all_inputs = list(model.input_order)
        self.control_variables = control_variables

        # Create a buffer tensor to store the full input template
        # This is updated ONCE when fixed variables change, not on every forward call
        self.register_buffer("input_buffer", torch.zeros(len(self.all_inputs)))

        # Pre-compute indices for fast lookup (computed once, used many times)
        # Store as buffer so it moves with the model to different devices
        self.register_buffer(
            "control_indices",
            torch.tensor(
                [self.all_inputs.index(var) for var in self.control_variables],
                dtype=torch.long,
            ),
        )
        self.fixed_indices = [self.all_inputs.index(var) for var in fixed_values.keys()]

        # Initialize buffer with fixed variables
        self.update_fixed_values(fixed_values)

        logger.info("FixedVariableModel initialized")
        logger.info(f"  Total inputs (from model): {len(self.all_inputs)}")
        logger.info(f"  Fixed variables: {len(self.fixed_indices)}")
        logger.info(f"  Control variables (derived): {len(self.control_variables)}")
        logger.debug(f"  Control variables: {self.control_variables}")
        logger.debug(f"  Control indices: {self.control_indices}")

    def update_fixed_values(self, fixed_values):
        """Update the buffer with new fixed variable values.

        This method directly updates the input_buffer tensor with new values for
        fixed variables. It should be called when fixed variable measurements change.

        Parameters
        ----------
        fixed_values : dict
            Dictionary mapping PV names to their new measured values.
            Keys should be PV names (str) that exist in self.all_inputs and are NOT
            control variables. Values should be floats.

        """
        logger.debug(f"Updating {len(fixed_values)} fixed variable values")
        for var_name, value in fixed_values.items():
            idx = self.all_inputs.index(var_name)
            self.input_buffer[idx] = value

    def forward(self, x) -> torch.Tensor:
        """Forward pass through the LUME model with control and fixed variables.

        Parameters
        ----------
        x : torch.Tensor
            Tensor containing only control variable values.
            Can have arbitrary batch dimensions.
            The last dimension must match len(self.control_variables).

        Returns
        -------
        torch.Tensor
            Output from the LUME model. Shape depends on the model's
            output structure and the input batch dimensions.

        """
        batch_shape = x.shape[
            :-1
        ]  # Get batch shape (everything except the last dimension)

        # Expand buffer to match batch dimensions
        expanded_buffer = self.input_buffer.view(*([1] * len(batch_shape)), -1).expand(
            *batch_shape, -1
        )

        # Clone to make it writable
        full_input = expanded_buffer.clone()

        # Scatter control values into the full input tensor
        # scatter_(dim, index, src)
        # We want to scatter along the last dimension
        indices_expanded = self.control_indices.view(
            *([1] * len(batch_shape)), -1
        ).expand(*batch_shape, -1)

        full_input.scatter_(dim=-1, index=indices_expanded, src=x)

        # Call LUME model
        return self.model(full_input)
