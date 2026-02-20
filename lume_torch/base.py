import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union
from types import FunctionType, MethodType
from io import TextIOWrapper

import yaml
import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from lume_torch.variables import ScalarVariable, get_variable, ConfigEnum
from lume_torch.utils import (
    try_import_module,
    verify_unique_variable_names,
    serialize_variables,
    deserialize_variables,
    variables_from_dict,
    replace_relative_paths,
)
from lume_torch.mlflow_utils import register_model

from lume.model import LUMEModel
from lume.variables import Variable
import importlib

logger = logging.getLogger(__name__)

JSON_ENCODERS = {
    # function/method type distinguished for class members and not recognized as callables
    FunctionType: lambda x: f"{x.__module__}.{x.__qualname__}",
    MethodType: lambda x: f"{x.__module__}.{x.__qualname__}",
    Callable: lambda x: f"{x.__module__}.{x.__qualname__}",
    type: lambda x: f"{x.__module__}.{x.__name__}",
    np.ndarray: lambda x: x.tolist(),
    np.int64: lambda x: int(x),
    np.float64: lambda x: float(x),
}


def process_torch_module(
    module,
    base_key: str = "",
    key: str = "",
    file_prefix: Union[str, os.PathLike] = "",
    save_modules: bool = True,
    save_jit: bool = False,
):
    """Optionally saves the given torch module to file and returns the filename.

    Parameters
    ----------
    module : torch.nn.Module
        The torch module to process.
    base_key : str, optional
        Base key at this stage of serialization.
    key : str, optional
        Key corresponding to the torch module.
    file_prefix : str or os.PathLike, optional
        Prefix for generated filenames.
    save_modules : bool, optional
        Determines whether torch modules are saved to file.
    save_jit : bool, optional
        Determines whether the model gets saved as TorchScript.

    Returns
    -------
    str
        Filename under which the torch module is (or would be) saved.

    """
    torch = try_import_module("torch")
    filepath_prefix, filename_prefix = os.path.split(file_prefix)
    prefixes = [ele for ele in [filename_prefix, base_key] if not ele == ""]
    filename = "{}.pt".format(key)
    jit_filename = "{}.jit".format(key)
    if prefixes:
        filename = "_".join((*prefixes, filename))
        jit_filename = "_".join((*prefixes, jit_filename))
    if save_modules:
        filepath = os.path.join(filepath_prefix, filename)
        torch.save(module, filepath)
        logger.debug(f"Saved torch module to: {filepath}")
    if save_jit:
        filepath = os.path.join(filepath_prefix, jit_filename)
        try:
            scripted_model = torch.jit.script(module)
            torch.jit.save(scripted_model, filepath)
            logger.debug(f"Saved JIT model to: {filepath}")
        except Exception as e:
            logger.warning(
                "Saving as JIT through scripting has only been evaluated "
                "for NN models that don't depend on BoTorch modules."
            )
            logger.error(f"Failed to script the model: {e}")
            raise e
    return jit_filename if save_jit else filename


def recursive_serialize(
    v: dict[str, Any],
    base_key: str = "",
    file_prefix: Union[str, os.PathLike] = "",
    save_models: bool = True,
    save_jit: bool = False,
):
    """Recursively performs custom serialization for the given object.

    Parameters
    ----------
    v : dict of str to Any
        Object to serialize.
    base_key : str, optional
        Base key at this stage of serialization.
    file_prefix : str or os.PathLike, optional
        Prefix for generated filenames.
    save_models : bool, optional
        Determines whether models are saved to file.
    save_jit : bool, optional
        Determines whether the model is saved as TorchScript.

    Returns
    -------
    dict
        Serialized object.

    """
    logger.debug(
        f"Serializing object with base_key: '{base_key}', {len(v)} top-level keys"
    )
    # try to import modules for LUMETorch child classes
    torch = try_import_module("torch")
    # serialize
    v = serialize_variables(v)
    for key, value in v.items():
        if isinstance(value, dict):
            logger.debug(f"Recursively serializing nested dict for key: '{key}'")
            v[key] = recursive_serialize(value, key)
        elif isinstance(value, list) and all(isinstance(ele, dict) for ele in value):
            # e.g. NN ensemble
            logger.debug(
                f"Serializing NN ensemble with {len(value)} models for key: '{key}'"
            )
            v[key] = [
                recursive_serialize(
                    value[i], f"{base_key}{i}", file_prefix, save_models, save_jit
                )
                for i in range(len(value))
            ]
            # For NN ensembles, we want v[key] to be a list of the filenames corresponding to each
            # model in the ensemble and not the serialized dict of each
            # NOTE: If this clause is reached for other models, we may need to do this differently
            v[key] = [v[key][i]["model"] for i in range(len(value))]
        elif torch is not None and isinstance(value, torch.nn.Module):
            logger.debug(f"Serializing torch.nn.Module for key: '{key}'")
            v[key] = process_torch_module(
                value, base_key, key, file_prefix, save_models, save_jit
            )
        elif (
            isinstance(value, list)
            and torch is not None
            and any(isinstance(ele, torch.nn.Module) for ele in value)
        ):
            # List of transformers
            logger.debug(
                f"Serializing {len(value)} torch.nn.Module transformers for key: '{key}'"
            )
            v[key] = [
                process_torch_module(
                    value[i], base_key, f"{key}_{i}", file_prefix, save_models, False
                )
                for i in range(len(value))
            ]
        else:
            for _type, func in JSON_ENCODERS.items():
                if isinstance(value, _type):
                    logger.debug(
                        f"Applying JSON encoder for type {_type.__name__} to key: '{key}'"
                    )
                    v[key] = func(value)
        # check to make sure object has been serialized, if not use a generic serializer
        try:
            json.dumps(v[key])
        except (TypeError, OverflowError):
            logger.debug(
                f"Using generic serializer for unserializable object at key: '{key}'"
            )
            v[key] = f"{v[key].__module__}.{v[key].__class__.__qualname__}"

    return v


def recursive_deserialize(v):
    """Recursively performs custom deserialization for the given object.

    Parameters
    ----------
    v : dict
        Object to deserialize.

    Returns
    -------
    dict
        Deserialized object.

    """
    logger.debug(f"Deserializing object with {len(v)} top-level keys")
    # deserialize
    v = deserialize_variables(v)
    for key, value in v.items():
        if isinstance(value, dict):
            logger.debug(f"Recursively deserializing nested dict for key: '{key}'")
            v[key] = recursive_deserialize(value)
    return v


def json_dumps(
    v,
    *,
    base_key="",
    file_prefix: Union[str, os.PathLike] = "",
    save_models: bool = True,
    save_jit: bool = False,
):
    """Serializes variables before dumping with json.

    Parameters
    ----------
    v : object
        Object to dump.
    base_key : str, optional
        Base key for serialization.
    file_prefix : str or os.PathLike, optional
        Prefix for generated filenames.
    save_models : bool, optional
        Determines whether models are saved to file.
    save_jit : bool, optional
        Determines whether the model is saved as TorchScript.

    Returns
    -------
    str
        JSON formatted string.

    """
    v = recursive_serialize(
        v.model_dump(), base_key, file_prefix, save_models, save_jit
    )
    v = json.dumps(v)
    return v


def json_loads(v):
    """Loads JSON formatted string and recursively deserializes the result.

    Parameters
    ----------
    v : str
        JSON formatted string to load.

    Returns
    -------
    dict
        Deserialized object.

    """
    v = json.loads(v)
    v = recursive_deserialize(v)
    return v


def parse_config(
    config: Union[dict, str, TextIOWrapper, os.PathLike],
    model_fields: dict = None,
) -> dict:
    """Parses model configuration and returns keyword arguments for model constructor.

    Parameters
    ----------
    config : dict, str, TextIOWrapper, or os.PathLike
        Model configuration as dictionary, YAML or JSON formatted string, file or file path.
    model_fields : dict, optional
        Fields expected by the model (required for replacing relative paths).

    Returns
    -------
    dict
        Configuration as keyword arguments for model constructor.

    """
    config_file = None
    if isinstance(config, dict):
        logger.debug("Parsing configuration from dictionary")
        d = config
    else:
        if isinstance(config, TextIOWrapper):
            logger.debug(f"Reading configuration from file wrapper: {config.name}")
            yaml_str = config.read()
            config_file = os.path.abspath(config.name)
        elif isinstance(config, (str, os.PathLike)) and os.path.exists(config):
            logger.debug(f"Loading configuration from file: {config}")
            with open(config) as f:
                yaml_str = f.read()
            config_file = os.path.abspath(config)
        else:
            logger.debug("Parsing configuration from YAML string")
            yaml_str = config
        d = recursive_deserialize(yaml.safe_load(yaml_str))
    if config_file is not None:
        config_dir = os.path.dirname(os.path.realpath(config_file))
        logger.debug(f"Replacing relative paths using config directory: {config_dir}")
        d = replace_relative_paths(d, model_fields, config_dir)
    return model_kwargs_from_dict(d)


def model_kwargs_from_dict(config: dict) -> dict:
    """Processes model configuration and returns the corresponding keyword arguments for model constructor.

    Parameters
    ----------
    config : dict
        Model configuration.

    Returns
    -------
    dict
        Configuration as keyword arguments for model constructor.

    """
    config = deserialize_variables(config)
    if all(key in config.keys() for key in ["input_variables", "output_variables"]):
        config["input_variables"], config["output_variables"] = variables_from_dict(
            config
        )
    config.pop("model_class", None)
    return config


class LUMETorch(BaseModel, ABC):
    """Abstract base class for models using lume-torch variables.

    Inheriting classes must define the _evaluate method and variable names must be unique.
    Models built using this framework will be compatible with the lume-epics EPICS server and associated tools.

    Attributes
    ----------
    input_variables : list of ScalarVariable
        List defining the input variables and their order.
    output_variables : list of ScalarVariable
        List defining the output variables and their order.
    input_validation_config : dict of str to ConfigEnum, optional
        Determines the behavior during input validation by specifying the validation
        config for each input variable: {var_name: value}. Value can be "warn", "error", or "none".
    output_validation_config : dict of str to ConfigEnum, optional
        Determines the behavior during output validation by specifying the validation
        config for each output variable: {var_name: value}. Value can be "warn", "error", or "none".

    Methods
    -------
    evaluate(input_dict, **kwargs)
        Main evaluation function that validates inputs, calls _evaluate, and validates outputs.
    input_validation(input_dict)
        Validates input dictionary values against input variable specifications.
    output_validation(output_dict)
        Validates output dictionary values against output variable specifications.
    yaml(base_key="", file_prefix="", save_models=False, save_jit=False)
        Serializes the model to a YAML formatted string.
    dump(file, base_key="", save_models=True, save_jit=False)
        Saves model configuration and associated files to disk.
    from_file(filename)
        Class method to load a model from a YAML file.
    from_yaml(yaml_obj)
        Class method to load a model from a YAML string or file object.
    register_to_mlflow(artifact_path, **kwargs)
        Registers the model to MLflow for experiment tracking.

    Notes
    -----
    Subclasses must implement the abstract method `_evaluate(input_dict, **kwargs)` which performs
    the actual model computation.

    """

    input_variables: list[ScalarVariable]
    output_variables: list[ScalarVariable]
    input_validation_config: Optional[dict[str, ConfigEnum]] = None
    output_validation_config: Optional[dict[str, ConfigEnum]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @field_validator("input_variables", "output_variables", mode="before")
    def validate_input_variables(cls, value):
        """Validates and converts input/output variables to proper format.

        Parameters
        ----------
        value : dict or list
            Variables as dictionary or list to validate and convert.

        Returns
        -------
        list of ScalarVariable
            List of validated variable instances.

        Raises
        ------
        TypeError
            If variable type is not supported.

        """
        new_value = []
        if isinstance(value, dict):
            for name, val in value.items():
                if isinstance(val, dict):
                    variable_class = get_variable(val["variable_class"])
                    new_value.append(variable_class(name=name, **val))
                elif isinstance(val, ScalarVariable):
                    new_value.append(val)
                else:
                    raise TypeError(f"type {type(val)} not supported")
        elif isinstance(value, list):
            new_value = value
        return new_value

    def __init__(self, *args, **kwargs):
        """Initializes LUMETorch.

        Parameters
        ----------
        *args : dict, str, or os.PathLike
            Accepts a single argument which is the model configuration as dictionary, YAML or JSON
            formatted string or file path.
        **kwargs
            See class attributes.

        Raises
        ------
        ValueError
            If both YAML config and keyword arguments are provided, or if more than one
            positional argument is provided.

        """
        if len(args) == 1:
            if len(kwargs) > 0:
                logger.error("Cannot specify both YAML config and keyword arguments")
                raise ValueError(
                    "Cannot specify YAML string and keyword arguments for LUMETorch init."
                )
            logger.debug("Initializing model from configuration")
            super().__init__(**parse_config(args[0], type(self).model_fields))
        elif len(args) > 1:
            logger.error(f"Too many positional arguments: {len(args)}")
            raise ValueError(
                "Arguments to LUMETorch must be either a single YAML string "
                "or keyword arguments passed directly to pydantic."
            )
        else:
            logger.debug("Initializing model from keyword arguments")
            super().__init__(**kwargs)

        logger.info(
            f"Initialized {self.__class__.__name__} with {len(self.input_variables)} inputs and {len(self.output_variables)} outputs"
        )

    @field_validator("input_variables", "output_variables")
    def unique_variable_names(cls, value):
        verify_unique_variable_names(value)
        return value

    @property
    def input_names(self) -> list[str]:
        return [var.name for var in self.input_variables]

    @property
    def output_names(self) -> list[str]:
        return [var.name for var in self.output_variables]

    @property
    def default_input_validation_config(self) -> dict[str, ConfigEnum]:
        """Determines default behavior during input validation (if input_validation_config is None)."""
        return {var.name: var.default_validation_config for var in self.input_variables}

    @property
    def default_output_validation_config(self) -> dict[str, ConfigEnum]:
        """Determines default behavior during output validation (if output_validation_config is None)."""
        return {
            var.name: var.default_validation_config for var in self.output_variables
        }

    def evaluate(self, input_dict: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Main evaluation function, child classes must implement the _evaluate method."""
        validated_input_dict = self.input_validation(input_dict)
        output_dict = self._evaluate(validated_input_dict, **kwargs)
        self.output_validation(output_dict)
        return output_dict

    @abstractmethod
    def _evaluate(self, input_dict: dict[str, Any], **kwargs) -> dict[str, Any]:
        pass

    def input_validation(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Validates input dictionary values against input variable specifications.

        Parameters
        ----------
        input_dict : dict of str to Any
            Dictionary of input variable names to values.

        Returns
        -------
        dict of str to Any
            Validated input dictionary.

        """
        for name, value in input_dict.items():
            _config = (
                "none"
                if self.input_validation_config is None
                else self.input_validation_config.get(name)
            )
            var = self.input_variables[self.input_names.index(name)]
            var.validate_value(value, config=_config)
        return input_dict

    def output_validation(self, output_dict: dict[str, Any]) -> dict[str, Any]:
        """Validates output dictionary values against output variable specifications.

        Parameters
        ----------
        output_dict : dict of str to Any
            Dictionary of output variable names to values.

        Returns
        -------
        dict of str to Any
            Validated output dictionary.

        """
        for name, value in output_dict.items():
            _config = (
                None
                if self.output_validation_config is None
                else self.output_validation_config.get(name)
            )
            var = self.output_variables[self.output_names.index(name)]
            var.validate_value(value, config=_config)
        return output_dict

    def to_json(self, **kwargs) -> str:
        """Serializes the model to a JSON formatted string.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments for serialization (base_key, file_prefix, save_models, save_jit).

        Returns
        -------
        str
            JSON formatted string defining the model.

        """
        return json_dumps(self, **kwargs)

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Dumps the model configuration as a dictionary.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments for Pydantic's model_dump.

        Returns
        -------
        dict of str to Any
            Dictionary containing the model configuration including model_class name.

        """
        config = super().model_dump(**kwargs)
        config["input_variables"] = [var.model_dump() for var in self.input_variables]
        config["output_variables"] = [var.model_dump() for var in self.output_variables]
        return {"model_class": self.__class__.__name__} | config

    def json(self, **kwargs) -> str:
        """Serializes the model to a JSON formatted string.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments for serialization.

        Returns
        -------
        str
            JSON formatted string defining the model.

        """
        result = self.to_json(**kwargs)
        config = json.loads(result)
        return json.dumps(config)

    def yaml(
        self,
        base_key: str = "",
        file_prefix: str = "",
        save_models: bool = False,
        save_jit: bool = False,
    ) -> str:
        """Serializes the object and returns a YAML formatted string defining the model.

        Parameters
        ----------
        base_key : str, optional
            Base key for serialization.
        file_prefix : str, optional
            Prefix for generated filenames.
        save_models : bool, optional
            Determines whether models are saved to file.
        save_jit : bool, optional
            Determines whether the model is saved as TorchScript.

        Returns
        -------
        str
            YAML formatted string defining the model.

        """
        output = json.loads(
            self.to_json(
                base_key=base_key,
                file_prefix=file_prefix,
                save_models=save_models,
                save_jit=save_jit,
            )
        )
        s = yaml.dump(output, default_flow_style=None, sort_keys=False)
        return s

    def dump(
        self,
        file: Union[str, os.PathLike],
        base_key: str = "",
        save_models: bool = True,
        save_jit: bool = False,
    ):
        """Returns and optionally saves YAML formatted string defining the model.

        Parameters
        ----------
        file : str or os.PathLike
            File path to which the YAML formatted string and corresponding files are saved.
        base_key : str, optional
            Base key for serialization.
        save_models : bool, optional
            Determines whether models are saved to file.
        save_jit : bool, optional
            Determines whether the model is saved as TorchScript.

        """
        logger.info(f"Dumping model configuration to: {file}")
        if save_models:
            logger.debug("Saving model files alongside configuration")
        if save_jit:
            logger.debug("Saving models as TorchScript (JIT)")
        file_prefix = os.path.splitext(os.path.abspath(file))[0]
        with open(file, "w") as f:
            f.write(
                self.yaml(
                    base_key=base_key,
                    file_prefix=file_prefix,
                    save_models=save_models,
                    save_jit=save_jit,
                )
            )

    @classmethod
    def from_file(cls, filename: str):
        """Loads a model from a YAML file.

        Parameters
        ----------
        filename : str
            Path to the YAML file containing the model configuration.

        Returns
        -------
        LUMETorch
            Instance of the model loaded from the file.

        Raises
        ------
        OSError
            If the file does not exist.

        """
        if not os.path.exists(filename):
            raise OSError(f"File {filename} is not found.")
        with open(filename, "r") as file:
            return cls.from_yaml(file)

    @classmethod
    def from_yaml(cls, yaml_obj: [str, TextIOWrapper]):
        """Loads a model from a YAML string or file object.

        Parameters
        ----------
        yaml_obj : str or TextIOWrapper
            YAML formatted string or file object containing the model configuration.

        Returns
        -------
        LUMETorch
            Instance of the model loaded from the YAML configuration.

        """
        return cls.model_validate(parse_config(yaml_obj, cls.model_fields))

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


class LUMETorchModel(LUMEModel):
    """
    This class Subclasses LUMEModel to create a wrapper around a LUMETorch model.

    Parameters
    ----------
    torch_model: LUMETorch
        An instance of a LUMETorch model to wrap.
    """

    def __init__(
        self, torch_model: LUMETorch, initial_inputs: Optional[dict[str, Any]] = None
    ):
        """
         Initialize the LUMETorchModel.

        Parameters
        ----------
        torch_model : LUMETorch
            The LUMETorch model to wrap.
        initial_inputs : dict[str, Any], optional
            Initial input values. If None, uses default values from variables.
        """
        self.torch_model = torch_model
        self._initial_inputs = initial_inputs

        # Initialize state
        self._state = {}

        # Set initial values
        if initial_inputs is None:
            initial_inputs = self._get_default_inputs()

        if initial_inputs:
            self.set(initial_inputs)

        # Store initialstate for reset
        self._initial_state = self._state.copy()

    def _get_default_inputs(self) -> dict[str, Any]:
        """Get default inputs from torch model's input variables"""
        default_inputs = {}
        for var in self.torch_model.input_variables:
            if hasattr(var, "default") and var.default is not None:
                default_inputs[var.name] = var.default
            elif hasattr(var, "default_value") and var.default_value is not None:
                default_inputs[var.name] = var.default_value
        return default_inputs

    def _get(self, names: list[str]) -> dict[str, Any]:
        """
        Internal method to retrieve current values for specified variables.

        Parameters
        ----------
        names : list[str]
            List of variable names to retrieve.

        Returns
        -------
        dict[str, Any]
            Dictionary mapping variable names to their current values.
        """
        return {name: self._state[name] for name in names}

    def _set(self, values: dict[str, Any]) -> None:
        """
        Internal method to set input variables and run the torch model.

        Parameters
        ----------
        values : dict[str, Any]
            Dictionary of variable names and values to set.
        """
        # Update input values in state
        self._state.update(values)

        # Prepare inputs for torch model evaluation
        input_dict = {name: self._state[name] for name in self.torch_model.input_names}

        # Evaluate the torch model
        output_dict = self.torch_model.evaluate(input_dict)

        # Update state with outputs
        self._state.update(output_dict)

    def reset(self) -> None:
        """Reset the model to its initial state."""
        self._state = self._initial_state.copy()

    @property
    def supported_variables(self) -> dict[str, Variable]:
        """Dictionary of all supported variables."""
        variables = {}

        # Add input variables
        for var in self.torch_model.input_variables:
            variables[var.name] = var

        # Add output variables (marked as read-only)
        for var in self.torch_model.output_variables:
            var.read_only = True
            variables[var.name] = var

        return variables

    def dump(
        self,
        file: Union[str, os.PathLike],
        base_key: str = "",
        save_models: bool = True,
        save_jit: bool = False,
    ):
        """Saves the LUMETorchModel wrapper configuration to a YAML file.

        This method serializes both the underlying torch_model and the wrapper's
        initial_inputs state. The torch model is saved using its own dump method,
        and the wrapper configuration references the torch model file.

        Parameters
        ----------
        file : str or os.PathLike
            File path to which the YAML formatted string and corresponding files are saved.
        base_key : str, optional
            Base key for serialization.
        save_models : bool, optional
            Determines whether models are saved to file.
        save_jit : bool, optional
            Determines whether the model is saved as TorchScript.

        """
        logger.info(f"Dumping LUMETorchModel wrapper configuration to: {file}")

        # Get the file prefix for the wrapper
        file_prefix = os.path.splitext(os.path.abspath(file))[0]
        file_dir = os.path.dirname(os.path.abspath(file))
        filename_prefix = os.path.basename(file_prefix)

        # Create a filename for the underlying torch model
        torch_model_filename = f"{filename_prefix}_torch_model.yaml"
        torch_model_filepath = os.path.join(file_dir, torch_model_filename)

        # Dump the underlying torch model
        logger.debug(f"Dumping underlying torch model to: {torch_model_filepath}")
        self.torch_model.dump(
            torch_model_filepath,
            base_key=base_key,
            save_models=save_models,
            save_jit=save_jit,
        )

        # Get the fully qualified model class name
        torch_model_class = self.torch_model.__class__
        torch_model_class_path = (
            f"{torch_model_class.__module__}.{torch_model_class.__name__}"
        )

        # Create wrapper configuration
        wrapper_config = {
            "model_class": "LUMETorchModel",
            "torch_model_file": torch_model_filename,  # Relative path
            "torch_model_class": torch_model_class_path,  # Store for easier loading
            "initial_inputs": self._initial_inputs,
        }

        # Write wrapper configuration to file
        with open(file, "w") as f:
            yaml.dump(wrapper_config, f, default_flow_style=None, sort_keys=False)

        logger.info(f"Successfully dumped LUMETorchModel wrapper to: {file}")

    @classmethod
    def from_file(cls, filename: str):
        """Loads a LUMETorchModel from a YAML file.

        Parameters
        ----------
        filename : str
            Path to the YAML file containing the wrapper configuration.

        Returns
        -------
        LUMETorchModel
            Instance of the wrapper loaded from the file.

        Raises
        ------
        OSError
            If the file does not exist.

        """
        if not os.path.exists(filename):
            raise OSError(f"File {filename} is not found.")

        logger.info(f"Loading LUMETorchModel from file: {filename}")
        with open(filename, "r") as file:
            return cls.from_yaml(file, filename)

    @classmethod
    def from_yaml(
        cls, yaml_obj: Union[str, TextIOWrapper], config_file: Optional[str] = None
    ):
        """Loads a LUMETorchModel from a YAML string or file object.

        Parameters
        ----------
        yaml_obj : str or TextIOWrapper
            YAML formatted string or file object containing the wrapper configuration.
        config_file : str, optional
            Path to the configuration file (used to resolve relative paths).

        Returns
        -------
        LUMETorchModel
            Instance of the wrapper loaded from the YAML configuration.

        Raises
        ------
        ValueError
            If the configuration is invalid or torch_model_file is missing.

        """
        # Load the YAML configuration
        if isinstance(yaml_obj, TextIOWrapper):
            logger.debug(f"Reading configuration from file wrapper: {yaml_obj.name}")
            config = yaml.safe_load(yaml_obj.read())
            config_file = os.path.abspath(yaml_obj.name)
        elif isinstance(yaml_obj, str):
            if os.path.exists(yaml_obj):
                logger.debug(f"Loading configuration from file: {yaml_obj}")
                with open(yaml_obj, "r") as f:
                    config = yaml.safe_load(f.read())
                config_file = os.path.abspath(yaml_obj)
            else:
                logger.debug("Parsing configuration from YAML string")
                config = yaml.safe_load(yaml_obj)
        else:
            raise ValueError("yaml_obj must be a string or file object")

        # Validate configuration
        if "torch_model_file" not in config:
            raise ValueError(
                "Configuration must include 'torch_model_file' specifying the path "
                "to the underlying torch model YAML file."
            )

        # Resolve the torch model file path
        torch_model_file = config["torch_model_file"]
        if config_file is not None and not os.path.isabs(torch_model_file):
            # Resolve relative path
            config_dir = os.path.dirname(config_file)
            torch_model_file = os.path.join(config_dir, torch_model_file)

        logger.debug(f"Loading underlying torch model from: {torch_model_file}")

        # Try to get the model class path from wrapper config first
        model_class_name = config.get("torch_model_class")

        # If not in wrapper config, read from torch model file
        if model_class_name is None:
            with open(torch_model_file, "r") as f:
                torch_config = yaml.safe_load(f.read())

            if "model_class" not in torch_config:
                raise ValueError(
                    f"Torch model configuration in {torch_model_file} must include 'model_class'"
                )
            model_class_name = torch_config["model_class"]

        # Try to import the model class
        torch_model_class = None

        # First check if it's in lume_torch.models
        try:
            from lume_torch.models import get_model

            torch_model_class = get_model(model_class_name)
            logger.debug(
                f"Loaded model class from lume_torch.models: {model_class_name}"
            )
        except (KeyError, ImportError) as e:
            logger.debug(
                f"Could not load model class {model_class_name} from lume_torch.models: {e}"
            )

        # If not found, try to import from the module path if it's a fully qualified name
        if torch_model_class is None and "." in model_class_name:
            try:
                module_path, class_name = model_class_name.rsplit(".", 1)
                module = importlib.import_module(module_path)
                torch_model_class = getattr(module, class_name)
                logger.debug(f"Loaded model class from module path: {model_class_name}")
            except (ImportError, AttributeError) as e:
                logger.debug(
                    f"Could not import from module path {model_class_name}: {e}"
                )

        # If still not found, raise an error
        if torch_model_class is None:
            raise ValueError(
                f"Could not load model class: {model_class_name}. "
                "The class must be either registered in lume_torch.models or "
                "accessible via a fully qualified module path."
            )

        # Load the torch model
        torch_model = torch_model_class.from_file(torch_model_file)

        # Get initial inputs from config
        initial_inputs = config.get("initial_inputs", None)

        logger.info("Successfully loaded LUMETorchModel wrapper")

        # Create and return the wrapper instance
        return cls(torch_model=torch_model, initial_inputs=initial_inputs)
