import os
import sys
import yaml
import logging
import importlib
from typing import Union, get_origin, get_args

from lume_torch.variables import ScalarVariable, get_variable

logger = logging.getLogger(__name__)


def try_import_module(name: str):
    """Tries to import module if required.

    Parameters
    ----------
    name : str
        Module name.

    Returns
    -------
    module or None
        Imported module if successful, None otherwise.

    """
    if name not in sys.modules:
        try:
            module = importlib.import_module(name)
            logger.debug(f"Successfully imported module: {name}")
        except ImportError as e:
            logger.debug(f"Failed to import module {name}: {e}")
            module = None
    else:
        module = sys.modules[name]
        logger.debug(f"Module {name} already in sys.modules")
    return module


def verify_unique_variable_names(variables: list[ScalarVariable]):
    """Verifies that variable names are unique.

    Raises a ValueError if any reoccurring variable names are found.

    Parameters
    ----------
    variables : list of ScalarVariable
        List of scalar variables.

    Raises
    ------
    ValueError
        If any variable names are not unique.

    """
    names = [var.name for var in variables]
    non_unique_names = [name for name in set(names) if names.count(name) > 1]
    if non_unique_names:
        logger.error(f"Variable names {non_unique_names} are not unique")
        raise ValueError(f"Variable names {non_unique_names} are not unique.")


def serialize_variables(v: dict):
    """Performs custom serialization for in- and output variables.

    Parameters
    ----------
    v : dict
        Object to serialize.

    Returns
    -------
    dict
        Dictionary with serialized in- and output variables.

    """
    logger.debug("Serializing variables")
    for key, value in v.items():
        if key in ["input_variables", "output_variables"] and isinstance(value, list):
            # Call model_dump with mode='python' to ensure proper serialization
            serialized_vars = [
                var.model_dump(mode="python") if hasattr(var, "model_dump") else var
                for var in value
            ]
            v[key] = {
                var_dict["name"]: {
                    var_k: var_v
                    for var_k, var_v in var_dict.items()
                    if not (var_k == "name" or var_v is None)
                }
                for var_dict in serialized_vars
            }
    return v


def deserialize_variables(v):
    """Performs custom deserialization for in- and output variables.

    Parameters
    ----------
    v : dict
        Object to deserialize.

    Returns
    -------
    dict
        Dictionary with deserialized in- and output variables.

    """
    logger.debug("Deserializing variables")
    for key, value in v.items():
        if key in ["input_variables", "output_variables"] and isinstance(value, dict):
            v[key] = [
                var_dict | {"name": var_name} for var_name, var_dict in value.items()
            ]
    return v


def variables_as_yaml(
    input_variables: list[ScalarVariable],
    output_variables: list[ScalarVariable],
    file: Union[str, os.PathLike] = None,
) -> str:
    """Returns and optionally saves YAML formatted string defining the in- and output variables.

    Parameters
    ----------
    input_variables : list of ScalarVariable
        List of input variables.
    output_variables : list of ScalarVariable
        List of output variables.
    file : str or os.PathLike, optional
        If not None, YAML formatted string is saved to given file path.

    Returns
    -------
    str
        YAML formatted string defining the in- and output variables.

    """
    logger.debug(
        f"Creating YAML for {len(input_variables)} input and {len(output_variables)} output variables"
    )
    for variables in [input_variables, output_variables]:
        verify_unique_variable_names(variables)
    v = {
        "input_variables": [var.model_dump() for var in input_variables],
        "output_variables": [var.model_dump() for var in output_variables],
    }
    s = yaml.safe_dump(serialize_variables(v), default_flow_style=None, sort_keys=False)
    if file is not None:
        logger.info(f"Saving variables to file: {file}")
        with open(file, "w") as f:
            f.write(s)
    return s


def variables_from_dict(
    config: dict,
) -> tuple[list[ScalarVariable], list[ScalarVariable]]:
    """Parses given config and returns in- and output variable lists.

    Parameters
    ----------
    config : dict
        Variable configuration.

    Returns
    -------
    tuple of (list of ScalarVariable, list of ScalarVariable)
        In- and output variable lists.

    """
    logger.debug("Parsing variables from dictionary")
    input_variables, output_variables = [], []
    for key, value in {**config}.items():
        if key in ["input_variables", "output_variables"]:
            for var in value:
                variable_class = get_variable(var["variable_class"])
                if key == "input_variables":
                    input_variables.append(variable_class(**var))
                elif key == "output_variables":
                    output_variables.append(variable_class(**var))
    for variables in [input_variables, output_variables]:
        verify_unique_variable_names(variables)
    logger.debug(
        f"Parsed {len(input_variables)} input and {len(output_variables)} output variables"
    )
    return input_variables, output_variables


def variables_from_yaml(
    yaml_obj: Union[str, os.PathLike],
) -> tuple[list[ScalarVariable], list[ScalarVariable]]:
    """Parses YAML object and returns in- and output variable lists.

    Parameters
    ----------
    yaml_obj : str or os.PathLike
        YAML formatted string or file path.

    Returns
    -------
    tuple of (list of ScalarVariable, list of ScalarVariable)
        In- and output variable lists.

    """
    if os.path.exists(yaml_obj):
        logger.debug(f"Loading variables from file: {yaml_obj}")
        with open(yaml_obj) as f:
            yaml_str = f.read()
    else:
        logger.debug("Parsing variables from YAML string")
        yaml_str = yaml_obj
    config = deserialize_variables(yaml.safe_load(yaml_str))
    return variables_from_dict(config)


def get_valid_path(
    path: Union[str, os.PathLike],
    directory: Union[str, os.PathLike] = "",
) -> Union[str, os.PathLike]:
    """Validates path exists either as relative or absolute path and returns the first valid option.

    Parameters
    ----------
    path : str or os.PathLike
        Path to validate.
    directory : str or os.PathLike, optional
        Directory against which relative paths are checked.

    Returns
    -------
    str or os.PathLike
        The first valid path option as an absolute path.

    Raises
    ------
    OSError
        If file is not found.

    """
    relative_path = os.path.join(directory, path)
    if os.path.exists(relative_path):
        logger.debug(f"Found relative path: {relative_path}")
        return os.path.abspath(relative_path)
    elif os.path.exists(path):
        logger.debug(f"Found absolute path: {path}")
        return os.path.abspath(path)
    else:
        logger.error(
            f"File {path} not found in directory {directory} or as absolute path"
        )
        raise OSError(f"File {path} is not found.")


def replace_relative_paths(
    d: dict,
    model_fields: dict = None,
    directory: Union[str, os.PathLike] = "",
) -> dict:
    """Replaces dictionary entries with absolute paths where the model field annotation is not string or path-like.

    Parameters
    ----------
    d : dict
        Dictionary to process.
    model_fields : dict, optional
        Model fields dictionary used to check expected type.
    directory : str or os.PathLike, optional
        Directory against which relative paths are checked.

    Returns
    -------
    dict
        Dictionary with replaced paths.

    """
    logger.debug(f"Replacing relative paths with directory base: {directory}")
    if model_fields is None:
        model_fields = {}
    for k, v in d.items():
        if isinstance(v, (str, os.PathLike)):
            if k in model_fields.keys():
                field_types = [model_fields[k].annotation]
                if get_origin(model_fields[k].annotation) is Union:
                    field_types = list(get_args(model_fields[k].annotation))
                if all([t not in field_types for t in [str, os.PathLike]]):
                    d[k] = get_valid_path(v, directory)
        elif isinstance(v, list):
            if k in model_fields.keys():
                field_types = []
                for i, field_type in enumerate(get_args(model_fields[k].annotation)):
                    if get_origin(field_type) is Union:
                        field_types.extend(list(get_args(field_type)))
                    else:
                        field_types.append(field_type)
                for i, ele in enumerate(v):
                    if isinstance(ele, (str, os.PathLike)) and all(
                        [t not in field_types for t in [str, os.PathLike]]
                    ):
                        v[i] = get_valid_path(ele, directory)
        elif isinstance(v, dict):
            model_subfields = {
                ".".join(key.split(".")[1:]): value
                for key, value in model_fields.items()
                if key.startswith(f"{k}.")
            }
            d[k] = replace_relative_paths(v, model_subfields, directory)
    return d
