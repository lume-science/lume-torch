import os
import yaml
from typing import Union

registered_models = []
__all__ = ["get_model", "model_from_yaml", "registered_models"]

# models requiring torch
try:
    from lume_torch.models.torch_model import TorchModel
    from lume_torch.models.torch_module import TorchModule
    from lume_torch.models.ensemble import NNEnsemble
    from lume_torch.models.gp_model import GPModel

    registered_models += [TorchModel, TorchModule, NNEnsemble, GPModel]
    __all__ += ["TorchModel", "TorchModule", "NNEnsemble", "GPModel"]
except (ModuleNotFoundError, ImportError):
    pass


def get_model(name: str):
    """Return the LUME model class for the given name.

    Parameters
    ----------
    name : str
        Name of the LUME model class.

    Returns
    -------
    type
        LUME model class corresponding to ``name``.

    Raises
    ------
    KeyError
        If no registered model with the given name exists.

    """
    model_lookup = {m.__name__: m for m in registered_models}
    if name not in model_lookup.keys():
        raise KeyError(
            f"No model named {name}, available models are {list(model_lookup.keys())}"
        )
    return model_lookup[name]


def model_from_yaml(yaml_str: Union[str, os.PathLike]):
    """Create a LUME model from a YAML string or file path.

    Parameters
    ----------
    yaml_str : str or os.PathLike
        YAML formatted string or path to a YAML file defining the model
        configuration.

    Returns
    -------
    LUMETorch
        Instantiated LUME model defined by the YAML configuration.

    """
    if os.path.exists(yaml_str):
        with open(yaml_str) as f:
            config = yaml.safe_load(f.read())
    else:
        config = yaml.safe_load(yaml_str)
    model_class = get_model(config["model_class"])
    return model_class(yaml_str)
