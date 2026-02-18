import os
import logging
import warnings
from typing import Union
from pathlib import Path

from pydantic import field_validator

import torch
from torch.distributions import Normal
from torch.distributions.distribution import Distribution as TDistribution

from lume_torch.models.prob_model_base import ProbabilisticBaseModel
from lume_torch.models.torch_model import TorchModel

logger = logging.getLogger(__name__)


class NNEnsemble(ProbabilisticBaseModel):
    """LUME-model class for neural network ensembles.

    This class allows for the evaluation of multiple neural network models as an
    ensemble. Each model is a :class:`TorchModel`, and their predictions are
    combined into a Gaussian ensemble distribution for each output.

    Parameters
    ----------
    models : list of TorchModel
        List of one or more LUME-Torch neural network models.

    Attributes
    ----------
    models : list of TorchModel
        The underlying neural network models that form the ensemble.

    """

    models: list[TorchModel]

    def __init__(self, *args, **kwargs):
        """Initialize an NNEnsemble instance.

        Parameters
        ----------
        *args
            Positional arguments forwarded to :class:`ProbabilisticBaseModel`.
        **kwargs
            Keyword arguments forwarded to :class:`ProbabilisticBaseModel`.

        Notes
        -----
        This class is marked as under development and may change in future
        versions.

        """
        super().__init__(*args, **kwargs)
        logger.warning("NNEnsemble class is still under development")
        warnings.warn("This class is still under development.")

    @field_validator("models", mode="before")
    def validate_torch_model_list(cls, v):
        """Validate and, if necessary, load the list of TorchModel instances.

        Parameters
        ----------
        v : list
            List of models or model file paths. Elements may be
            :class:`TorchModel` instances or paths to serialized TorchModel
            configurations and weights.

        Returns
        -------
        list of TorchModel
            Validated list of :class:`TorchModel` instances.

        Raises
        ------
        OSError
            If required model or configuration files are missing on disk.
        TypeError
            If any element cannot be resolved to a :class:`TorchModel`.

        """
        if all(isinstance(m, (str, os.PathLike)) for m in v):
            for i, m in enumerate(v):
                if m.endswith(".pt"):
                    fname = m.split("_model.pt")[0]
                elif m.endswith(".jit"):
                    fname = m.split("_model.jit")[0]
                if os.path.exists(m) and os.path.exists(f"{fname}.yml"):
                    # if it's a wrapper around TorchModel, might need a different class or a different way to load
                    logger.debug(f"Loading TorchModel from: {fname}.yml")
                    v[i] = TorchModel(Path(f"{fname}.yml"))
                else:
                    logger.error(
                        f"Missing required files for model loading: {m} or {fname}.yml"
                    )
                    raise OSError(
                        f"Both files, {m} and {fname}.yml, are required to load the models."
                    )

        if not all(isinstance(m, TorchModel) for m in v):
            logger.error("Not all models are TorchModel instances")
            raise TypeError("All models must be of type TorchModel.")

        logger.info(f"Validated {len(v)} models for ensemble")
        return v

    def _get_predictions(
        self, input_dict: dict[str, float | torch.Tensor]
    ) -> dict[str, TDistribution]:
        """Get the predictions of the ensemble of models.

        This implements the abstract method from :class:`ProbabilisticBaseModel` by
        evaluating each model in the ensemble and aggregating their outputs.

        Parameters
        ----------
        input_dict : dict of str to float or torch.Tensor
            Dictionary of input variable names to values.

        Returns
        -------
        dict of str to TDistribution
            Dictionary of output variable names to (Gaussian) distributions
            representing the ensemble prediction.

        """
        predictions = []
        for model in self.models:
            predictions.append(model.evaluate(input_dict))
        # Return a dictionary of output variable names to distributions
        return self._create_output_dict(predictions)

    def _create_output_dict(self, output_list: list) -> dict[str, TDistribution]:
        """Create the output dictionary from the ensemble output.

        Parameters
        ----------
        output_list : list of dict
            List of output dictionaries produced by each model in the
            ensemble. Each dictionary maps output names to tensors.

        Returns
        -------
        dict of str to TDistribution
            Dictionary of output variable names to Normal distributions whose
            parameters are computed across the ensemble members.

        """
        # Ensemble output is a list of dicts of output names to values
        # need to map them to a dict of output names to distributions
        ensemble_output_dict = {}

        for key in output_list[0]:  # for each named output
            output_tensor = torch.tensor([d[key].tolist() for d in output_list])
            ensemble_output_dict[key] = Normal(
                output_tensor.mean(axis=0),
                torch.sqrt(output_tensor.var(axis=0)),
            )
        return ensemble_output_dict

    @property
    def _tkwargs(self):
        """Return tensor keyword arguments for this ensemble.

        Returns
        -------
        dict
            Dictionary with ``"device"`` and ``"dtype"`` keys derived from the
            ensemble configuration.

        """
        return {"device": self.device, "dtype": self.dtype}

    def dump(
        self,
        file: Union[str, os.PathLike],
        base_key: str = "",
        save_models: bool = True,
        save_jit: bool = False,
    ):
        """Dump the ensemble configuration and constituent models to disk.

        Parameters
        ----------
        file : str or os.PathLike
            Path to the YAML file to save the ensemble configuration.
        base_key : str, optional
            Base key for the model in the serialized configuration.
        save_models : bool, optional
            Whether to save the underlying models' weights.
        save_jit : bool, optional
            Whether to save JIT-compiled versions of the models (if
            supported).

        """
        # Save each model in the ensemble
        mod_file = file.split(".yaml")[0].split(".yml")[0]
        for idx, model in enumerate(self.models):
            model.dump(
                f"{mod_file}_{idx}.yml",
                base_key=base_key,
                save_models=save_models,
                save_jit=save_jit,
            )

        # Save the ensemble of models
        super().dump(file, base_key, save_models, save_jit)
