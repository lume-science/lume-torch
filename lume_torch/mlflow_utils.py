import os
import warnings
import logging
from typing import Any
from contextlib import nullcontext

from torch import nn

logger = logging.getLogger(__name__)

try:
    import mlflow

    HAS_MLFLOW = True

    mlflow_logger = logging.getLogger("mlflow")
    # Set log level to error until annoying signature warnings are fixed
    mlflow_logger.setLevel(logging.ERROR)
except ImportError:
    HAS_MLFLOW = False


def register_model(
    lume_torch,
    artifact_path: str,
    registered_model_name: str | None = None,
    tags: dict[str, Any] | None = None,
    version_tags: dict[str, Any] | None = None,
    alias: str | None = None,
    run_name: str | None = None,
    log_model_dump: bool = True,
    save_jit: bool = False,
    **kwargs,
) -> Any:
    """Registers the model to MLflow if mlflow is installed.

    Each time this function is called, a new version of the model is created. The model is saved to the tracking
    server or local directory, depending on the MLFLOW_TRACKING_URI.

    If no tracking server is set up, data and artifacts are saved directly under your current directory. To set up
    a tracking server, set the environment variable MLFLOW_TRACKING_URI, e.g. a local port/path. See
    https://mlflow.org/docs/latest/getting-started/intro-quickstart/ for more info.

    Note that at the moment, this does not log artifacts for custom models other than the YAML dump file.

    Parameters
    ----------
    lume_torch : LUMETorch
        LUMETorch to register.
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
        If True, the model dump is logged to MLflow.
    save_jit : bool, optional
        If True, the model is saved as a JIT model when calling model.dump, if log_model_dump=True.
    **kwargs
        Additional arguments for mlflow.pyfunc.log_model.

    Returns
    -------
    mlflow.models.model.ModelInfo
        Model info metadata.

    Raises
    ------
    ImportError
        If MLflow is not installed.

    """
    if not HAS_MLFLOW:
        logger.error("MLflow is not installed - cannot register model")
        raise ImportError(
            "MLflow is not installed. Please install mlflow to use this feature: "
            "pip install 'lume-torch[mlflow]'"
        )
    if "MLFLOW_TRACKING_URI" not in os.environ:
        logger.warning(
            "MLFLOW_TRACKING_URI not set - artifacts will be saved to current directory"
        )
        warnings.warn(
            "MLFLOW_TRACKING_URI is not set. Data and artifacts will be saved directly under your current directory."
        )

    logger.info(f"Registering model to MLflow with artifact_path: {artifact_path}")
    if registered_model_name:
        logger.info(f"Registered model name: {registered_model_name}")

    # Log the model to MLflow
    ctx = (
        mlflow.start_run(run_name=run_name)
        if mlflow.active_run() is None
        else nullcontext()
    )
    with ctx:
        if isinstance(lume_torch, nn.Module):
            logger.debug("Logging PyTorch nn.Module model to MLflow")
            model_info = mlflow.pytorch.log_model(
                pytorch_model=lume_torch,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
                **kwargs,
            )
        else:
            # Create pyfunc model for MLflow to be able to log/load the model
            logger.debug("Logging custom model as MLflow pyfunc")
            pf_model = create_mlflow_model(lume_torch)
            model_info = mlflow.pyfunc.log_model(
                python_model=pf_model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
                **kwargs,
            )

        if log_model_dump:
            logger.debug("Logging model dump files to MLflow")
            # Log the model dump files to MLflow
            # TODO: pass directory where user wants local dump to, default to working directory
            run_name = mlflow.active_run().info.run_name
            name = registered_model_name or f"{run_name}"

            lume_torch.dump(f"{name}.yml", save_jit=save_jit)
            mlflow.log_artifact(f"{name}.yml", artifact_path)
            os.remove(f"{name}.yml")

            from lume_torch.models import registered_models

            if type(lume_torch) in registered_models:
                # all registered models are torch models at the moment
                # may change in the future
                mlflow.log_artifact(f"{name}_model.pt", artifact_path)
                os.remove(f"{name}_model.pt")
                if save_jit:
                    mlflow.log_artifact(f"{name}_model.jit", artifact_path)
                    os.remove(f"{name}_model.jit")

                # Get and log the input and output transformers
                lume_torch = (
                    lume_torch._model
                    if isinstance(lume_torch, nn.Module)
                    else lume_torch
                )
                for i in range(len(lume_torch.input_transformers)):
                    mlflow.log_artifact(
                        f"{name}_input_transformers_{i}.pt", artifact_path
                    )
                    os.remove(f"{name}_input_transformers_{i}.pt")
                for i in range(len(lume_torch.output_transformers)):
                    mlflow.log_artifact(
                        f"{name}_output_transformers_{i}.pt", artifact_path
                    )
                    os.remove(f"{name}_output_transformers_{i}.pt")

    if (tags or alias or version_tags) and registered_model_name:
        logger.debug("Setting MLflow model tags and aliases")
        from mlflow import MlflowClient

        client = MlflowClient()
        # Get the latest version of the registered model that we just registered
        latest_version = model_info.registered_model_version

        if tags:
            logger.debug(f"Setting {len(tags)} registered model tags")
            for key, value in tags.items():
                client.set_registered_model_tag(registered_model_name, key, value)
        if version_tags:
            logger.debug(f"Setting {len(version_tags)} version tags")
            for key, value in version_tags.items():
                client.set_model_version_tag(
                    registered_model_name, latest_version, key, value
                )
        if alias:
            logger.debug(f"Setting alias: {alias}")
            client.set_registered_model_alias(
                registered_model_name, alias, latest_version
            )

    elif (tags or alias or version_tags) and not registered_model_name:
        logger.warning(
            "No registered model name provided - tags and aliases will not be set"
        )
        warnings.warn(
            "No registered model name provided. Tags and aliases will not be set."
        )

    logger.info("Model successfully registered to MLflow")
    return model_info


def create_mlflow_model(model) -> Any:
    """Creates an MLflow model from the given model."""
    if not HAS_MLFLOW:
        raise ImportError(
            "MLflow is not installed. Please install mlflow to use this feature: "
            "pip install 'lume-torch[mlflow]'"
        )
    return PyFuncModel(model=model)


if HAS_MLFLOW:

    class PyFuncModel(mlflow.pyfunc.PythonModel):
        """Custom MLflow model class for LUMETorch.

        Uses Pyfunc to define a model that can be saved and loaded with MLflow.
        Must implement the `predict` method.

        """

        # Disable type hint validation for the predict method to avoid annoying warnings
        # since we have type validation in the lume-torch itself.
        _skip_type_hint_validation = True

        def __init__(self, model):
            self.model = model

        def predict(self, model_input):
            """Evaluate the model with the given input.

            Parameters
            ----------
            model_input : dict
                Input dictionary for model evaluation.

            Returns
            -------
            dict
                Model evaluation results.

            """
            return self.model.evaluate(model_input)

        def save_model(self):
            raise NotImplementedError("Save model not implemented")

        def load_model(self):
            raise NotImplementedError("Load model not implemented")

        def get_lume_torch(self):
            return self.model
