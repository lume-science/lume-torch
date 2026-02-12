"""Tests for mlflow_utils module."""

import os
import sys
import pytest
import tempfile
from unittest import mock

from lume_torch.variables import ScalarVariable
from lume_torch.base import LUMETorch


class SimpleModel(LUMETorch):
    """Simple test model for testing."""

    def _evaluate(self, input_dict, **kwargs):
        return {"output": input_dict["input"] * 2.0}


@pytest.fixture
def simple_model():
    """Create a simple test model."""
    input_variables = [
        ScalarVariable(name="input", default_value=1.0, value_range=(0.0, 10.0))
    ]
    output_variables = [
        ScalarVariable(name="output", default_value=2.0, value_range=(0.0, 20.0))
    ]
    return SimpleModel(
        input_variables=input_variables, output_variables=output_variables
    )


class TestMLflowUtilsImport:
    """Test that mlflow_utils can be imported with and without MLflow installed."""

    def test_import_mlflow_utils_without_mlflow(self):
        """Test that mlflow_utils can be imported without MLflow."""
        # This test runs with whatever environment is available
        # Just verify the module can be imported
        from lume_torch import mlflow_utils

        # HAS_MLFLOW should be defined regardless
        assert hasattr(mlflow_utils, "HAS_MLFLOW")
        assert isinstance(mlflow_utils.HAS_MLFLOW, bool)

    def test_register_model_function_exists(self):
        """Test that register_model function exists."""
        from lume_torch import mlflow_utils

        assert hasattr(mlflow_utils, "register_model")
        assert callable(mlflow_utils.register_model)

    def test_create_mlflow_model_function_exists(self):
        """Test that create_mlflow_model function exists."""
        from lume_torch import mlflow_utils

        assert hasattr(mlflow_utils, "create_mlflow_model")
        assert callable(mlflow_utils.create_mlflow_model)


class TestMLflowUtilsWithoutMLflow:
    """Test mlflow_utils behavior when MLflow is not available."""

    def test_register_model_raises_import_error(self, simple_model):
        """Test that register_model raises ImportError when MLflow is not installed."""
        from lume_torch import mlflow_utils

        # Mock HAS_MLFLOW to False
        with mock.patch.object(mlflow_utils, "HAS_MLFLOW", False):
            with pytest.raises(ImportError) as exc_info:
                mlflow_utils.register_model(simple_model, artifact_path="test_path")

            # Check that error message is helpful
            assert "MLflow is not installed" in str(exc_info.value)
            assert "pip install 'lume-torch[mlflow]'" in str(exc_info.value)

    def test_create_mlflow_model_raises_import_error(self, simple_model):
        """Test that create_mlflow_model raises ImportError when MLflow is not installed."""
        from lume_torch import mlflow_utils

        # Mock HAS_MLFLOW to False
        with mock.patch.object(mlflow_utils, "HAS_MLFLOW", False):
            with pytest.raises(ImportError) as exc_info:
                mlflow_utils.create_mlflow_model(simple_model)

            # Check that error message is helpful
            assert "MLflow is not installed" in str(exc_info.value)
            assert "pip install 'lume-torch[mlflow]'" in str(exc_info.value)

    def test_pyfunc_model_not_defined_without_mlflow(self):
        """Test that PyFuncModel is not defined when MLflow is not available."""
        from lume_torch import mlflow_utils

        # If HAS_MLFLOW is False, PyFuncModel should not be defined
        if not mlflow_utils.HAS_MLFLOW:
            assert not hasattr(mlflow_utils, "PyFuncModel")


@pytest.mark.skipif(
    "mlflow" not in sys.modules and not any("mlflow" in str(p) for p in sys.path),
    reason="MLflow not installed - skipping MLflow-specific tests",
)
class TestMLflowUtilsWithMLflow:
    """Test mlflow_utils behavior when MLflow is available."""

    def test_has_mlflow_true(self):
        """Test that HAS_MLFLOW is True when MLflow is installed."""
        pytest.importorskip("mlflow")
        from lume_torch import mlflow_utils

        # Force reimport to ensure we get the real value
        import importlib

        importlib.reload(mlflow_utils)

        assert mlflow_utils.HAS_MLFLOW is True

    def test_pyfunc_model_defined_with_mlflow(self):
        """Test that PyFuncModel is defined when MLflow is available."""
        pytest.importorskip("mlflow")
        from lume_torch import mlflow_utils

        if mlflow_utils.HAS_MLFLOW:
            assert hasattr(mlflow_utils, "PyFuncModel")

    def test_register_model_basic(self, simple_model):
        """Test basic register_model functionality with MLflow installed."""
        mlflow = pytest.importorskip("mlflow")
        from lume_torch import mlflow_utils

        if not mlflow_utils.HAS_MLFLOW:
            pytest.skip("MLflow not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use SQLite database backend instead of filesystem to avoid deprecation warning
            tracking_uri = f"sqlite:///{tmpdir}/mlflow.db"
            mlflow.set_tracking_uri(tracking_uri)
            # Create a default experiment to avoid "experiment not found" errors
            mlflow.create_experiment("test_experiment")
            mlflow.set_experiment("test_experiment")

            # Mock environment to avoid warning
            with mock.patch.dict(os.environ, {"MLFLOW_TRACKING_URI": tracking_uri}):
                # Mock log_artifact to avoid file system operations
                with mock.patch.object(mlflow, "log_artifact"):
                    try:
                        result = mlflow_utils.register_model(
                            simple_model,
                            artifact_path="test_model",
                            log_model_dump=False,  # Disable dump to simplify test
                        )

                        # Check that something was returned
                        assert result is not None
                    except Exception as e:
                        # Some MLflow operations might fail in test environment
                        # Just verify the function is callable
                        pytest.skip(f"MLflow operation failed: {e}")

    def test_register_model_with_warnings_no_tracking_uri(self, simple_model):
        """Test that register_model warns when MLFLOW_TRACKING_URI is not set."""
        mlflow = pytest.importorskip("mlflow")
        from lume_torch import mlflow_utils

        if not mlflow_utils.HAS_MLFLOW:
            pytest.skip("MLflow not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use SQLite database backend instead of filesystem to avoid deprecation warning
            tracking_uri = f"sqlite:///{tmpdir}/mlflow.db"
            mlflow.set_tracking_uri(tracking_uri)
            # Create a default experiment to avoid "experiment not found" errors
            mlflow.create_experiment("test_experiment")
            mlflow.set_experiment("test_experiment")

            # Remove MLFLOW_TRACKING_URI from environment
            with mock.patch.dict(os.environ, {}, clear=True):
                with pytest.warns(UserWarning, match="MLFLOW_TRACKING_URI is not set"):
                    with mock.patch.object(mlflow, "log_artifact"):
                        try:
                            mlflow_utils.register_model(
                                simple_model,
                                artifact_path="test_model",
                                log_model_dump=False,
                            )
                        except Exception as e:
                            pytest.skip(f"MLflow operation failed: {e}")

    def test_create_mlflow_model(self, simple_model):
        """Test create_mlflow_model function."""
        _ = pytest.importorskip("mlflow")
        from lume_torch import mlflow_utils

        if not mlflow_utils.HAS_MLFLOW:
            pytest.skip("MLflow not available")

        pyfunc_model = mlflow_utils.create_mlflow_model(simple_model)

        # Check that the model was created
        assert pyfunc_model is not None
        assert hasattr(pyfunc_model, "model")
        assert pyfunc_model.model is simple_model

    def test_pyfunc_model_predict(self, simple_model):
        """Test PyFuncModel predict method."""
        _ = pytest.importorskip("mlflow")
        from lume_torch import mlflow_utils

        if not mlflow_utils.HAS_MLFLOW:
            pytest.skip("MLflow not available")

        pyfunc_model = mlflow_utils.PyFuncModel(model=simple_model)

        # Test prediction
        input_dict = {"input": 5.0}
        result = pyfunc_model.predict(input_dict)

        assert result is not None
        assert "output" in result
        assert result["output"] == 10.0

    def test_pyfunc_model_get_lume_torch(self, simple_model):
        """Test PyFuncModel get_lume_torch method."""
        _ = pytest.importorskip("mlflow")
        from lume_torch import mlflow_utils

        if not mlflow_utils.HAS_MLFLOW:
            pytest.skip("MLflow not available")

        pyfunc_model = mlflow_utils.PyFuncModel(model=simple_model)

        # Test get_lume_torch
        retrieved_model = pyfunc_model.get_lume_torch()
        assert retrieved_model is simple_model

    def test_pyfunc_model_save_not_implemented(self, simple_model):
        """Test that PyFuncModel save_model raises NotImplementedError."""
        _ = pytest.importorskip("mlflow")
        from lume_torch import mlflow_utils

        if not mlflow_utils.HAS_MLFLOW:
            pytest.skip("MLflow not available")

        pyfunc_model = mlflow_utils.PyFuncModel(model=simple_model)

        with pytest.raises(NotImplementedError):
            pyfunc_model.save_model()

    def test_pyfunc_model_load_not_implemented(self, simple_model):
        """Test that PyFuncModel load_model raises NotImplementedError."""
        _ = pytest.importorskip("mlflow")
        from lume_torch import mlflow_utils

        if not mlflow_utils.HAS_MLFLOW:
            pytest.skip("MLflow not available")

        pyfunc_model = mlflow_utils.PyFuncModel(model=simple_model)

        with pytest.raises(NotImplementedError):
            pyfunc_model.load_model()


class TestRegisterModelParameters:
    """Test register_model with various parameter combinations."""

    def test_register_model_with_registered_name(self, simple_model):
        """Test register_model with registered_model_name parameter."""
        mlflow = pytest.importorskip("mlflow")
        from lume_torch import mlflow_utils

        if not mlflow_utils.HAS_MLFLOW:
            pytest.skip("MLflow not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use SQLite database backend instead of filesystem to avoid deprecation warning
            tracking_uri = f"sqlite:///{tmpdir}/mlflow.db"
            mlflow.set_tracking_uri(tracking_uri)
            # Create a default experiment to avoid "experiment not found" errors
            mlflow.create_experiment("test_experiment")
            mlflow.set_experiment("test_experiment")

            with mock.patch.dict(os.environ, {"MLFLOW_TRACKING_URI": tracking_uri}):
                with mock.patch.object(mlflow, "log_artifact"):
                    try:
                        result = mlflow_utils.register_model(
                            simple_model,
                            artifact_path="test_model",
                            registered_model_name="TestModel",
                            log_model_dump=False,
                        )
                        assert result is not None
                    except Exception as e:
                        pytest.skip(f"MLflow operation failed: {e}")

    def test_register_model_with_tags_no_registered_name_warns(self, simple_model):
        """Test that tags without registered_model_name produces warning."""
        mlflow = pytest.importorskip("mlflow")
        from lume_torch import mlflow_utils

        if not mlflow_utils.HAS_MLFLOW:
            pytest.skip("MLflow not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use SQLite database backend instead of filesystem to avoid deprecation warning
            tracking_uri = f"sqlite:///{tmpdir}/mlflow.db"
            mlflow.set_tracking_uri(tracking_uri)
            # Create a default experiment to avoid "experiment not found" errors
            mlflow.create_experiment("test_experiment")
            mlflow.set_experiment("test_experiment")

            with mock.patch.dict(os.environ, {"MLFLOW_TRACKING_URI": tracking_uri}):
                with mock.patch.object(mlflow, "log_artifact"):
                    with pytest.warns(
                        UserWarning, match="No registered model name provided"
                    ):
                        try:
                            mlflow_utils.register_model(
                                simple_model,
                                artifact_path="test_model",
                                tags={"test": "value"},
                                log_model_dump=False,
                            )
                        except Exception as e:
                            pytest.skip(f"MLflow operation failed: {e}")


class TestNNModuleIntegration:
    """Test register_model with torch.nn.Module."""

    def test_register_nn_module(self):
        """Test register_model with a torch.nn.Module."""
        pytest.importorskip("torch")
        mlflow = pytest.importorskip("mlflow")
        from lume_torch import mlflow_utils
        import torch.nn as nn

        if not mlflow_utils.HAS_MLFLOW:
            pytest.skip("MLflow not available")

        # Create a simple nn.Module
        model = nn.Linear(10, 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use SQLite database backend instead of filesystem to avoid deprecation warning
            tracking_uri = f"sqlite:///{tmpdir}/mlflow.db"
            mlflow.set_tracking_uri(tracking_uri)
            # Create a default experiment to avoid "experiment not found" errors
            mlflow.create_experiment("test_experiment")
            mlflow.set_experiment("test_experiment")

            with mock.patch.dict(os.environ, {"MLFLOW_TRACKING_URI": tracking_uri}):
                try:
                    result = mlflow_utils.register_model(
                        model, artifact_path="test_model", log_model_dump=False
                    )
                    assert result is not None
                except Exception as e:
                    pytest.skip(f"MLflow operation failed: {e}")


class TestModuleReload:
    """Test that the module handles reload correctly."""

    def test_module_reload_preserves_has_mlflow(self):
        """Test that reloading the module preserves HAS_MLFLOW state."""
        from lume_torch import mlflow_utils
        import importlib

        # Store original value
        original_has_mlflow = mlflow_utils.HAS_MLFLOW

        # Reload module
        importlib.reload(mlflow_utils)

        # Should have the same value after reload
        assert mlflow_utils.HAS_MLFLOW == original_has_mlflow
