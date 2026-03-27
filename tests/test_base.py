import os
import pytest
import yaml

from lume_torch.base import LUMETorch, LUMETorchModel
from lume_torch.variables import TorchScalarVariable


class ExampleModel(LUMETorch):
    def _evaluate(self, input_dict):
        pass


class SimpleModel(LUMETorch):
    """Simple model for testing LUMETorchModel wrapper."""

    def _evaluate(self, input_dict):
        # Simple computation: output = input * 2
        return {
            "output1": input_dict["input1"] * 2.0,
            "output2": input_dict["input2"] * 2.0,
        }


class TestBaseModel:
    def test_init(self, simple_variables):
        # init with no variable specification
        with pytest.raises(TypeError):
            _ = LUMETorch()

        # init child class with no _evaluate function
        class NoEvaluateModel(LUMETorch):
            def predict(self, input_dict):
                pass

        with pytest.raises(TypeError):
            _ = NoEvaluateModel(**simple_variables)

        # TODO: move to test_torch_model.py
        # # init child class with input variables missing default value
        # simple_variables_no_default = copy.deepcopy(simple_variables)
        # simple_variables_no_default["input_variables"][0].default_value = None
        # with pytest.raises(ValueError):
        #     _ = ExampleModel(**simple_variables_no_default)

        # init child class with evaluate function
        example_model = ExampleModel(**simple_variables)
        assert example_model.input_variables == simple_variables["input_variables"]
        assert example_model.output_variables == simple_variables["output_variables"]

        # input and output variables sharing names is fine
        input_variables = simple_variables["input_variables"]
        output_variables = simple_variables["output_variables"]
        original_name = input_variables[0].name
        input_variables[0].name = output_variables[0].name
        _ = ExampleModel(**simple_variables)
        input_variables[0].name = original_name

    def test_dict(self, simple_variables):
        example_model = ExampleModel(**simple_variables)
        dict_output = example_model.model_dump()
        assert isinstance(dict_output["input_variables"], list)
        assert isinstance(dict_output["output_variables"], list)
        assert len(dict_output["input_variables"]) == 2

    def test_json(self, simple_variables):
        example_model = ExampleModel(**simple_variables)
        _ = example_model.json()

    def test_yaml_serialization(self, simple_variables):
        example_model = ExampleModel(**simple_variables)
        yaml_output = example_model.yaml()
        dict_output = yaml.safe_load(yaml_output)
        dict_output["input_variables"]["input1"]["variable_class"] = (
            TorchScalarVariable.__name__
        )

        # test loading from yaml
        loaded_model = ExampleModel(**dict_output)
        assert loaded_model == example_model

    def test_file_serialization(self, simple_variables):
        example_model = ExampleModel(**simple_variables)
        file = "test_model.yml"
        example_model.dump(file)

        os.remove(file)

    def test_deserialization_from_config(self, simple_variables):
        example_model = ExampleModel(**simple_variables)
        file = "test_model.yml"
        _ = example_model.dump(file)
        loaded_model = ExampleModel(file)
        os.remove(file)
        assert loaded_model.input_variables == example_model.input_variables
        assert loaded_model.output_variables == example_model.output_variables

    def test_input_names(self, simple_variables):
        example_model = ExampleModel(**simple_variables)
        for i, var in enumerate(simple_variables["input_variables"]):
            assert example_model.input_names.index(var.name) == i

    def test_output_names(self, simple_variables):
        example_model = ExampleModel(**simple_variables)
        for i, var in enumerate(simple_variables["output_variables"]):
            assert example_model.output_names.index(var.name) == i

    def test_input_validation(self, simple_variables, monkeypatch):
        example_model = ExampleModel(**simple_variables)
        input_variables = simple_variables["input_variables"]
        input_dict = {input_variables[0].name: 2.0, input_variables[1].name: 1.5}
        example_model.input_validation(input_dict)
        with pytest.raises(TypeError):
            input_dict[input_variables[0].name] = True
            example_model.input_validation(input_dict)

        # setting strictness flag
        assert input_variables[0].default_validation_config == "none"
        with pytest.raises(ValueError):
            # has to be a ConfigEnum type
            example_model.input_validation_config = {input_variables[0].name: "test"}

        # range check with strictness flag
        example_model.input_validation_config = {input_variables[0].name: "error"}
        with pytest.raises(ValueError):
            input_dict[input_variables[0].name] = 6.0
            example_model.input_validation(input_dict)

        # a warning is printed
        example_model.input_validation_config = {input_variables[0].name: "warn"}
        input_dict[input_variables[0].name] = 6.0
        with pytest.warns(UserWarning):
            example_model.input_validation(input_dict)

        # nothing is printed/raised
        example_model.input_validation_config = {input_variables[0].name: "none"}
        input_dict[input_variables[0].name] = 6.0
        example_model.input_validation(input_dict)

    def test_input_validation_partial_config(self, simple_variables):
        example_model = ExampleModel(**simple_variables)
        input_variables = simple_variables["input_variables"]

        example_model.input_validation_config = {input_variables[0].name: "error"}

        input_dict = {input_variables[0].name: 6.0, input_variables[1].name: 2.0}
        with pytest.raises(ValueError):
            example_model.input_validation(input_dict)

        input_dict = {input_variables[0].name: 1.0, input_variables[1].name: 99.0}
        example_model.input_validation(input_dict)

    def test_input_validation_missing_input_validates_default(self, simple_variables):
        """Variables absent from input_dict have their default_value validated
        instead of raising, when require_all_inputs is False (the default).
        Uses model_construct to bypass pydantic's own construction-time check
        so that we can set an out-of-range default and confirm input_validation
        catches it."""
        # Bypass pydantic on both the variable AND the model to get an invalid default in
        var_bad = TorchScalarVariable.model_construct(
            name="bad",
            default_value=99.0,
            value_range=(0.0, 5.0),
            read_only=False,
            default_validation_config="error",
        )
        var_good = TorchScalarVariable(name="good", default_value=2.0)
        model = ExampleModel.model_construct(
            input_variables=[var_bad, var_good],
            output_variables=simple_variables["output_variables"],
            input_validation_config=None,
            output_validation_config=None,
        )
        # "bad" is absent; its default (99.0) is outside range and config is "error"
        with pytest.raises(ValueError):
            model.input_validation({"good": 2.0})

        # Try with no default
        model.input_variables[0].default_value = None
        # "bad" is absent; tries to grab the default but it's None, which should raise a ValueError about missing required input
        with pytest.raises(ValueError):
            model.input_validation({"good": 2.0})

    def test_input_validation_check_no_missing_inputs(self, simple_variables):
        """require_all_inputs=True raises ValueError when a variable is absent."""
        example_model = ExampleModel(**simple_variables)
        input_variables = simple_variables["input_variables"]

        example_model.require_all_inputs = True

        with pytest.raises(ValueError, match="Missing required input variable"):
            example_model.input_validation(
                {input_variables[0].name: 1.0},
            )

        # providing all inputs does not raise
        example_model.input_validation(
            {input_variables[0].name: 1.0, input_variables[1].name: 2.0},
        )

    def test_input_validation_check_read_only(self, simple_variables):
        """Read-only enforcement is unconditional — driven by var.read_only."""
        read_only_var = TorchScalarVariable(
            name="fixed",
            default_value=1.0,
            read_only=True,
        )
        normal_var = simple_variables["input_variables"][1]
        model = ExampleModel(
            input_variables=[read_only_var, normal_var],
            output_variables=simple_variables["output_variables"],
        )
        model.input_validation_config = {"fixed": "error"}

        # value matches default — should pass
        model.input_validation({"fixed": 1.0, normal_var.name: 2.0})

        # value differs from default — always raises (no opt-out toggle)
        with pytest.raises(Exception):
            model.input_validation({"fixed": 9.0, normal_var.name: 2.0})

    def test_input_validation_config_unknown_key_raises(self, simple_variables):
        """Setting input_validation_config with a key that isn't an input variable name raises."""
        example_model = ExampleModel(**simple_variables)
        with pytest.raises(ValueError, match="unknown variable name"):
            example_model.input_validation_config = {"nonexistent_var": "error"}

    def test_output_validation_config_unknown_key_raises(self, simple_variables):
        """Setting output_validation_config with a key that isn't an output variable name raises."""
        example_model = ExampleModel(**simple_variables)
        with pytest.raises(ValueError, match="unknown variable name"):
            example_model.output_validation_config = {"nonexistent_var": "error"}

    def test_output_validation(self, simple_variables):
        example_model = ExampleModel(**simple_variables)
        output_variables = simple_variables["output_variables"]
        output_dict = {output_variables[0].name: 3.0, output_variables[1].name: 1.7}
        example_model.output_validation(output_dict)
        with pytest.raises(TypeError):
            output_dict[output_variables[0].name] = "test"
            example_model.output_validation(output_dict)


class TestLUMETorchModel:
    """Test suite for LUMETorchModel wrapper class."""

    @pytest.fixture
    def simple_torch_model(self, simple_variables):
        """Create a simple torch model for testing."""
        return SimpleModel(**simple_variables)

    def test_init_with_torch_model(self, simple_torch_model):
        """Test initialization of LUMETorchModel with a torch model."""
        wrapper = LUMETorchModel(simple_torch_model)
        assert wrapper.torch_model == simple_torch_model
        assert len(wrapper._state) > 0  # Should have initial state

    def test_init_with_initial_inputs(self, simple_torch_model):
        """Test initialization with explicit initial inputs."""
        initial_inputs = {"input1": 3.0, "input2": 2.5}
        wrapper = LUMETorchModel(simple_torch_model, initial_inputs=initial_inputs)
        assert wrapper._initial_inputs == initial_inputs
        assert wrapper._state["input1"] == 3.0
        assert wrapper._state["input2"] == 2.5

    def test_set_and_get(self, simple_torch_model):
        """Test set and get operations on the wrapper."""
        wrapper = LUMETorchModel(simple_torch_model)

        # Set input values
        wrapper.set({"input1": 4.0, "input2": 3.0})

        # Get values
        result = wrapper.get(["input1", "input2", "output1", "output2"])
        assert result["input1"] == 4.0
        assert result["input2"] == 3.0
        assert result["output1"] == 8.0  # 4.0 * 2
        assert result["output2"] == 6.0  # 3.0 * 2

    def test_supported_variables(self, simple_torch_model):
        """Test that supported_variables includes both input and output variables."""
        wrapper = LUMETorchModel(simple_torch_model)
        variables = wrapper.supported_variables

        assert "input1" in variables
        assert "input2" in variables
        assert "output1" in variables
        assert "output2" in variables

        # Output variables should be read-only
        assert variables["output1"].read_only is True
        assert variables["output2"].read_only is True

    def test_reset(self, simple_torch_model):
        """Test reset functionality."""
        initial_inputs = {"input1": 2.0, "input2": 1.5}
        wrapper = LUMETorchModel(simple_torch_model, initial_inputs=initial_inputs)

        # Change state
        wrapper.set({"input1": 5.0})
        assert wrapper._state["input1"] == 5.0

        # Reset
        wrapper.reset()
        assert wrapper._state["input1"] == 2.0

    def test_dump_and_load(self, simple_torch_model, tmp_path):
        """Test dumping and loading the wrapper."""
        # Create wrapper with initial inputs
        initial_inputs = {"input1": 3.0, "input2": 2.5}
        wrapper = LUMETorchModel(simple_torch_model, initial_inputs=initial_inputs)

        # Set some values
        wrapper.set({"input1": 4.0, "input2": 3.0})
        original_output = wrapper.get(["output1", "output2"])

        # Dump to file
        config_file = tmp_path / "test_wrapper.yaml"
        wrapper.dump(str(config_file))

        # Verify files were created
        assert config_file.exists()
        torch_model_file = tmp_path / "test_wrapper_torch_model.yaml"
        assert torch_model_file.exists()

        # Load from file
        loaded_wrapper = LUMETorchModel.from_file(str(config_file))

        # Verify initial inputs were restored
        assert loaded_wrapper._initial_inputs == initial_inputs

        # Verify we can run set and get on loaded model
        loaded_wrapper.set({"input1": 4.0, "input2": 3.0})
        loaded_output = loaded_wrapper.get(["output1", "output2"])

        # Outputs should match
        assert loaded_output["output1"] == original_output["output1"]
        assert loaded_output["output2"] == original_output["output2"]

    def test_dump_with_no_initial_inputs(self, simple_torch_model, tmp_path):
        """Test dumping and loading when no initial inputs were provided."""
        # Create wrapper without initial inputs (uses defaults)
        wrapper = LUMETorchModel(simple_torch_model)

        # Dump to file
        config_file = tmp_path / "test_wrapper_no_init.yaml"
        wrapper.dump(str(config_file))

        # Load from file
        loaded_wrapper = LUMETorchModel.from_file(str(config_file))

        # Should be able to set and get
        loaded_wrapper.set({"input1": 2.0, "input2": 1.5})
        result = loaded_wrapper.get(["output1", "output2"])
        assert result["output1"] == 4.0  # 2.0 * 2
        assert result["output2"] == 3.0  # 1.5 * 2

    def test_from_yaml_string(self, simple_torch_model, tmp_path):
        """Test loading from a YAML string."""
        # First, dump to file
        config_file = tmp_path / "test_yaml_string.yaml"
        initial_inputs = {"input1": 3.0, "input2": 2.5}
        wrapper = LUMETorchModel(simple_torch_model, initial_inputs=initial_inputs)
        wrapper.dump(str(config_file))

        # Read the YAML content
        with open(config_file, "r") as f:
            yaml_content = f.read()

        # Load from YAML string (but we need the config_file path for relative paths)
        loaded_wrapper = LUMETorchModel.from_yaml(yaml_content, str(config_file))

        # Verify it works
        loaded_wrapper.set({"input1": 4.0, "input2": 3.0})
        result = loaded_wrapper.get(["output1", "output2"])
        assert result["output1"] == 8.0
        assert result["output2"] == 6.0

    def test_yaml_config_structure(self, simple_torch_model, tmp_path):
        """Test that the dumped YAML has the expected structure."""
        config_file = tmp_path / "test_structure.yaml"
        initial_inputs = {"input1": 3.0, "input2": 2.5}
        wrapper = LUMETorchModel(simple_torch_model, initial_inputs=initial_inputs)
        wrapper.dump(str(config_file))

        # Load and verify YAML structure
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        assert config["model_class"] == "LUMETorchModel"
        assert "torch_model_file" in config
        assert config["torch_model_file"] == "test_structure_torch_model.yaml"
        assert config["initial_inputs"] == initial_inputs

    def test_set_partial_inputs_preserves_others(self, simple_torch_model):
        """Test that setting only some inputs preserves the others and re-evaluates correctly."""
        initial_inputs = {"input1": 2.0, "input2": 3.0}
        wrapper = LUMETorchModel(simple_torch_model, initial_inputs=initial_inputs)

        # Get initial output
        initial_result = wrapper.get(["output1", "output2"])
        assert initial_result["output1"] == 4.0  # 2.0 * 2
        assert initial_result["output2"] == 6.0  # 3.0 * 2

        # Set only input1, input2 should remain at 3.0
        wrapper.set({"input1": 5.0})

        # Check that input2 wasn't changed
        assert wrapper._state["input2"] == 3.0

        # Check that outputs were recalculated with new input1
        result = wrapper.get(["output1", "output2"])
        assert result["output1"] == 10.0  # 5.0 * 2
        assert result["output2"] == 6.0  # 3.0 * 2 (unchanged)


def test_load_from_nonexistent_file_raises_error():
    """Test that loading from a nonexistent file raises an appropriate error."""
    with pytest.raises(OSError, match="not found"):
        LUMETorchModel.from_file("nonexistent_model.yaml")
