import os
import pytest
import yaml

from lume_torch.base import LUMETorch
from lume_torch.variables import ScalarVariable


class ExampleModel(LUMETorch):
    def _evaluate(self, input_dict):
        pass


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
            ScalarVariable.__name__
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

    def test_output_validation(self, simple_variables):
        example_model = ExampleModel(**simple_variables)
        output_variables = simple_variables["output_variables"]
        output_dict = {output_variables[0].name: 3.0, output_variables[1].name: 1.7}
        example_model.output_validation(output_dict)
        with pytest.raises(TypeError):
            output_dict[output_variables[0].name] = "test"
            example_model.output_validation(output_dict)
