from typing import Union, Any, Callable
from inspect import signature, bind
from errors.ParameterErrors import (
    OutOfBoundsException,
    InvalidSelectionException,
    NonExistentHyperparameterException
)

class ParameterValidator:
    def __init__ (self):
        self.INVALID_NUMERIC_ARGUMENT = (
            "[-] Error: A string has been detected on a int/float-only hyperparameter"
            "Detected invalid string argument: {} | Parameter constraint: {}"
        )
        self.INVALID_STRING_ARGUMENT = (
            "[-] Error: A numerical value has been detected on a string-only hyperparameter"
            "Detected invalid numeric argument: {} | Parameter constraint: {}"
        )

    def _map_parameter_arguments (self, user_args: list[Any], model_instance: Callable):
        model_signature = signature(model_instance)
        parameter = model_signature.bind(user_args)
        return parameter.kwargs

    def validate_parameters (
        self,
        user_arguments: list[Any],
        model_instance: Callable,
        parameter_constraints: dict[str, Any]
    ):
        parameters = self._map_parameter_arguments(user_arguments, model_instance)
        for parameter, parameter_constraint in zip(parameters.values(), parameter_constraints.values()):
            constraint_type = parameter_constraint.get("type")
            minimum_threshold = parameter_constraint.get("min_thresh", None)
            maximum_threshold = parameter_constraint.get("max_thresh", None)
            choices_selection = parameter_constraint.get("choices", None)
            
            try:
                if constraint_type == "int" or constraint_type == "float":
                    if parameter > maximum_threshold or parameter < minimum_threshold:
                        raise ValueError(
                            self.INVALID_NUMERIC_ARGUMENT.format(parameter, parameter_constraint)
                        )

                if constraint_type == "string":
                    if parameter not in choices_selection:
                        raise ValueError(
                            self.INVALID_STRING_ARGUMENT.format(parameter, parameter_constraint)
                        )
            except ValueError as incorrect_argument:
                print(incorrect_argument)
                exit(1)
            else:
                return True

