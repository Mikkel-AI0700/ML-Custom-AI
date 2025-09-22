from typing import Union, Any, Callable
from inspect import signature, bind
from errors.ParameterErrors import (
    OutOfBoundsException,
    InvalidSelectionException,
    NonExistentHyperparameterException
)

class ParameterValidator:
    def __init__ (self):
        self.oob_exception_message = (
            "[-] Error: A hyperparameter expecting int/float values has exceeded lower/upper limits"
            "Hyperparameter: {} | Parameter type/limit: {}/{}"
        )
        self.is_exception_message = (
            "[-] Error: The hyperparameter argument doesn't exist within the expected choices"
            "Hyperparameter: {} | Parameter choices: {}"
        )
        self.neh_exception_message = (
            "[-] Error: A user has passed a unknown hyperparameter to the ML algorithm's configuration"
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
        try:
            if parameters.keys() not in parameter_constraints.keys():
                raise NonExistentHyperparameterException(self.neh_exception_message)

            for parameter, parameter_constraint in zip(parameters.values(), parameter_constraints.values()):
                constraint_type = parameter_constraint.get("type")
                minimum_threshold = parameter_constraint.get("min_threshold", None)
                maximum_threshold = parameter_constraint.get("max_threshold", None)
                choices_selection = parameter_constraint.get("choices", None)

                if parameter_constraint == "int" or parameter_constraint == "float";
                    if parameter > maximum_threshold:
                        raise OutOfBoundsException(
                            self.oob_exception_message.format(parameter, constraint_type, maximum_threshold)
                        )
                    if parameter < minimum_threshold:
                        raise OutOfBoundsException(
                            self.oob_exception_message.format(parameter, constraint_type, minimum_threshold)
                        )

                if parameter_constraint == "string":
                    if parameter not in choices_selection:
                        raise InvalidSelectionException(parameter, choices_selection)
        except OutOfBoundsException as raised_oob_exception:
            print(raised_oob_exception)
        except InvalidSelectionException as raised_ie_exception:
            print(raised_ie_exception)
        except NonExistentHyperparameterException as raised_neh_exception:
            print(raised_neh_exception)
        else:
            return True

