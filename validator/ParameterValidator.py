from typing import Union, Any, Callable
from inspect import signature, bind

class ParameterValidator:
    def _map_parameter_arguments (self, user_args: list[Any], model_instance: Callable):
        model_signature = signature(model_instance)
        parameter = model_signature.bind(user_args)
        return parameter.kwargs

    def __call__ (
        self,
        user_supplied_arguments: list[Any],
        model_instance: Callable,
        parameter_constraints: dict[str, Any]
    ):
        parameters = self._map_parameter_arguments(user_supplied_arguments, model_instance)
        for param, param_const in zip(parameters.values(), parameter_constraints.values()):
            if not isinstance(param, param_const):
                raise TypeError()
        return True

