class OutOfBoundsException (ValueError):
    """Raised when a numeric hyperparameter exceeds allowed bounds.

    Parameters
    ----------
    error_message : str
        Human-readable error message describing the bounds violation.

    Attributes
    ----------
    passed_oob_message : str
        The provided error message.
    """
    def __init__ (self, error_message):
        self.passed_oob_message = error_message

    def __str__ (self):
        return self.passed_oob_message

class InvalidSelectionException (ValueError):
    """Raised when a hyperparameter value is not in the allowed choices.

    Parameters
    ----------
    error_message : str
        Human-readable error message describing the invalid selection.

    Attributes
    ----------
    passed_ie_message : str
        The provided error message.
    """
    def __init__ (self, error_message):
        self.passed_ie_message = error_message

    def __str__ (self):
        return self.passed_ie_message

class NonExistentHyperparameterException (ValueError):
    """Raised when an unknown hyperparameter key is supplied.

    Parameters
    ----------
    error_message : str
        Human-readable error message describing the unknown hyperparameter.

    Attributes
    ----------
    passed_neh_message : str
        The provided error message.
    """
    def __init__ (self, error_message):
        self.passed_neh_message = error_message

    def __str__ (self):
        return self.passed_neh_message

