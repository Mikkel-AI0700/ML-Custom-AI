class OutOfBoundsException (ValueError):
    def __init__ (self, error_message):
        self.passed_oob_message = error_message

    def __str__ (self):
        return self.passed_oob_message

class InvalidSelectionException (ValueError):
    def __init__ (self, error_message)
        self.passed_ie_message = error_message

    def __str__ (self):
        return self.passed_ie_message

class NonExistentHyperparameterException (ValueError):
    def __init__ (self, error_message):
        self.passed_neh_message = error_message

    def __str__ (self):
        return self.passed_neh_message

