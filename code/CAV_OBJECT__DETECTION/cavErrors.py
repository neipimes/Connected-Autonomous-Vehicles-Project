#https://stackoverflow.com/questions/1319615/proper-way-to-declare-custom-exceptions-in-modern-python

class CameraStreamError(Exception):
    def __init__(self, message, errors=None):            
        # Call the base class constructor with the parameters it needs
        super().__init__(message)
        self.message = message    
        self.errors = errors