import os

def conditional_mkdir(path):
    if not os.path.exists(path): # path does not exist, create it
        os.makedirs(path)
        return True

    return True

class AttrDict(dict):
    """Class to convert a dictionary to a class.

    Parameters
    ----------
    dict: dictionary

    """

    def __init__(self, *args, **kwargs):
        """Return a class with attributes equal to the input dictionary."""
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        self.check_required_arguments()

    def check_required_arguments(self):
        """Check that the config object has required attributes."""
        if not hasattr(self, 'root_dir'):
            print(f"Error: required argument 'root_dir' is not configured.")
            raise AttributeError

