import os

def conditional_mkdir(path):
    if not os.path.exists(path): # path does not exist, create it
        os.makedirs(path)
        return True

    return True

class AttrDict(dict):
    """ Nested Attribute Dictionary

    A class to convert a nested Dictionary into an object with key-values
    accessible using attribute notation (AttrDict.attribute) in addition to
    key notation (Dict["key"]). This class recursively sets Dicts to objects,
    allowing you to recurse into nested dicts (like: AttrDict.attr.attr)

    Adapted from: https://stackoverflow.com/a/48806603
    """

    def __init__(self, mapping=None, *args, **kwargs):
        """Return a class with attributes equal to the input dictionary."""
        super(AttrDict, self).__init__(*args, **kwargs)
        if mapping is not None:
            for key, value in mapping.items():
                self.__setitem__(key, value)

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = AttrDict(value)
        super(AttrDict, self).__setitem__(key, value)
        self.__dict__[key] = value  # for code completion in editors

    def __getattr__(self, item):
        try:
            return self.__getitem__(item)
        except KeyError:
            raise AttributeError(item)
