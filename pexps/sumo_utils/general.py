"""
Some general helper classes 
"""
from imported_utils import *

class Namespace(Namespace):
    """
    A wrapper around dictionary with nicer formatting for the Entity subclasses
    """
    @classmethod
    def format(cls, x):
        def helper(x):
            if isinstance(x, Entity):
                return f"{type(x).__name__}('{x.id}')"
            elif isinstance(x, (list, tuple, set, np.ndarray)):
                return [helper(y) for y in x]
            elif isinstance(x, np.generic):
                return x.item()
            elif isinstance(x, dict):
                return {helper(k): helper(v) for k, v in x.items()}
            return x
        return format_yaml(helper(x))

    def __repr__(self):
        return Namespace.format(self)

class Entity(Namespace):
    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return self.id

    def __repr__(self):
        inner_content = Namespace.format(dict(self)).replace('\n', '\n  ').rstrip(' ')
        return f"{type(self).__name__}('{self.id}',\n  {inner_content})\n\n"
