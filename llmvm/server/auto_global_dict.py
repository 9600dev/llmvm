from typing import Optional

class AutoGlobalDict(dict):
    def __init__(
            self,
            globals_dict: Optional[dict] = None,
            locals_dict: Optional[dict] = None
        ):
        super().__init__()
        import builtins

        # Add builtins first
        for name in dir(builtins):
            attr = getattr(builtins, name)
            if callable(attr):
                self[name] = attr

        if globals_dict is not None:
            self.update(globals_dict)
        if locals_dict is not None:
            self.update(locals_dict)

        # Add self reference
        self['AutoGlobalDict'] = self

    def __missing__(self, key):
        """Handle missing keys by checking globals."""
        global_vars = globals()
        if key in global_vars:
            value = global_vars[key]
            self[key] = value  # Cache the value
            return value
        raise KeyError(f"name '{key}' is not defined")

    def __getitem__(self, key):
        """Get an item, falling back to __missing__ if not found."""
        try:
            return super().__getitem__(key)
        except KeyError:
            return self.__missing__(key)

    def __setitem__(self, key, value):
            return super().__setitem__(key, value)

    def copy(self) -> 'AutoGlobalDict':
        """Create a new AutoGlobalDict with the same contents."""
        return AutoGlobalDict(dict(self))

