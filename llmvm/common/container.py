import inspect
import os
from typing import Any, Type, cast

import yaml


class Singleton (type):
    _instances = {}  # type: ignore

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Container(metaclass=Singleton):
    def __init__(
            self,
            config_file: str = os.path.expanduser('~/.config/llmvm/config.yaml'),
            throw: bool = True
        ):
        self.config_file = config_file

        if os.getenv('LLMVM_CONFIG'):
            self.config_file = cast(str, os.getenv('LLMVM_CONFIG'))

        if not os.path.exists(self.config_file) and throw:
            raise ValueError('configuration_file {} is not found. Put config in ~/.config/llmvm or set LLMVM_CONFIG'.format(config_file))
        elif not os.path.exists(self.config_file) and not throw:
            return

        with open(self.config_file, 'r') as conf_file:
            self.configuration: dict = yaml.load(conf_file, Loader=yaml.FullLoader)  # type: ignore
            self.type_instance_cache: dict[Type, object] = {}

    def resolve(self, t: Type, **extra_args):
        args = {}
        for param in inspect.signature(t.__init__).parameters.values():
            if param == 'self':
                continue
            if extra_args and param.name in extra_args.keys():
                args[param.name] = extra_args[param.name]
            elif os.getenv(param.name.upper()):
                args[param.name] = os.getenv(param.name.upper())
            elif param.name in self.configuration and self.configuration[param.name]:
                args[param.name] = self.configuration[param.name]
        return t(**args)

    def get(self, key: str, default: Any = '') -> Any:
        if key not in self.configuration:
            return default

        value = self.configuration[key]
        if isinstance(value, str) and '~' in value and '/' in value:
            return os.path.expanduser(value)
        else:
            return value

    def has(self, key: str) -> bool:
        return key in self.configuration

    def config(self) -> dict:
        return self.configuration

    def resolve_cache(self, t: Type, **extra_args):
        if t in self.type_instance_cache:
            return self.type_instance_cache[t]
        else:
            self.type_instance_cache[t] = self.resolve(t, extra_args=extra_args)
            return self.type_instance_cache[t]

    @staticmethod
    def get_config_variable(name: str, alternate_name: str = '', default: Any = '') -> Any:
        def parse(value) -> Any:
            if isinstance(value, str) and (value == 'true' or value == 'True'):
                return True
            elif isinstance(value, str) and (value == 'false' or value == 'False'):
                return False
            elif isinstance(value, str) and value.lower() == 'none':
                return None
            elif isinstance(value, str) and str.isnumeric(value):
                return int(value)
            elif isinstance(value, str) and str.isdecimal(value):
                return float(value)
            else:
                return value

        if isinstance(default, str) and default.startswith('~'):
            default = os.path.expanduser(default)

        # environment variables take precendence
        if name in os.environ:
            return parse(os.environ.get(name, default))

        if alternate_name in os.environ:
            return parse(os.environ.get(alternate_name, default))

        # otherwise, try the config file
        config_file = os.environ.get('LLMVM_CONFIG', default='~/.config/llmvm/config.yaml')
        if config_file.startswith('~'):
            config_file = os.path.expanduser(config_file)

        if not os.path.exists(config_file):
            return parse(default)

        container = Container(config_file)
        if container.has(name.replace('LLMVM_', '').lower()):
            return parse(container.get(name.replace('LLMVM_', '').lower()))
        elif container.has(alternate_name.replace('LLMVM_', '').lower()):
            return parse(container.get(alternate_name.replace('LLMVM_', '').lower()))
        else:
            return default
