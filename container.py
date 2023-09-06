import inspect
import os
from typing import Dict, Type

import yaml


class Singleton (type):
    _instances = {}  # type: ignore

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Container(metaclass=Singleton):
    def __init__(self, config_file: str = os.path.expanduser('~/.config/llmvm/config.yaml')):
        self.config_file = config_file

        # try finding the filename
        def find_file(filename, search_path='.'):
            for root, dir, files in os.walk(search_path):
                if filename in files:
                    return os.path.join(root, filename)
            return None

        self.config_file = find_file(self.config_file) or self.config_file  # type: ignore

        if not os.path.exists(self.config_file) and not find_file(self.config_file):  # type: ignore
            raise ValueError('configuration_file is not found {} or TRADER_CONFIG set incorrectly'.format(config_file))

        with open(self.config_file, 'r') as conf_file:
            self.configuration: Dict = yaml.load(conf_file, Loader=yaml.FullLoader)
            self.type_instance_cache: Dict[Type, object] = {}

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

    def get(self, key: str) -> str:
        value = self.configuration[key]
        if isinstance(value, str) and '~' in value and '/' in value:
            return os.path.expanduser(value)
        else:
            return value

    def has(self, key: str) -> bool:
        return key in self.configuration

    def config(self) -> Dict:
        return self.configuration

    def resolve_cache(self, t: Type, **extra_args):
        if t in self.type_instance_cache:
            return self.type_instance_cache[t]
        else:
            self.type_instance_cache[t] = self.resolve(t, extra_args=extra_args)
            return self.type_instance_cache[t]
