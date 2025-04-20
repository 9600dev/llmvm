import os
from typing import Dict, Generic, Iterable, List, TypeVar

import dill

K = TypeVar('K')
V = TypeVar('V')

class MemoryCache(Generic[K, V]):
    def __init__(self):
        self.cache: Dict[K, V] = {}

    def get(self, key: K) -> V | None:
        return self.cache.get(key)

    def set(self, key: K, value: V) -> None:
        self.cache[key] = value

    def setup(self) -> None:
        pass

    def delete(self, key: K) -> None:
        self.cache.pop(key, None)

    def has_key(self, key: K) -> bool:
        return key in self.cache

    def keys(self) -> Iterable[K]:
        return self.cache.keys()


class PersistentCache:
    def __init__(self, cache_directory: str):
        if not os.path.exists(cache_directory):
            os.makedirs(cache_directory)

        if not os.path.exists(cache_directory):
            raise Exception(f'Cache directory {cache_directory} does not exist')

        self.cache_directory = cache_directory
        self.cache = {}

    def _serialize_key(self, key: int):
        return dill.dumps(key)

    def _deserialize_key(self, serialized_key: int):
        return dill.loads(serialized_key)

    def _write_cache(self, key: int, value):
        if not os.path.exists(self.cache_directory):
            os.makedirs(self.cache_directory)

        with open(self.cache_directory + f'/{key}.cache', 'wb') as f:
            dill.dump(value, f)

    def set(self, key: int, value):
        serialized_key = self._serialize_key(key)
        self.cache[serialized_key] = value
        self._write_cache(key, value)

    def get(self, key: int):
        cache_hit = self.cache.get(self._serialize_key(key))
        if cache_hit:
            return cache_hit

        if not os.path.exists(self.cache_directory):
            os.makedirs(self.cache_directory)

        with open(self.cache_directory + f'/{key}.cache', 'rb') as f:
            value = dill.load(f)
            self.cache[self._serialize_key(key)] = value
            return value

    def delete(self, key):
        if key in self.cache:
            del self.cache[self._serialize_key(key)]

        if os.path.exists(self.cache_directory + f'/{key}.cache'):
            os.remove(self.cache_directory + f'/{key}.cache')

    def has_key(self, key):
        return self._serialize_key(key) in self.cache or os.path.exists(self.cache_directory + f'/{key}.cache')

    def keys(self) -> List[int]:
        if not os.path.exists(self.cache_directory):
            return []

        result = [int(f.split('.')[0]) for f in os.listdir(self.cache_directory) if f.endswith('.cache') and f[0].isdigit()]
        return list(sorted(result))

    def gen_key(self):
        keys = self.keys()
        return keys[-1] + 1 if keys else 1
