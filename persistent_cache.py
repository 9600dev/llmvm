import os
from typing import Optional

import dill


class PersistentCache:
    def __init__(self, filename: Optional[str] = None):
        self.filename = filename

        if self.filename:
            if not os.path.isfile(self.filename):
                with open(self.filename, 'wb') as f:
                    dill.dump({}, f)

    def _serialize_key(self, key):
        return dill.dumps(key)

    def _deserialize_key(self, serialized_key):
        return dill.loads(serialized_key)

    def set(self, key, value):
        if self.filename:
            with open(self.filename, 'rb+') as f:
                cache = dill.load(f)
                serialized_key = self._serialize_key(key)
                cache[serialized_key] = value
                f.seek(0)
                dill.dump(cache, f)

    def get(self, key):
        if self.filename:
            with open(self.filename, 'rb') as f:
                cache = dill.load(f)
                serialized_key = self._serialize_key(key)
                return cache.get(serialized_key)
        return None

    def delete(self, key):
        if self.filename:
            with open(self.filename, 'rb+') as f:
                cache = dill.load(f)
                serialized_key = self._serialize_key(key)
                if serialized_key in cache:
                    del cache[serialized_key]
                    f.seek(0)
                    dill.dump(cache, f)

    def has_key(self, key):
        if self.filename:
            with open(self.filename, 'rb') as f:
                cache = dill.load(f)
                serialized_key = self._serialize_key(key)
                return serialized_key in cache
        return False

    def keys(self):
        if self.filename:
            with open(self.filename, 'rb') as f:
                cache = dill.load(f)
                return list(cache.keys())  # type: ignore
        return []
