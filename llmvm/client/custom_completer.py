import time
from prompt_toolkit.completion import Completer as PromptCompleter
from prompt_toolkit.completion import Completion as PromptCompletion
from functools import lru_cache
import os
import fnmatch
from prompt_toolkit.completion import Completer, Completion as PromptCompletion
from typing import Dict, Set, Iterator

class CustomCompleter(PromptCompleter):
    def __init__(self, max_depth: int = 2):
        super().__init__()
        self.max_depth = max_depth
        self.cache: Dict[str, Set[str]] = {}
        self.last_refresh = 0
        self.cache_ttl = 5  # seconds

    @lru_cache(maxsize=1000)
    def is_dir(self, path: str) -> bool:
        """Cached directory check"""
        return os.path.isdir(path) and not os.path.islink(path)

    def get_path_before_cursor(self, document) -> str:
        text_before_cursor = document.text_before_cursor
        last_space = text_before_cursor.rfind(' ')
        if last_space == -1:
            return text_before_cursor
        return text_before_cursor[last_space + 1:]

    def refresh_cache_if_needed(self, current_dir: str) -> None:
        """Refresh the file/directory cache if TTL has expired"""
        current_time = time.time()
        if current_time - self.last_refresh > self.cache_ttl:
            self.cache.clear()
            self.is_dir.cache_clear()
            self.last_refresh = current_time

    def get_filtered_entries(self, base_dir: str, current_path: str, filter_out: list, depth: int = 0) -> Iterator[str]:
        """Get filtered entries with depth limit and caching"""
        if depth > self.max_depth:
            return

        dir_path = os.path.join(base_dir, current_path) if current_path else base_dir
        cache_key = dir_path

        if cache_key in self.cache:
            yield from self.cache[cache_key]
            return

        try:
            entries = set()
            for entry in os.scandir(dir_path):  # Using scandir instead of listdir for better performance
                if any(fnmatch.fnmatch(entry.name, pattern) for pattern in filter_out):
                    continue

                entry_path = os.path.join(current_path, entry.name) if current_path else entry.name
                entries.add(entry_path)

                # Only recurse if it's a directory and we haven't hit max depth
                if entry.is_dir() and not entry.is_symlink() and depth < self.max_depth:
                    entries.update(self.get_filtered_entries(base_dir, entry_path, filter_out, depth + 1))

            self.cache[cache_key] = entries
            yield from entries

        except Exception:
            return

    def get_completions(self, document, complete_event):
        word = self.get_path_before_cursor(document)
        current_dir = os.getcwd()
        filter_out = ['.git', '.venv', '.vscode', '.pytest_cache', '__pycache__']

        # Refresh cache if needed
        self.refresh_cache_if_needed(current_dir)

        # Get all possible completions
        for entry_path in self.get_filtered_entries(current_dir, "", filter_out):
            if len(word) == 0 or entry_path.startswith(word):
                full_path = os.path.join(current_dir, entry_path)

                # Use cached directory check
                if self.is_dir(full_path):
                    style = "bg:#2C3E50 fg:#E8F6F3 bold"
                else:
                    style = "bg:#34495E fg:#ECF0F1"

                yield PromptCompletion(
                    entry_path,
                    start_position=-len(word),
                    style=style
                )