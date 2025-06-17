#!/usr/bin/env python3
"""
recollq_json_fullpath.py: A wrapper to run recollq and output JSON including full file paths.

Usage:
  ./recollq_json_fullpath.py "search query"
  ./recollq_json_fullpath.py -f "url filename abstract mtime size" -l 10 -o output.json "search query"
"""
import subprocess
import base64
import json
import sys
from typing import List, Dict, Any

class RecollQ:
    def __init__(
        self,
        recollq_cmd: str = "recollq",
        fields: List[str] = None,
    ):
        """
        :param recollq_cmd: Path to the recollq binary
        :param fields: List of fields to retrieve (default includes 'url' for full path)
        """
        self.cmd = recollq_cmd
        # Default fields include 'url' to get file:// URLs
        self.fields = fields or ["url", "filename", "abstract", "mtime", "size"]

    def search(self, query: str, extra_opts: List[str] = None) -> List[Dict[str, Any]]:
        """
        Run recollq and return a list of dicts, each representing one hit.
        """
        # Build recollq command: -N prints query and hit count, -F specifies fields
        opts = ["-N", "-F", " ".join(self.fields)]
        if extra_opts:
            opts.extend(extra_opts)
        cmd = [self.cmd] + opts + [query]

        # Execute recollq
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        lines = proc.stdout.decode("utf-8", errors="replace").splitlines()

        # If fewer than 3 lines, there are no results
        if len(lines) < 3:
            return []

        results: List[Dict[str, Any]] = []
        # Data starts after the two header lines
        for raw in lines[2:]:
            raw = raw.strip()
            if not raw:
                continue

            # Split into alternating key/value tokens
            parts = raw.split()
            if len(parts) % 2 != 0:
                # Unexpected format; skip
                continue

            entry: Dict[str, Any] = {}
            for key, cell in zip(parts[0::2], parts[1::2]):
                # Decode base64 values, fallback to raw on error
                try:
                    entry[key] = base64.b64decode(cell).decode("utf-8", errors="replace")
                except Exception:
                    entry[key] = cell

            # Build 'path' by stripping file:// prefix from 'url'
            url = entry.get("url", "")
            if url.startswith("file://"):
                entry["path"] = url[len("file://"):]
            else:
                entry["path"] = entry.get("filename", "")

            # Convert numeric fields to int
            for fld in ("mtime", "size"):
                if fld in entry:
                    try:
                        entry[fld] = int(entry[fld])
                    except ValueError:
                        pass

            results.append(entry)

        return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="recollq → JSON with full paths")
    parser.add_argument(
        "-f", "--fields",
        help="Space-separated fields to retrieve (default: 'url filename abstract mtime size')",
        default="url filename abstract mtime size"
    )
    parser.add_argument(
        "-l", "--limit",
        type=int,
        help="Limit the number of results to the top N hits",
        default=None
    )
    parser.add_argument(
        "-o", "--output",
        help="Write JSON to this file (default stdout)"
    )
    parser.add_argument(
        "query",
        help="Recoll search query (in quotes if containing spaces)"
    )
    args = parser.parse_args()

    # Normalize fields list
    fields = args.fields.split()
    wrapper = RecollQ(fields=fields)
    hits = wrapper.search(args.query)

    # Apply limit if specified
    if args.limit is not None:
        hits = hits[:args.limit]

    json_out = json.dumps(hits, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fp:
            fp.write(json_out)
    else:
        sys.stdout.write(json_out)
