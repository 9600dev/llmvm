#!/usr/bin/env python3
"""
Recoll MCP Server - A streamable MCP server for searching with Recoll
"""
import subprocess
import base64
import click
from typing import Dict, List, Optional, Any
from fastmcp import FastMCP

# Create the FastMCP server instance
mcp = FastMCP("Recoll Search Server (Streaming)")

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

    async def search(self, query: str, extra_opts: List[str] = None) -> List[Dict[str, Any]]:
        """
        Run recollq and return a list of dicts, each representing one hit.
        """
        # Build recollq command: -N prints query and hit count, -F specifies fields
        opts = ["-N", "-F", " ".join(self.fields)]
        if extra_opts:
            opts.extend(extra_opts)
        cmd = [self.cmd] + opts + [query]

        # Execute recollq asynchronously
        import asyncio
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            return [{
                "error": "Recoll command failed",
                "message": stderr.decode("utf-8", errors="replace")
            }]
        
        lines = stdout.decode("utf-8", errors="replace").splitlines()

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

# Initialize the RecollQ instance
recoll = RecollQ()

@mcp.tool
async def search_documents(
    query: str,
    limit: Optional[int] = 10,
    fields: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Search documents using Recoll full-text search.
    
    Args:
        query: The search query (supports Recoll query syntax)
        limit: Maximum number of results to return (default: 10)
        fields: List of fields to retrieve (default: url, filename, abstract, mtime, size)
    
    Returns:
        A list of matching documents with their metadata
    """
    # Create a new RecollQ instance with custom fields if provided
    search_instance = RecollQ(fields=fields) if fields else recoll
    
    # Perform the search
    results = await search_instance.search(query)
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        results = results[:limit]
    
    return results

@mcp.tool
async def search_by_type(
    query: str,
    file_type: str,
    limit: Optional[int] = 10
) -> List[Dict[str, Any]]:
    """
    Search documents of a specific type using Recoll.
    
    Args:
        query: The search query
        file_type: File type to filter (e.g., "pdf", "txt", "doc", "py")
        limit: Maximum number of results to return
    
    Returns:
        A list of matching documents of the specified type
    """
    # Combine query with mime type filter
    full_query = f"{query} mime:{file_type}"
    
    results = await recoll.search(full_query)
    
    if limit is not None and limit > 0:
        results = results[:limit]
    
    return results

@mcp.tool
async def search_in_directory(
    query: str,
    directory: str,
    limit: Optional[int] = 10
) -> List[Dict[str, Any]]:
    """
    Search documents within a specific directory using Recoll.
    
    Args:
        query: The search query
        directory: Directory path to search in
        limit: Maximum number of results to return
    
    Returns:
        A list of matching documents within the specified directory
    """
    # Use Recoll's dir: filter
    full_query = f"{query} dir:{directory}"
    
    results = await recoll.search(full_query)
    
    if limit is not None and limit > 0:
        results = results[:limit]
    
    return results

@mcp.tool
async def search_recent(
    query: str,
    days: int = 7,
    limit: Optional[int] = 10
) -> List[Dict[str, Any]]:
    """
    Search recently modified documents using Recoll.
    
    Args:
        query: The search query
        days: Number of days to look back (default: 7)
        limit: Maximum number of results to return
    
    Returns:
        A list of matching documents modified within the specified time period
    """
    # Use Recoll's date filter
    full_query = f"{query} date:{days}d"
    
    results = await recoll.search(full_query)
    
    if limit is not None and limit > 0:
        results = results[:limit]
    
    return results

@mcp.tool
async def stream_search_updates(
    queries: List[str],
    interval_seconds: int = 300
) -> Dict[str, Any]:
    """
    Stream search results for multiple queries.
    Note: This returns current results - in a real streaming scenario,
    this would be called periodically by the client to check for new documents.
    
    Args:
        queries: List of search queries to monitor
        interval_seconds: Update interval in seconds (for client reference)
    
    Returns:
        Current search results for all queries
    """
    results = {}
    
    for query in queries:
        search_results = await recoll.search(query)
        results[query] = {
            "count": len(search_results),
            "latest": search_results[:5] if search_results else [],
            "query": query
        }
    
    return {
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "interval_seconds": interval_seconds,
        "results": results
    }

# Add resources
@mcp.resource("recoll://server/info")
async def get_server_info() -> Dict[str, Any]:
    """Get information about the Recoll search server."""
    return {
        "name": "Recoll Search Server (Streaming)",
        "version": "1.0.0",
        "transport": "SSE (Server-Sent Events)",
        "description": "Provides full-text search capabilities using Recoll via SSE transport",
        "data_source": "Local Recoll index",
        "streaming_support": "Yes - use stream_search_updates tool",
        "available_tools": [
            "search_documents",
            "search_by_type",
            "search_in_directory",
            "search_recent",
            "stream_search_updates"
        ],
        "query_syntax": "Supports full Recoll query syntax including boolean operators, phrases, and filters"
    }

@mcp.resource("recoll://search/capabilities")
async def get_search_capabilities() -> Dict[str, Any]:
    """Get detailed information about search capabilities."""
    return {
        "supported_filters": {
            "mime": "Filter by MIME type (e.g., mime:pdf)",
            "dir": "Filter by directory (e.g., dir:/home/user/docs)",
            "date": "Filter by date (e.g., date:7d for last 7 days)",
            "size": "Filter by file size",
            "author": "Filter by document author",
            "title": "Filter by document title"
        },
        "query_operators": {
            "AND": "Both terms must be present (default)",
            "OR": "Either term must be present",
            "NOT": "Exclude documents with term",
            "\"phrase\"": "Exact phrase search",
            "*": "Wildcard (e.g., doc* matches document, docs, etc.)"
        },
        "available_fields": [
            "url", "filename", "abstract", "mtime", "size",
            "mimetype", "author", "title", "keywords"
        ]
    }

# Add prompts
@mcp.prompt
def code_search_prompt(language: str, pattern: str) -> str:
    """Generate a prompt for searching code files."""
    return f"""Search for {language} code files containing: {pattern}

Use the search_by_type tool with file_type="{language}" to find relevant code files.
Look for function definitions, class declarations, or specific patterns in the code.

Provide a summary of the most relevant matches with their file paths."""

@mcp.prompt
def document_search_prompt(topic: str, file_types: List[str]) -> str:
    """Generate a prompt for searching documents by topic."""
    types_str = ", ".join(file_types) if file_types else "all types"
    return f"""Search for documents about: {topic}

File types to search: {types_str}

Use the appropriate search tools to find relevant documents.
Provide a summary of the most relevant matches including:
- File path
- Brief description or abstract
- Last modified date
- File size"""

@mcp.prompt
def monitor_changes_prompt(directories: List[str], keywords: List[str]) -> str:
    """Generate a prompt for monitoring document changes."""
    return f"""Monitor the following directories for new or modified documents: {', '.join(directories)}

Keywords to watch for: {', '.join(keywords)}

Use the stream_search_updates tool to periodically check for new documents matching these criteria.
Alert when new documents are found that match the keywords in the specified directories."""

@click.command()
@click.option(
    "-p", "--port",
    default=8071,
    type=int,
    help="TCP port number to listen on"
)
@click.option(
    "--recollq-path",
    default="recollq",
    help="Path to the recollq binary"
)
def main(port, recollq_path):
    # Update the global recoll instance with custom path if provided
    global recoll
    recoll = RecollQ(recollq_cmd=recollq_path)
    
    print(f"Starting Recoll Search MCP Server (SSE) on port {port}")
    print(f"Server will be available at: http://localhost:{port}/sse")
    print(f"Using recollq at: {recollq_path}")
    print("Press Ctrl+C to stop\n")

    mcp.run(
        transport="sse",
        host="127.0.0.1",  # Use localhost for security
        port=port,
    )

if __name__ == "__main__":
    main()