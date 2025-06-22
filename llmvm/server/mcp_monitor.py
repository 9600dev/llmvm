"""
MCP Server Monitor and Integration

This module provides background monitoring for MCP servers and integrates
their tools into the LLMVM helpers system.
"""

import asyncio
import json
import logging
import os
import anyio
import subprocess
import signal
from contextlib import AsyncExitStack
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.types import Tool, TextContent as MCPTextContent

from llmvm.common.helpers import Helpers
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import TextContent

logging = setup_logging()

class MCPToolWrapper:
    def __init__(self, connection: 'MCPServerConnection', tool: Tool):
        self.connection = connection
        self.tool = tool
        self.name = tool.name
        self.description = tool.description
        self.input_schema = tool.inputSchema

        self.__name__ = tool.name
        self.__qualname__ = tool.name
        self.__doc__ = tool.description

        self._extract_parameters()

        def get_str(self):
            Helpers.str_get_str(self)

    def _extract_parameters(self):
        self.__mcp_params__ = []

        if self.input_schema and isinstance(self.input_schema, dict):
            props = self.input_schema.get('properties', {})
            required = set(self.input_schema.get('required', []))

            params = []
            for name, schema in props.items():
                params.append({
                    'name': name,
                    'type': schema.get('type', 'any'),
                    'description': schema.get('description', ''),
                    'required': name in required
                })
            self.__mcp_params__ = params

    def get_function_description(self) -> Dict[str, Any]:
        parameters, types = [], []
        type_map = {
            'string': 'str',
            'number': 'float',
            'integer': 'int',
            'boolean': 'bool',
            'array': 'List',
            'object': 'Dict'
        }

        for param in self.__mcp_params__:
            parameters.append(param['name'])
            types.append(type_map.get(param['type'], 'Any'))

        return {
            "parameters": parameters,
            "types": types,
            "return_type": "Any",
            "invoked_by": self.name,
            "class_name": None,
            "description": self.__doc__ or "",
            "is_async": True
        }

    async def _call_async(self, *args, **kwargs) -> Any:
        payload = self._build_payload(args, kwargs)
        if not self.connection.session:
            asyncio.run(self.connection.connect())

        # double shot try
        if self.connection.session:
            try:
                result = await self.connection.session.call_tool(self.name, payload)
            except anyio.ClosedResourceError:
                asyncio.run(self.connection.connect())
                result = await self.connection.session.call_tool(self.name, payload)

        if all(isinstance(x, MCPTextContent) for x in result.content):
            return '\n'.join([cast(MCPTextContent, x).text for x in result.content])
        else:
            return result.content

    def _run_sync(self, *args, **kwargs) -> Any:
        try:
            # detect if an event loop is already running
            asyncio.get_running_loop()
        except RuntimeError:
            # no loop running, create/run one
            return asyncio.get_event_loop().run_until_complete(
                self._call_async(*args, **kwargs)
            )
        else:
            # loop is running, cannot call sync
            raise RuntimeError("Cannot call sync method inside an active event loop")

    def _build_payload(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Map positional args to parameter names and merge with kwargs."""
        payload: Dict[str, Any] = {}
        for val, param in zip(args, self.__mcp_params__):
            payload[param['name']] = val
        payload.update(kwargs)
        return payload

    def __call__(self, *args, **kwargs) -> Any:
        """
        If called in async context (i.e. inside an active loop), returns a coroutine
        so it can be awaited.
        If called outside an event loop, runs synchronously and returns the result.
        """
        # Try to detect running loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No loop: safe to run sync
            return self._run_sync(*args, **kwargs)
        else:
            # In loop: return coroutine for async usage
            return self._call_async(*args, **kwargs)


    def __repr__(self) -> str:
        return f"<MCPToolWrapper: {self.name}>"


class MCPServerConnection:
    def __init__(self, server_info: Dict[str, Any]):
        self.server_info = server_info
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None
        self.tools: List[MCPToolWrapper] = []
        self.connected = False
        self.connection_type: Optional[str] = None  # 'stdio' or 'sse'
        self.port: Optional[int] = None

    def _get_process_listening_port(self, pid: int) -> Optional[int]:
        """Get listening port for a process using lsof or netstat."""
        try:
            # Try lsof first (more common on Unix-like systems)
            try:
                result = subprocess.run(
                    ['lsof', '-iTCP', '-sTCP:LISTEN', '-P', '-n', f'-p{pid}'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines[1:]:  # Skip header
                        parts = line.split()
                        if len(parts) >= 9:
                            # Extract port from the NAME column (e.g., "*:8080" or "127.0.0.1:8080")
                            name = parts[8]
                            if ':' in name:
                                port_str = name.split(':')[-1]
                                try:
                                    return int(port_str)
                                except ValueError:
                                    continue
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
            
            # Try netstat as fallback
            try:
                result = subprocess.run(
                    ['netstat', '-anp', '2>/dev/null'],
                    capture_output=True,
                    text=True,
                    shell=True,
                    timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if f'{pid}/' in line and 'LISTEN' in line:
                            parts = line.split()
                            if len(parts) >= 4:
                                # Extract port from local address
                                local_addr = parts[3]
                                if ':' in local_addr:
                                    port_str = local_addr.split(':')[-1]
                                    try:
                                        return int(port_str)
                                    except ValueError:
                                        continue
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
            
            # Try ss as another fallback (modern Linux)
            try:
                result = subprocess.run(
                    ['ss', '-tlnp'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if f'pid={pid}' in line:
                            parts = line.split()
                            if len(parts) >= 4:
                                # Extract port from local address
                                local_addr = parts[3]
                                if ':' in local_addr:
                                    port_str = local_addr.split(':')[-1]
                                    try:
                                        return int(port_str)
                                    except ValueError:
                                        continue
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
                
        except Exception as e:
            logging.debug(f"Error getting listening port for PID {pid}: {e}")
        
        return None

    async def connect(self) -> bool:
        try:
            cmdline = self.server_info.get('cmdline', '')
            pid = self.server_info.get('pid')

            # First, try to detect if it's an SSE server by checking for listening ports
            port = self._get_process_listening_port(pid)

            if port:
                # This is likely an SSE server
                logging.info(f"Detected SSE MCP server on port {port} (PID: {pid})")
                self.port = port
                self.connection_type = 'sse'

                # Connect via SSE
                url = f"http://localhost:{port}/sse"
                sse_transport = await self.exit_stack.enter_async_context(
                    sse_client(url)
                )

                read_stream, write_stream = sse_transport
                self.session = await self.exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
            else:
                # No listening port detected - this might be a stdio server
                # Note: We can't connect via stdio to an already-running process,
                # so this will only work if we can restart the server
                logging.info(f"No listening port detected for PID {pid}, assuming stdio server")
                self.connection_type = 'stdio'

                server_script = None
                parts = cmdline.split()

                for part in parts:
                    if part.endswith('.py'):
                        if os.path.exists(part):
                            server_script = part
                            break

                        abs_path = os.path.abspath(part)
                        if os.path.exists(abs_path):
                            server_script = abs_path
                            break

                if not server_script:
                    logging.warning(f"Could not find server script from cmdline: {cmdline}")
                    return False

                server_params = StdioServerParameters(
                    command="python",
                    args=[server_script],
                    env=None
                )

                stdio_transport = await self.exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                stdio, write = stdio_transport

                self.session = await self.exit_stack.enter_async_context(
                    ClientSession(stdio, write)
                )

            await self.session.initialize()

            response = await self.session.list_tools()

            self.tools = [
                MCPToolWrapper(self, tool)
                for tool in response.tools
            ]

            if not response.tools:
                logging.warning(f"Connected but no tools found - may not be a valid MCP server (PID: {pid})")

            self.connected = True
            logging.info(f"Connected to MCP server ({self.connection_type}) with {len(self.tools)} tools")
            return True

        except Exception as e:
            logging.error(f"Failed to connect to MCP server (PID: {self.server_info.get('pid')}): {e}")
            logging.debug(f"Connection details - Type: {self.connection_type}, Port: {self.port}")
            await self.disconnect()
            return False

    async def disconnect(self):
        self.connected = False
        self.tools = []
        await self.exit_stack.aclose()

    def get_tools_as_callables(self) -> Dict[str, MCPToolWrapper]:
        return {tool.name: tool for tool in self.tools}


class MCPMonitor:
    def __init__(self, update_helpers_callback: Optional[Callable[[Dict[str, Callable]], None]] = None,
                 server_patterns: Optional[List[str]] = None,
                 exclude_patterns: Optional[List[str]] = None):
        self.known_servers: Dict[int, MCPServerConnection] = {}
        self.update_helpers_callback = update_helpers_callback
        self.monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None

        self.server_patterns = server_patterns or [
            'currency_server.py',
            'recoll_server.py',
            'streaming_currency_server.py',
            'mcp_server.py',
            'mcp-server.py'
        ]

        self.exclude_patterns = exclude_patterns or [
            'mcp_monitor.py',  # Don't detect ourselves
        ]

    def find_mcp_processes(self) -> Dict[int, dict]:
        """Find MCP server processes using ps command."""
        mcp_processes = {}

        try:
            # Use ps to get process information
            # -e: all processes, -o: output format
            result = subprocess.run(
                ['ps', '-eo', 'pid,comm,args,lstart'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                logging.error(f"ps command failed: {result.stderr}")
                return mcp_processes
            
            lines = result.stdout.strip().split('\n')
            if not lines:
                return mcp_processes
                
            # Skip header line
            for line in lines[1:]:
                try:
                    # Parse ps output
                    parts = line.strip().split(None, 3)  # Split into max 4 parts
                    if len(parts) < 3:
                        continue
                        
                    pid_str = parts[0]
                    name = parts[1]
                    
                    # Handle case where there's no args column
                    if len(parts) >= 3:
                        cmdline = parts[2]
                        if len(parts) > 3:
                            # Append any additional args
                            cmdline = parts[2] + ' ' + parts[3]
                    else:
                        cmdline = name
                    
                    # Convert PID to int
                    try:
                        pid = int(pid_str)
                    except ValueError:
                        continue
                    
                    cmdline_lower = cmdline.lower()
                    
                    # Check exclusion patterns
                    if any(exclude in cmdline_lower for exclude in self.exclude_patterns):
                        continue
                    
                    is_mcp = False
                    
                    # Check for MCP patterns
                    if 'fastmcp' in cmdline_lower:
                        is_mcp = True
                    elif any(pattern in cmdline_lower for pattern in self.server_patterns):
                        is_mcp = True
                    elif 'mcp.run' in cmdline_lower or 'mcp run' in cmdline_lower:
                        is_mcp = True
                    elif 'mcp' in cmdline_lower and cmdline_lower.endswith('.py'):
                        if 'server' in cmdline_lower:
                            is_mcp = True
                    
                    if is_mcp:
                        # Get start time (approximation using ps)
                        mcp_processes[pid] = {
                            'pid': pid,
                            'name': name,
                            'cmdline': cmdline,
                            'started': 'Unknown'  # ps doesn't provide exact start time in standard format
                        }
                        
                except Exception as e:
                    logging.debug(f"Error parsing ps line: {line}, error: {e}")
                    continue
                    
        except subprocess.TimeoutExpired:
            logging.error("ps command timed out")
        except Exception as e:
            logging.error(f"Error finding MCP processes: {e}")
            
        return mcp_processes

    async def handle_new_server(self, pid: int, server_info: dict):
        cmdline = server_info.get('cmdline', 'Unknown')
        logging.info(f"Potential MCP server detected - PID: {pid}, Command: {cmdline[:100]}...")

        connection = MCPServerConnection(server_info)

        if await connection.connect():
            self.known_servers[pid] = connection

            await self.update_all_helpers()
        else:
            logging.debug(f"Could not establish MCP connection to PID: {pid}")

    async def handle_stopped_server(self, pid: int):
        logging.info(f"MCP server stopped - PID: {pid}")

        if pid in self.known_servers:
            connection = self.known_servers[pid]
            await connection.disconnect()
            del self.known_servers[pid]

            # Update helpers to remove tools
            await self.update_all_helpers()

    async def update_all_helpers(self):
        all_mcp_tools = {}

        for pid, connection in self.known_servers.items():
            if connection.connected:
                tools = connection.get_tools_as_callables()
                all_mcp_tools.update(tools)

        if self.update_helpers_callback:
            self.update_helpers_callback(all_mcp_tools)
            logging.info(f"Updated helpers with {len(all_mcp_tools)} MCP tools")

    async def monitor_loop(self, interval: int = 5):
        logging.info("Starting MCP server monitoring...")
        logging.debug(f"Looking for patterns: {self.server_patterns}")
        logging.debug(f"Excluding patterns: {self.exclude_patterns}")

        current_processes = self.find_mcp_processes()
        current_pids = set(current_processes.keys())

        if current_pids:
            logging.info(f"Found {len(current_pids)} potential MCP server(s) on startup")

        for pid, info in current_processes.items():
            await self.handle_new_server(pid, info)

        while self.monitoring:
            await asyncio.sleep(interval)

            current_processes = self.find_mcp_processes()
            current_pids = set(current_processes.keys())
            known_pids = set(self.known_servers.keys())

            new_servers = current_pids - known_pids
            for pid in new_servers:
                await self.handle_new_server(pid, current_processes[pid])

            stopped_servers = known_pids - current_pids
            for pid in stopped_servers:
                await self.handle_stopped_server(pid)

    async def start(self, interval: int = 5):
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_task = asyncio.create_task(self.monitor_loop(interval))

    async def stop(self):
        self.monitoring = False

        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        for connection in list(self.known_servers.values()):
            await connection.disconnect()

        self.known_servers.clear()

    async def get_current_tools_async(self) -> Dict[str, MCPToolWrapper]:
        current_processes = self.find_mcp_processes()
        current_pids = set(current_processes.keys())
        known_pids = set(self.known_servers.keys())

        stopped_servers = known_pids - current_pids
        for pid in stopped_servers:
            if pid in self.known_servers:
                await self.known_servers[pid].disconnect()
                del self.known_servers[pid]

        new_servers = current_pids - known_pids
        for pid in new_servers:
            server_info = current_processes[pid]
            connection = MCPServerConnection(server_info)

            if await connection.connect():
                self.known_servers[pid] = connection
                logging.info(f"Connected to MCP server {pid} with {len(connection.tools)} tools")

        all_tools = {}
        for pid, connection in self.known_servers.items():
            if connection.connected:
                tools = connection.get_tools_as_callables()
                all_tools.update(tools)

        return all_tools
