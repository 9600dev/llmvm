#!/usr/bin/env python3
"""
MCP Server Monitor

Monitor MCP servers and alert when they start/stop.
"""

import asyncio
import psutil
import time
from datetime import datetime
from typing import Set, Dict

class MCPMonitor:
    def __init__(self):
        self.known_servers: Set[int] = set()
        self.server_info: Dict[int, dict] = {}
    
    def find_mcp_processes(self) -> Dict[int, dict]:
        """Find all MCP-related processes"""
        mcp_processes = {}
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
            try:
                cmdline_list = proc.info.get('cmdline', [])
                if cmdline_list:
                    cmdline = ' '.join(cmdline_list)
                else:
                    cmdline = proc.info.get('name', 'Unknown')
                    
                if any(pattern in cmdline.lower() for pattern in [
                    'fastmcp', 'mcp.run', 'mcp-server', 'mcp_server', 'currency_server'
                ]):
                    mcp_processes[proc.info['pid']] = {
                        'name': proc.info['name'],
                        'cmdline': cmdline[:100] + '...' if len(cmdline) > 100 else cmdline,
                        'started': datetime.fromtimestamp(proc.info['create_time']).strftime('%Y-%m-%d %H:%M:%S')
                    }
            except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
                continue
        
        return mcp_processes
    
    async def monitor(self, interval: int = 5):
        """Monitor for MCP server changes"""
        print("🔍 Starting MCP Server Monitor...")
        print(f"Checking every {interval} seconds\n")
        
        # Initial scan
        self.known_servers = set(self.find_mcp_processes().keys())
        self.server_info = self.find_mcp_processes()
        
        if self.known_servers:
            print("📊 Found existing MCP servers:")
            for pid, info in self.server_info.items():
                print(f"   PID {pid}: {info['cmdline']}")
            print()
        else:
            print("📊 No MCP servers currently running\n")
        
        while True:
            current_processes = self.find_mcp_processes()
            current_pids = set(current_processes.keys())
            
            # Check for new servers
            new_servers = current_pids - self.known_servers
            for pid in new_servers:
                info = current_processes[pid]
                print(f"🟢 NEW MCP Server Started:")
                print(f"   PID: {pid}")
                print(f"   Command: {info['cmdline']}")
                print(f"   Started: {info['started']}\n")
            
            # Check for stopped servers
            stopped_servers = self.known_servers - current_pids
            for pid in stopped_servers:
                info = self.server_info.get(pid, {})
                print(f"🔴 MCP Server Stopped:")
                print(f"   PID: {pid}")
                print(f"   Command: {info.get('cmdline', 'Unknown')}\n")
            
            # Update tracking
            self.known_servers = current_pids
            self.server_info = current_processes
            
            # Show current status periodically
            if new_servers or stopped_servers:
                if current_pids:
                    print(f"📊 Currently running: {len(current_pids)} MCP server(s)")
                else:
                    print("📊 No MCP servers currently running")
                print()
            
            await asyncio.sleep(interval)

if __name__ == "__main__":
    monitor = MCPMonitor()
    
    try:
        asyncio.run(monitor.monitor())
    except KeyboardInterrupt:
        print("\n👋 Monitor stopped")
