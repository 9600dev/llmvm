import sys
import asyncio
from typing import Optional, cast
from urllib.parse import urlparse
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import click
import rich

sys.path.append('..')

from llmvm.client.client import get_executor, llm
from llmvm.common.helpers import Helpers
from llmvm.common.object_transformers import ObjectTransformers
from llmvm.common.objects import (Assistant, AstNode, Content, Executor,
                                  MarkdownContent, Message, TextContent, User)
from llmvm.common.pdf import Pdf
from llmvm.server.tools.chrome import ChromeHelpers
from llmvm.server.tools.webhelpers import WebHelpers
from anthropic import Anthropic


async def stream_handler(node: AstNode):
    pass


class MCPClient():
    def __init__(self, executor_name: str, model: str):
        self.executor = get_executor(executor_name, model, '')
        self.model = model
        self.exit_stack = AsyncExitStack()
        self.session: ClientSession
        self.anthropic = Anthropic()

    async def connect_to_server(self, server_script_path: str):
        command = "python"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])


    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        # Initial Claude API call
        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )

        # Process response and handle tool calls
        tool_results = []
        final_text = []

        assistant_message_content = []
        for content in response.content:
            if content.type == 'text':
                final_text.append(content.text)
                assistant_message_content.append(content)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input

                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                tool_results.append({"call": tool_name, "result": result})
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                assistant_message_content.append(content)
                messages.append({
                    "role": "assistant",
                    "content": assistant_message_content
                })
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": result.content
                        }
                    ]
                })

                # Get next response from Claude
                response = self.anthropic.messages.create(
                    model="claude-3-7-sonnet-latest",
                    max_tokens=1000,
                    messages=messages,
                    tools=available_tools
                )

                final_text.append(response.content[0].text)

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def run(server_name: str, executor_name: str, model: str):
    client = MCPClient(executor_name=executor_name, model=model)
    try:
        await client.connect_to_server(server_name)
        await client.chat_loop()
    finally:
        await client.cleanup()


@click.command()
@click.argument('server_name', type=str, required=True)
@click.option('--executor_name', '-e', default='anthropic', required=True)
@click.option('--model', '-m', default='', required=False)
def main(
    server_name: str,
    executor_name: str,
    model: str,
):
    asyncio.run(run(server_name, executor_name, model))

if __name__ == '__main__':
    main()

