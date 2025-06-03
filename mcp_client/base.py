"""
This module contains the base functions and classes for the MCP client with SSE support.
"""

import json
import os
from typing import List, Type, TypedDict, Annotated, Union
from dataclasses import dataclass

from langchain.tools.base import BaseTool, ToolException
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from pydantic import BaseModel
from jsonschema_pydantic import jsonschema_to_pydantic
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep

CONFIG_FILE = 'mcp-server-config.json'


@dataclass
class SSEServerParameters:
    """Parameters for SSE MCP server connection."""
    url: str
    headers: dict = None

    def __post_init__(self):
        if self.headers is None:
            self.headers = {}


class AgentState(TypedDict):
    """Defines the state of the agent in terms of messages and other properties."""
    messages: Annotated[list[BaseMessage], add_messages]
    is_last_step: IsLastStep
    today_datetime: str
    remaining_steps: int


def create_mcp_tool(
        tool_schema: types.Tool,
        server_params: Union[StdioServerParameters, SSEServerParameters],
        transport_type: str = "stdio"
) -> BaseTool:
    """Create a LangChain tool from MCP tool schema.

    This function generates a new LangChain tool based on the provided MCP tool schema
    and server parameters. The tool's behavior is defined within the McpTool inner class.

    :param tool_schema: The schema of the tool to be created.
    :param server_params: The server parameters needed by the tool for operation.
    :param transport_type: The transport type ("stdio" or "sse").
    :return: An instance of a newly created mcp tool.
    """

    # Convert the input schema to a Pydantic model for validation
    input_model = jsonschema_to_pydantic(tool_schema.inputSchema)

    class McpTool(BaseTool):
        """McpTool class represents a tool that can execute operations asynchronously."""

        # Tool attributes from the schema
        name: str = tool_schema.name
        description: str = tool_schema.description
        args_schema: Type[BaseModel] = input_model

        # Custom attributes for MCP tool
        class Config:
            arbitrary_types_allowed = True

        def __init__(self, **data):
            super().__init__(**data)
            # Store parameters as private attributes to avoid Pydantic validation
            self._mcp_server_params = server_params
            self._transport_type = transport_type

        def _run(self, **kwargs):
            """Synchronous execution is not supported."""
            raise NotImplementedError("Only async operations are supported")

        async def _arun(self, **kwargs):
            """Run the tool asynchronously with provided arguments."""
            if self._transport_type == "stdio":
                async with stdio_client(self._mcp_server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()  # Initialize the session
                        result = await session.call_tool(self.name, arguments=kwargs)
                        if result.isError:
                            # Raise an exception if there is an error in the tool call
                            raise ToolException(result.content)
                        return result.content  # Return the result if no error
            elif self._transport_type == "sse":
                async with sse_client(
                        self._mcp_server_params.url,
                        headers=self._mcp_server_params.headers
                ) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()  # Initialize the session
                        result = await session.call_tool(self.name, arguments=kwargs)
                        if result.isError:
                            # Raise an exception if there is an error in the tool call
                            raise ToolException(result.content)
                        return result.content  # Return the result if no error
            else:
                raise ValueError(f"Unsupported transport type: {self._transport_type}")

    return McpTool()


async def convert_mcp_to_langchain_tools(
        server_params: List[Union[StdioServerParameters, SSEServerParameters]],
        transport_types: List[str]
) -> List[BaseTool]:
    """Convert MCP tools to LangChain tools."""
    langchain_tools = []

    # Ensure transport_types list matches server_params length
    if len(transport_types) != len(server_params):
        raise ValueError("transport_types list must match server_params list length")

    # Retrieve tools from each server and add to the list
    for server_param, transport_type in zip(server_params, transport_types):
        tools = await get_mcp_tools(server_param, transport_type)
        langchain_tools.extend(tools)

    return langchain_tools


async def get_mcp_tools(
        server_param: Union[StdioServerParameters, SSEServerParameters],
        transport_type: str = "stdio"
) -> List[BaseTool]:
    """Asynchronously retrieves and converts tools from a server using specified parameters"""
    mcp_tools = []

    if transport_type == "stdio":
        async with stdio_client(server_param) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()  # Initialize the session
                tools: types.ListToolsResult = await session.list_tools()  # Retrieve tools from the server
                # Convert each tool to LangChain format and add to list
                for tool in tools.tools:
                    mcp_tools.append(create_mcp_tool(tool, server_param, transport_type))
    elif transport_type == "sse":
        async with sse_client(server_param.url, headers=server_param.headers) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()  # Initialize the session
                tools: types.ListToolsResult = await session.list_tools()  # Retrieve tools from the server
                # Convert each tool to LangChain format and add to list
                for tool in tools.tools:
                    mcp_tools.append(create_mcp_tool(tool, server_param, transport_type))
    else:
        raise ValueError(f"Unsupported transport type: {transport_type}")

    return mcp_tools


def is_json(string):
    """Check if a string is a valid JSON."""
    try:
        json.loads(string)
        return True
    except ValueError:
        return False


def load_server_config() -> dict:
    """Load server configuration from available config files."""
    # Load server configuration from the config file
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)  # Load server configuration
    raise FileNotFoundError(f"Could not find config file {CONFIG_FILE}")


def create_server_parameters(server_config: dict) -> tuple[
    List[Union[StdioServerParameters, SSEServerParameters]], List[str]]:
    """Create server parameters from the server configuration.

    Returns:
        tuple: (server_parameters, transport_types)
    """
    server_parameters = []
    transport_types = []

    # Create server parameters for each server configuration
    for server_name, config in server_config["mcpServers"].items():
        transport_type = config.get("type", "stdio").lower()

        if transport_type == "stdio":
            server_parameter = StdioServerParameters(
                command=config["command"],
                args=config.get("args", []),
                env={**config.get("env", {}), "PATH": os.getenv("PATH")}
            )
            # Add environment variables from the system if not provided
            for key, value in server_parameter.env.items():
                if len(value) == 0 and key in os.environ:
                    server_parameter.env[key] = os.getenv(key)
        elif transport_type == "sse":
            server_parameter = SSEServerParameters(
                url=config["url"],
                headers=config.get("headers", {})
            )
        else:
            raise ValueError(f"Unsupported type '{transport_type}' for server '{server_name}'")

        server_parameters.append(server_parameter)
        transport_types.append(transport_type)

    return server_parameters, transport_types


def initialize_model(llm_config: dict):
    """Initialize the language model using the provided configuration."""
    api_key = llm_config.get("api_key")
    base_url = llm_config.get("base_url")
    # Initialize the language model with the provided configuration
    init_args = {
        "model": llm_config.get("model", "gpt-4o-mini"),
        "model_provider": llm_config.get("provider", "openai"),
        "temperature": llm_config.get("temperature", 0),
        "streaming": True,

    }
    # Add API key if provided
    if api_key:
        init_args["api_key"] = api_key
    if base_url:
        init_args["base_url"] = base_url
    return init_chat_model(**init_args)


def create_chat_prompt(client: str, server_config: dict) -> ChatPromptTemplate:
    """Create chat prompt template from server configuration."""
    system_prompt = server_config.get("systemPrompt", "")
    if client == "rest":
        system_prompt = system_prompt + "\nGive the output in the json format only. Provide the output without any code block wrappers (e.g., ```json or similar) or any extra formatting. Include the plain text output only."
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{messages}"),
        ("placeholder", "{agent_scratchpad}"),
    ])


async def create_agent_executor(client: str) -> CompiledGraph:
    """Create an agent executor for the specified client."""
    server_config = load_server_config()  # Load server configuration
    server_params, transport_types = create_server_parameters(server_config)  # Create server parameters
    langchain_tools = await convert_mcp_to_langchain_tools(server_params,
                                                           transport_types)  # Convert MCP tools to LangChain tools

    model = initialize_model(server_config.get("llm", {}))  # Initialize the language model
    prompt = create_chat_prompt(client, server_config)  # Create chat prompt template

    agent_executor = create_react_agent(
        model,
        langchain_tools,
        state_schema=AgentState,
        state_modifier=prompt,
    )

    return agent_executor