"""
This module contains the Cli client for the MCP servers.
"""
import asyncio
import os
import sys
import traceback
from datetime import datetime
from typing import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
from langgraph.graph.graph import CompiledGraph

from mcp_client.base import (
    load_server_config,
    create_server_parameters,
    convert_mcp_to_langchain_tools,
    create_agent_executor
)


async def list_tools() -> None:
    """List available tools from the server."""
    server_config = load_server_config()
    server_params = create_server_parameters(server_config)
    langchain_tools = await convert_mcp_to_langchain_tools(server_params)

    for tool in langchain_tools:
        print(f"{tool.name}")


async def handle_chat_mode():
    """Handle chat mode for the LangChain agent."""
    print("\nInitializing chat mode...")
    agent_executor_cli = await create_agent_executor("cli")
    print("\nInitialized chat mode...")

    # Maintain a chat history of messages
    chat_history = []

    # Start the chat loop
    while True:
        try:
            user_message = input("\nYou: ").strip()
            if user_message.lower() in ["exit", "quit"]:
                print("Exiting chat mode.")
                break
            if user_message.lower() in ["clear", "cls"]:
                os.system("cls" if sys.platform == "win32" else "clear")
                chat_history = []
                continue
            all_messages = []
            # Append the chat history to all messages
            all_messages.extend(chat_history)
            all_messages = [HumanMessage(content=user_message)]
            input_messages = {
                "messages": all_messages,
                "today_datetime": datetime.now().isoformat(),
            }
            # Query the assistant and get a fully formed response
            assistant_response = await query_response(input_messages, agent_executor_cli)

            # Append the assistant's response to the history
            chat_history.append(AIMessage(content=assistant_response))
        except Exception as e:
            error_trace = traceback.format_exc()
            print(error_trace)
            print(f"\nError processing message: {e}")
            continue


async def query_response(input_messages: TypedDict, agent_executor: CompiledGraph) -> str:
    """Query the assistant and get a fully formed response."""
    collected_response = []

    async for chunk in agent_executor.astream(
            input_messages,
            stream_mode=["messages", "values"]
    ):
        # Process the chunk and append the response to the collected response
        process_chunk(chunk)
        if isinstance(chunk, dict) and "messages" in chunk:
            collected_response.append(chunk["messages"][-1].content)

    print("")  # Ensure a newline after the conversation ends
    return "".join(collected_response)


def process_chunk(chunk):
    """Process the chunk and print the response."""
    if isinstance(chunk, tuple) and chunk[0] == "messages":
        process_message_chunk(chunk[1][0])
    elif isinstance(chunk, dict) and "messages" in chunk:
        process_final_value_chunk()
    elif isinstance(chunk, tuple) and chunk[0] == "values":
        process_tool_calls(chunk[1]['messages'][-1])


def process_message_chunk(message_chunk):
    """Process the message chunk and print the content."""
    if isinstance(message_chunk, AIMessageChunk):
        content = message_chunk.content  # Get the content of the message chunk
        if isinstance(content, list):
            extracted_text = ''.join(item['text'] for item in content if 'text' in item)
            print(extracted_text, end="", flush=True)  # Print message content incrementally
        else:
            print(content, end="", flush=True)


def process_final_value_chunk():
    """Process the final value chunk and print the content."""
    print("\n", flush=True)  # Ensure a newline after complete message


def process_tool_calls(message):
    """Process the tool calls and print the results."""
    if isinstance(message, AIMessage) and message.tool_calls:
        message.pretty_print()  # Format and print tool call results


async def interactive_mode():
    """Run the CLI in interactive mode."""
    print("\nWelcome to the Interactive MCP Command-Line Tool")
    print("Type 'help' for available commands or 'chat' to start chat or 'quit' to exit")

    while True:
        try:
            command = input(">>> ").strip()  # Get user input
            if not command:
                continue
            should_continue = await handle_command(command)  # Handle the command
            if not should_continue:
                return
        except KeyboardInterrupt:
            print("\nUse 'quit' or 'exit' to close the program")
        except EOFError:
            break
        except Exception as e:
            print(f"\nError: {e}")


async def handle_command(command: str):
    """ Handle specific commands dynamically."""
    try:
        if command == "list-tools":
            print("\nFetching Tools List...\n")
            # Implement list-tools logic here
            await list_tools()
        elif command == "chat":
            print("\nEntering chat mode...")
            await handle_chat_mode()
            # Implement chat mode logic here
        elif command in ["quit", "exit"]:
            print("\nGoodbye!")
            return False
        elif command == "clear":
            if sys.platform == "win32":
                os.system("cls")
            else:
                os.system("clear")
        elif command == "help":
            print("\nAvailable commands:")
            print("  list-tools    - Display available tools")
            print("  chat          - Enter chat mode")
            print("  clear         - Clear the screen")
            print("  help          - Show this help message")
            print("  quit/exit     - Exit the program")
        else:
            print(f"\nUnknown command: {command}")
            print("Type 'help' for available commands")
    except Exception as e:
        print(f"\nError executing command: {e}")

    return True


def main() -> None:
    """ Entry point for the script."""


asyncio.run(interactive_mode())  # Run the main asynchronous function

if __name__ == "__main__":
    main()  # Execute the main function when script is run directly
