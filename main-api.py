"""
OpenAI compatible API server for MCP clients.
"""
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import ToolMessage
from pydantic import BaseModel
import uvicorn

from mcp_client.base import (
    load_server_config,
    create_server_parameters,
    convert_mcp_to_langchain_tools,
    create_agent_executor
)
from langchain_core.messages import HumanMessage, AIMessage


# OpenAI API compatible models
class ChatMessage(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "mcp-agent"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int]


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "mcp-client"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# Global agent executor
agent_executor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the MCP agent on startup."""
    global agent_executor
    print("Initializing MCP agent...")
    try:
        agent_executor = await create_agent_executor("api")
        print("MCP agent initialized successfully")
    except Exception as e:
        print(f"Failed to initialize MCP agent: {e}")
        raise
    yield
    # Cleanup if needed
    agent_executor = None


app = FastAPI(
    title="MCP OpenAI Compatible API",
    description="OpenAI format API for MCP clients",
    version="1.0.0",
    lifespan=lifespan
)


def convert_messages_to_langchain(messages: List[ChatMessage]) -> List:
    """Convert OpenAI format messages to LangChain format."""
    langchain_messages = []

    for msg in messages:
        if msg.role == "user":
            langchain_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            langchain_messages.append(AIMessage(content=msg.content))
        # Skip system messages for now or handle them differently

    return langchain_messages


async def get_agent_response(messages: List[ChatMessage]) -> str:
    """Get response from the MCP agent."""
    if not agent_executor:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    langchain_messages = convert_messages_to_langchain(messages)

    input_data = {
        "messages": langchain_messages,
        "today_datetime": datetime.now().isoformat(),
    }

    collected_response = []

    async for chunk in agent_executor.astream(
            input_data,
            stream_mode=["messages", "values"]
    ):
        if isinstance(chunk, dict) and "messages" in chunk:
            if chunk["messages"]:
                last_message = chunk["messages"][-1]
                if hasattr(last_message, 'content'):
                    collected_response.append(last_message.content)

    return "".join(collected_response)


async def stream_agent_response(messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
    """Stream response from the MCP agent."""
    if not agent_executor:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    langchain_messages = convert_messages_to_langchain(messages)

    input_data = {
        "messages": langchain_messages,
        "today_datetime": datetime.now().isoformat(),
    }

    request_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    created = int(datetime.now().timestamp())

    async for chunk in agent_executor.astream(
            input_data,
            stream_mode=["messages", "values"]
    ):
        if isinstance(chunk, tuple) and chunk[0] == "messages":
            message_chunk = chunk[1][0]
            if hasattr(message_chunk, 'content'):
                content = message_chunk.content
                if content:
                    if isinstance(message_chunk, ToolMessage):
                        print(f"Tool Call ID: {message_chunk.tool_call_id}", "TOOL")
                        print(f"Tool Response: {message_chunk.content}", "TOOL")
                        print(f"Tool Name: {message_chunk.name}", "TOOL")
                        print(f"Tool ID: {message_chunk.id}", "TOOL")
                    else:
                        stream_chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            created=created,
                            model="mcp-agent",
                            choices=[ChatCompletionStreamChoice(
                                index=0,
                                delta={"content": content}
                            )]
                        )
                        yield f"data: {stream_chunk.model_dump_json()}\n\n"

    # Send final chunk
    final_chunk = ChatCompletionStreamResponse(
        id=request_id,
        created=created,
        model="mcp-agent",
        choices=[ChatCompletionStreamChoice(
            index=0,
            delta={},
            finish_reason="stop"
        )]
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/v1/models")
async def list_models() -> ModelsResponse:
    """List available models."""
    return ModelsResponse(
        data=[
            ModelInfo(
                id="mcp-agent",
                created=int(datetime.now().timestamp())
            )
        ]
    )


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion."""
    try:
        if request.stream:
            # Return streaming response
            return StreamingResponse(
                stream_agent_response(request.messages),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            # Return regular response
            response_content = await get_agent_response(request.messages)

            response = ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:29]}",
                created=int(datetime.now().timestamp()),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=response_content),
                        finish_reason="stop"
                    )
                ],
                usage={
                    "prompt_tokens": sum(len(msg.content.split()) for msg in request.messages),
                    "completion_tokens": len(response_content.split()),
                    "total_tokens": sum(len(msg.content.split()) for msg in request.messages) + len(
                        response_content.split())
                }
            )

            return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "agent_ready": agent_executor is not None}


@app.get("/v1/tools")
async def list_tools():
    """List available MCP tools."""
    try:
        server_config = load_server_config()
        server_params = create_server_parameters(server_config)
        langchain_tools = await convert_mcp_to_langchain_tools(server_params)

        tools = []
        for tool in langchain_tools:
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": getattr(tool, 'args_schema', {})
            })

        return {"tools": tools}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main-api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )