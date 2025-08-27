#!/usr/bin/env python3
"""
Talk Box React Chat Server

A FastAPI server that provides REST API endpoints for the Talk Box React chat component.
This server acts as a bridge between the React frontend and Talk Box's Python backend.
"""

import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Try to import Talk Box components
try:
    from talk_box import ChatBot

    TALK_BOX_AVAILABLE = True
except ImportError:
    TALK_BOX_AVAILABLE = False
    print("Warning: Talk Box not available. Running in demo mode.")


# Pydantic models for API requests/responses
class ChatBotConfigModel(BaseModel):
    name: Optional[str] = "Talk Box Assistant"
    description: Optional[str] = None
    model: Optional[str] = "gpt-4"
    temperature: Optional[float] = 0.7
    maxTokens: Optional[int] = 1000
    preset: Optional[str] = None
    persona: Optional[str] = None
    avoidTopics: Optional[List[str]] = None
    tools: Optional[List[str]] = None
    systemPrompt: Optional[str] = None


class ConversationCreateRequest(BaseModel):
    config: ChatBotConfigModel


class MessageRequest(BaseModel):
    content: str


class MessageResponse(BaseModel):
    id: str
    content: str
    role: str  # 'user' or 'assistant'
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class ConversationResponse(BaseModel):
    id: str
    messages: List[MessageResponse]
    metadata: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    talk_box_available: bool


# Initialize FastAPI app
app = FastAPI(
    title="Talk Box React Chat API",
    description="REST API for Talk Box React chat component",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for conversations (use database in production)
conversations: Dict[str, Dict[str, Any]] = {}

# Global bot configuration (set by ReactChatServer)
global_bot_config: Optional[Dict[str, Any]] = None


def create_message_response(
    content: str,
    role: str,
    message_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> MessageResponse:
    """Helper function to create a MessageResponse."""
    return MessageResponse(
        id=message_id or str(uuid.uuid4()),
        content=content,
        role=role,
        timestamp=datetime.now(),
        metadata=metadata or {},
    )


def setup_chatbot(config: ChatBotConfigModel) -> Optional["ChatBot"]:
    """Set up a Talk Box ChatBot from configuration."""
    if not TALK_BOX_AVAILABLE:
        return None

    try:
        bot = ChatBot(
            name=config.name,
            description=config.description,
        )

        # Configure the bot based on the provided config
        if config.model:
            bot = bot.model(config.model)
        if config.temperature is not None:
            bot = bot.temperature(config.temperature)
        if config.maxTokens:
            bot = bot.max_tokens(config.maxTokens)
        if config.preset:
            bot = bot.preset(config.preset)
        if config.persona:
            bot = bot.persona(config.persona)
        if config.avoidTopics:
            bot = bot.avoid(config.avoidTopics)
        if config.tools:
            bot = bot.tools(config.tools)
        if config.systemPrompt:
            bot = bot.system_prompt(config.systemPrompt)

        return bot
    except Exception as e:
        print(f"Error setting up chatbot: {e}")
        return None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy", timestamp=datetime.now(), talk_box_available=TALK_BOX_AVAILABLE
    )


@app.get("/config")
async def get_bot_config():
    """Get the current bot configuration."""
    if global_bot_config:
        # Return the stored configuration (already in correct format from ReactChatServer)
        return global_bot_config
    else:
        # Return default configuration in camelCase format for React app
        return {
            "name": "Talk Box Assistant",
            "description": "A helpful AI assistant powered by Talk Box and Chatlas",
            "model": "gpt-4",
            "temperature": 0.7,
            "maxTokens": 1000,
            "preset": None,  # Changed from "helpful_assistant" to None to reflect no preset set
            "persona": "a knowledgeable and friendly AI assistant",
            "avoidTopics": ["harmful content", "illegal activities"],
            "tools": ["web_search", "code_analysis", "file_operations"],
            "systemPrompt": "You are a helpful AI assistant powered by Talk Box and Chatlas.",
        }


@app.post("/config")
async def set_bot_config(config: Dict[str, Any]):
    """Set the bot configuration (called by ReactChatServer)."""
    global global_bot_config
    global_bot_config = config
    return {"message": "Bot configuration updated successfully"}


@app.post("/conversation", response_model=ConversationResponse)
async def create_conversation(request: ConversationCreateRequest):
    """Create a new conversation with a configured chatbot."""
    conversation_id = str(uuid.uuid4())

    # Set up the chatbot
    chatbot = setup_chatbot(request.config) if TALK_BOX_AVAILABLE else None

    # Store conversation data
    conversations[conversation_id] = {
        "id": conversation_id,
        "config": request.config.dict(),
        "chatbot": chatbot,
        "messages": [],
        "created_at": datetime.now(),
    }

    # Create initial response if there's an opening message in the preset
    messages = []
    if TALK_BOX_AVAILABLE and chatbot:
        # Check if the preset has an opening message
        try:
            system_prompt = chatbot.get_system_prompt()
            if "welcome" in system_prompt.lower() or "hello" in system_prompt.lower():
                # Add a greeting message
                greeting = create_message_response(
                    content=f"Hello! I'm {request.config.name or 'your Talk Box assistant'}. How can I help you today?",
                    role="assistant",
                    metadata={"type": "greeting"},
                )
                messages.append(greeting)
                conversations[conversation_id]["messages"] = [greeting]
        except Exception:
            pass

    return ConversationResponse(
        id=conversation_id, messages=messages, metadata={"created_at": datetime.now()}
    )


@app.post("/conversation/{conversation_id}/message", response_model=MessageResponse)
async def send_message(conversation_id: str, request: MessageRequest):
    """Send a message to a conversation and get the assistant's response."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conversation_data = conversations[conversation_id]
    chatbot = conversation_data.get("chatbot")

    # Create user message (we'll add it after getting the response)
    user_message = create_message_response(content=request.content, role="user")

    # Generate assistant response
    if TALK_BOX_AVAILABLE and chatbot:
        try:
            # Import required classes for conversation reconstruction
            from talk_box.conversation import Conversation

            # Reconstruct the conversation from stored messages (excluding current message)
            conversation = Conversation(conversation_id=conversation_id)

            # Add only previous messages to maintain context
            for msg in conversation_data["messages"]:
                conversation.add_message(
                    content=msg.content, role=msg.role, metadata=msg.metadata or {}
                )

            # Use Talk Box chat method with conversation history
            # This will add the current user message and generate the assistant response
            updated_conversation = chatbot.chat(request.content, conversation)

            # Extract the assistant's message from the updated conversation
            last_message = updated_conversation.get_last_message()
            assistant_content = last_message.content

            assistant_message = create_message_response(
                content=assistant_content,
                role="assistant",
                metadata={"model": conversation_data["config"].get("model", "unknown")},
            )

            # Now add both messages to our stored conversation
            conversation_data["messages"].append(user_message)
            conversation_data["messages"].append(assistant_message)

        except Exception as e:
            print(f"Error generating response: {e}")
            assistant_message = create_message_response(
                content=f"I apologize, but I encountered an error: {str(e)}",
                role="assistant",
                metadata={"error": True},
            )
            # Still add the user message even if there was an error
            conversation_data["messages"].append(user_message)
            conversation_data["messages"].append(assistant_message)
    else:
        # Demo mode response
        assistant_message = create_message_response(
            content=f"Echo (Demo Mode): {request.content}",
            role="assistant",
            metadata={"demo_mode": True},
        )
        # Add both messages in demo mode too
        conversation_data["messages"].append(user_message)
        conversation_data["messages"].append(assistant_message)

    return assistant_message


@app.get("/conversation/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str):
    """Get a conversation by ID."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conversation_data = conversations[conversation_id]

    return ConversationResponse(
        id=conversation_id,
        messages=conversation_data["messages"],
        metadata={
            "created_at": conversation_data["created_at"],
            "message_count": len(conversation_data["messages"]),
        },
    )


@app.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    del conversations[conversation_id]
    return {"message": "Conversation deleted successfully"}


@app.get("/conversations")
async def list_conversations():
    """List all conversations (for debugging/admin purposes)."""
    return {
        "conversations": [
            {
                "id": conv_id,
                "created_at": data["created_at"],
                "message_count": len(data["messages"]),
                "config": data["config"],
            }
            for conv_id, data in conversations.items()
        ]
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("chat_server:app", host="0.0.0.0", port=port, reload=True, log_level="info")
