"""
Buddhi AI - FastAPI Backend
Main application server handling chat requests and streaming responses.
"""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import models


# Load environment variables from .env file
load_dotenv()


# =============================================================================
# Application Setup
# =============================================================================

app = FastAPI(
    title="Buddhi AI",
    description="AI Chat with conversation memory",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Request/Response Models
# =============================================================================

class ChatRequest(BaseModel):
    """Request body for chat endpoints."""
    prompt: str
    model: str = "groq"
    temp: float = 0.7
    memory: str = ""  # JSON-serialized memory


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main HTML page."""
    html_path = Path(__file__).parent / "index.html"
    
    if html_path.exists():
        content = html_path.read_text(encoding="utf-8")
        return HTMLResponse(content=content)
    else:
        return HTMLResponse(
            content="<h1>Frontend not found</h1>",
            status_code=404
        )


@app.get("/api/models")
async def get_models():
    """Get list of available AI models."""
    available_models = models.get_available_models()
    return {"models": available_models}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Send a chat message and get a response.
    Non-streaming endpoint for simpler integrations.
    """
    # Validate input
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    # Parse memory from JSON
    memory = models.Memory.from_json(request.memory)
    
    # Get response from model
    response = await models.chat(
        model_id=request.model,
        prompt=request.prompt,
        memory=memory,
        temperature=request.temp
    )
    
    # Handle errors
    if response.error:
        return {
            "response": f"Error: {response.error}",
            "model": response.model or request.model,
            "memory": request.memory,
            "error": True
        }
    
    # Extract keywords using AI for smart context storage
    keywords = await models.extract_keywords(request.prompt, response.text)
    
    # Add this exchange to memory with extracted keywords
    memory.add_exchange(request.prompt, response.text, keywords)
    
    return {
        "response": response.text,
        "model": response.model,
        "memory": memory.to_json(),
        "error": False
    }


@app.post("/api/stream")
async def stream_chat(request: ChatRequest):
    """
    Stream a chat response using Server-Sent Events (SSE).
    Provides real-time token-by-token response.
    """
    # Validate input
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    # Parse memory from JSON
    memory = models.Memory.from_json(request.memory)
    
    # Collect chunks to build full response for memory
    collected_chunks = []
    
    async def generate_stream():
        """Generator function for SSE stream."""
        nonlocal collected_chunks
        
        # Stream response chunks
        async for chunk in models.stream(
            model_id=request.model,
            prompt=request.prompt,
            memory=memory,
            temperature=request.temp
        ):
            collected_chunks.append(chunk)
            
            # Send chunk as SSE data
            yield f"data: {json.dumps(chunk)}\n\n"
        
        # Build full response from chunks
        full_response = "".join(collected_chunks)
        
        # Update memory if we got a valid response
        if full_response and not full_response.startswith("[Error"):
            # Remove fallback prefix for memory
            clean_response = full_response
            if full_response.startswith("[Fallback:"):
                parts = full_response.split("] ", 1)
                if len(parts) > 1:
                    clean_response = parts[1]
            
            # Extract keywords using AI for smart context storage
            keywords = await models.extract_keywords(request.prompt, clean_response)
            
            # Add this exchange to memory with extracted keywords
            memory.add_exchange(request.prompt, clean_response, keywords)
        
        # Send memory update as separate event
        yield f"event: memory\ndata: {memory.to_json()}\n\n"
        
        # Signal end of stream
        yield 'data: "[DONE]"\n\n'
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )
