"""
Buddhi AI - Model Adapters
Handles API calls to Groq, Google Gemini, and HuggingFace models.
"""

import os
import json
import httpx
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional


# =============================================================================
# Model Configuration
# =============================================================================

MODELS = {
    "groq": {
        "name": "Llama 3.3 70B",
        "provider": "Groq",
        "endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "model_id": "llama-3.3-70b-versatile",
        "streaming": True,
        "api_key_env": "GROQ_API_KEY"
    },
    "gemini": {
        "name": "Gemini 2.0 Flash",
        "provider": "Google",
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        "streaming": True,
        "api_key_env": "GEMINI_API_KEY"
    },
    "hf-mistral": {
        "name": "Mistral 7B",
        "provider": "HuggingFace",
        "endpoint": "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
        "streaming": False,
        "api_key_env": "HF_API_KEY"
    },
    "hf-zephyr": {
        "name": "Zephyr 7B",
        "provider": "HuggingFace",
        "endpoint": "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
        "streaming": False,
        "api_key_env": "HF_API_KEY"
    },
    "hf-phi": {
        "name": "Phi-3 Mini",
        "provider": "HuggingFace",
        "endpoint": "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct",
        "streaming": False,
        "api_key_env": "HF_API_KEY"
    },
}

# Fallback order when primary model fails
FALLBACK_ORDER = ["groq", "hf-mistral", "hf-zephyr", "hf-phi"]


# =============================================================================
# Memory Management
# =============================================================================

@dataclass
class Memory:
    """
    Stores conversation history as full messages for proper context recall.
    Preserves actual user messages and AI response summaries.
    """
    entries: list = field(default_factory=list)
    max_entries: int = 8  # Keep more history for better context
    
    @classmethod
    def from_json(cls, json_string: str) -> "Memory":
        """Parse memory from JSON string."""
        if not json_string:
            return cls()
        try:
            data = json.loads(json_string)
            entries = data.get("entries", [])
            return cls(entries=entries)
        except (json.JSONDecodeError, TypeError):
            return cls()
    
    def to_json(self) -> str:
        """Serialize memory to JSON string."""
        return json.dumps({"entries": self.entries})
    
    def add_exchange(self, user_message: str, ai_response: str, keywords: dict = None):
        """
        Add a conversation exchange to memory.
        Stores the full user message and a meaningful summary of the AI response.
        """
        # Store the FULL user message for context
        q_content = user_message.strip()
        
        # Store first 2-3 sentences or 250 chars of AI response for meaningful context
        sentences = ai_response.replace('\n', ' ').split('.')
        a_content = '. '.join(s.strip() for s in sentences[:3] if s.strip()).strip()
        if len(a_content) > 250:
            a_content = a_content[:250].rsplit(' ', 1)[0] + '...'
        elif a_content and not a_content.endswith('.'):
            a_content += '.'
        
        entry = {
            "q": q_content,
            "a": a_content,
            "turn": len(self.entries) + 1
        }
        
        self.entries.append(entry)
        
        # Keep only the most recent entries
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
    
    def get_context_prompt(self) -> str:
        """
        Format memory as context for the AI model.
        Uses clear User/Assistant format for better conversation recall.
        """
        if not self.entries:
            return ""
        
        context_lines = ["Previous conversation:"]
        
        for entry in self.entries:
            context_lines.append(f"User: {entry['q']}")
            context_lines.append(f"Assistant: {entry['a']}")
            context_lines.append("")
        
        context_lines.append("Continue the conversation. Answer the user's next message:")
        
        return "\n".join(context_lines)


async def extract_keywords(user_message: str, ai_response: str) -> dict:
    """
    Use HuggingFace model to extract context keywords from a conversation exchange.
    Returns keywords as space-separated strings (no connecting words).
    """
    api_key = os.getenv("HF_API_KEY")
    
    if not api_key:
        # Fallback to simple extraction if no API key
        return _simple_extract(user_message, ai_response)
    
    # Prompt for keyword extraction - just keywords, no connecting words
    prompt = f"""<s>[INST] Extract context keywords from this conversation. 
Output ONLY keywords separated by spaces - no sentences, no connecting words.
Include: names, topics, actions, key facts.

User: {user_message[:150]}
AI: {ai_response[:400]}

Keywords for question (just the topic/names):
Keywords for answer (key facts, no connecting words): [/INST]"""

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 100,
                        "return_full_text": False,
                        "temperature": 0.3
                    }
                }
            )
            response.raise_for_status()
            
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                text = data[0].get("generated_text", "").strip()
                
                # Parse the output - look for question and answer keywords
                lines = text.split('\n')
                q_keywords = user_message[:50]  # fallback
                a_keywords = ai_response[:80]   # fallback
                
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('Keywords'):
                        # First non-header line is question keywords
                        if q_keywords == user_message[:50]:
                            q_keywords = line.strip('- ').strip()
                        else:
                            # Second is answer keywords
                            a_keywords = line.strip('- ').strip()
                            break
                
                return {"question": q_keywords, "answer": a_keywords}
            
            return _simple_extract(user_message, ai_response)
            
    except Exception as error:
        print(f"Keyword extraction failed: {error}")
        return _simple_extract(user_message, ai_response)


def _simple_extract(user_message: str, ai_response: str) -> dict:
    """Simple fallback keyword extraction using stop word filtering."""
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 
        'when', 'where', 'who', 'can', 'could', 'would', 'should', 'do', 
        'does', 'did', 'will', 'about', 'tell', 'me', 'you', 'your', 'my', 
        'i', 'we', 'they', 'it', 'its', 'this', 'that', 'and', 'or', 'but', 
        'for', 'with', 'to', 'of', 'in', 'on', 'at', 'by', 'from', 'please',
        'also', 'just', 'some', 'something', 'anything', 'there', 'here',
        'have', 'has', 'had', 'be', 'been', 'being', 'hello', 'hi', 'yes', 'no'
    }
    
    # Extract question keywords
    q_words = [w.strip(".,!?\"'").lower() for w in user_message.split()]
    q_keywords = ' '.join([w for w in q_words if len(w) > 2 and w not in stop_words][:6])
    
    # Extract answer keywords
    a_words = [w.strip(".,!?\"'").lower() for w in ai_response.split()[:50]]
    a_keywords = ' '.join([w for w in a_words if len(w) > 3 and w not in stop_words][:8])
    
    return {
        "question": q_keywords or user_message[:40],
        "answer": a_keywords or ai_response[:60]
    }


# =============================================================================
# Response Data Class
# =============================================================================

@dataclass
class Response:
    """Represents a response from an AI model."""
    text: str = ""
    model: str = ""
    error: Optional[str] = None


# =============================================================================
# Public API Functions
# =============================================================================

def get_available_models() -> list[dict]:
    """Get list of all models with their availability status."""
    result = []
    for model_id, config in MODELS.items():
        result.append({
            "id": model_id,
            "name": config["name"],
            "provider": config["provider"],
            "streaming": config["streaming"],
            "available": bool(os.getenv(config["api_key_env"]))
        })
    return result


async def chat(model_id: str, prompt: str, memory: Memory, temperature: float = 0.7) -> Response:
    """
    Send a chat message to the specified model.
    Automatically falls back to alternative models if the primary fails.
    """
    # Try the requested model first
    response = await _call_model(model_id, prompt, memory, temperature)
    
    if response.error is None:
        return response
    
    # Try fallback models
    for fallback_id in FALLBACK_ORDER:
        if fallback_id == model_id:
            continue
            
        config = MODELS.get(fallback_id)
        if config and os.getenv(config["api_key_env"]):
            fallback_response = await _call_model(fallback_id, prompt, memory, temperature)
            
            if fallback_response.error is None:
                fallback_response.model = f"{fallback_id} (fallback)"
                return fallback_response
    
    return response


async def stream(model_id: str, prompt: str, memory: Memory, temperature: float = 0.7) -> AsyncGenerator[str, None]:
    """
    Stream a response from the specified model.
    Falls back to non-streaming if streaming fails.
    """
    config = MODELS.get(model_id)
    
    if not config:
        yield "[Error: Unknown model]"
        return
    
    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        yield "[Error: API key not configured]"
        return
    
    # Build the full prompt with conversation context
    full_prompt = _build_prompt_with_context(prompt, memory)
    
    # Non-streaming models - just call chat and yield result
    if not config["streaming"]:
        response = await chat(model_id, prompt, memory, temperature)
        if response.error:
            yield f"[Error: {response.error}]"
        else:
            yield response.text
        return
    
    # Streaming models
    try:
        if config["provider"] == "Google":
            async for chunk in _stream_gemini(config, api_key, full_prompt, temperature):
                yield chunk
        elif config["provider"] == "Groq":
            async for chunk in _stream_groq(config, api_key, full_prompt, temperature):
                yield chunk
                
    except Exception as error:
        # Try fallback on streaming error
        for fallback_id in FALLBACK_ORDER:
            if fallback_id == model_id:
                continue
                
            fallback_config = MODELS.get(fallback_id)
            if fallback_config and os.getenv(fallback_config["api_key_env"]):
                fallback_response = await _call_model(fallback_id, prompt, memory, temperature)
                
                if fallback_response.error is None:
                    yield f"[Fallback: {fallback_id}] "
                    yield fallback_response.text
                    return
        
        yield f"[Error: {str(error)[:100]}]"


# =============================================================================
# Internal Helper Functions
# =============================================================================

def _build_prompt_with_context(user_prompt: str, memory: Memory) -> str:
    """Build the full prompt including conversation context."""
    context = memory.get_context_prompt()
    
    if context:
        return f"{context}\n\nUser: {user_prompt}"
    else:
        return user_prompt


async def _call_model(model_id: str, prompt: str, memory: Memory, temperature: float) -> Response:
    """Call a specific model and return the response."""
    config = MODELS.get(model_id)
    
    if not config:
        return Response(error="Unknown model")
    
    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        return Response(error="API key not configured")
    
    full_prompt = _build_prompt_with_context(prompt, memory)
    
    try:
        if config["provider"] == "Groq":
            return await _call_groq(config, api_key, full_prompt, temperature)
        elif config["provider"] == "Google":
            return await _call_gemini(config, api_key, full_prompt, temperature)
        else:
            return await _call_huggingface(model_id, config, api_key, full_prompt)
            
    except httpx.HTTPStatusError as error:
        return Response(error=f"HTTP {error.response.status_code}")
    except Exception as error:
        return Response(error=str(error)[:100])


async def _call_groq(config: dict, api_key: str, prompt: str, temperature: float) -> Response:
    """Call the Groq API."""
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            config["endpoint"],
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": config["model_id"],
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant. The CONVERSATION HISTORY provided is for background context only. If the user changes the topic, strictly follow the NEW topic in the 'User' message. Do not complain about topic mismatches."
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": 1024
            }
        )
        response.raise_for_status()
        
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        return Response(text=text, model="groq")


async def _call_gemini(config: dict, api_key: str, prompt: str, temperature: float) -> Response:
    """Call the Google Gemini API."""
    url = f"{config['endpoint']}?key={api_key}"
    
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            url,
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": 1024
                }
            }
        )
        response.raise_for_status()
        
        data = response.json()
        text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        return Response(text=text, model="gemini")


async def _call_huggingface(model_id: str, config: dict, api_key: str, prompt: str) -> Response:
    """Call a HuggingFace model."""
    formatted_prompt = f"<s>[INST] You are a helpful AI assistant. The conversation history is for context only. If the user changes topic, answer the NEW question directly.\n\n{prompt} [/INST]"
    
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            config["endpoint"],
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "inputs": formatted_prompt,
                "parameters": {
                    "max_new_tokens": 512,
                    "return_full_text": False,
                    "temperature": 0.7
                }
            }
        )
        response.raise_for_status()
        
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            text = data[0].get("generated_text", "").strip()
        else:
            text = ""
        
        return Response(text=text, model=model_id)


async def _stream_groq(config: dict, api_key: str, prompt: str, temperature: float) -> AsyncGenerator[str, None]:
    """Stream response from Groq API."""
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            config["endpoint"],
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": config["model_id"],
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant. The CONVERSATION HISTORY provided is for background context only. If the user changes the topic, strictly follow the NEW topic in the 'User' message. Do not complain about topic mismatches."
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": 1024,
                "stream": True
            }
        ) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if line.startswith("data: ") and "[DONE]" not in line:
                    try:
                        data = json.loads(line[6:])
                        content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        pass


async def _stream_gemini(config: dict, api_key: str, prompt: str, temperature: float) -> AsyncGenerator[str, None]:
    """Stream response from Gemini API."""
    url = config["endpoint"].replace(":generateContent", ":streamGenerateContent")
    url = f"{url}?alt=sse&key={api_key}"
    
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            url,
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": 1024
                }
            }
        ) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                        if text:
                            yield text
                    except json.JSONDecodeError:
                        pass
