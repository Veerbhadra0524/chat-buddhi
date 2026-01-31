# AI Chat

A modern, minimal chatbot with conversation context memory, inspired by ChatGPT and Perplexity.

## Features

- **Modern UI** - Clean ChatGPT/Perplexity-style interface
- **Conversation Memory** - Context preserved for follow-up questions
- **Multiple Models** - Groq (Llama 3) and Google Gemini
- **Streaming** - Real-time typewriter effect
- **Auto Fallback** - Graceful degradation on API failures

## How Memory Works

The chatbot uses a 2-call pattern:
1. **Response Call** - Send user prompt + current memory to the selected model
2. **Memory Update Call** - Use HuggingFace (free) to extract key context from the exchange

This allows natural follow-up questions like:
- "What's the capital of France?" → "Paris"
- "What's its population?" → (understands "its" refers to Paris)

## Project Structure

```
chatbot/
├── app.py          # FastAPI backend
├── models.py       # Model API adapters + memory
├── index.html      # Modern chat UI
└── README.md       # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install fastapi uvicorn httpx python-dotenv
```

### 2. Configure API Keys

Create a `.env` file in the chatbot folder:

```env
# Required: At least one main model
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key

# Optional: For conversation memory (falls back to simple if missing)
HF_API_KEY=your_huggingface_api_key
```

### Getting API Keys (Free)

| Provider | URL | Free Tier |
|----------|-----|-----------|
| Google Gemini | [aistudio.google.com](https://aistudio.google.com/app/apikey) | Free |
| Groq | [console.groq.com](https://console.groq.com) | Free |
| HuggingFace | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Free |

### 3. Run

```bash
cd d:\Veerbhadra\Joshithasri\chatbot
python -m uvicorn app:app --reload --port 8000
```

### 4. Open

Navigate to: **http://localhost:8000**

## Usage

1. Type your message and press Enter
2. Ask follow-up questions naturally
3. Click "New" to start a fresh conversation
4. Switch models anytime with the dropdown
5. Adjust temperature for creativity vs precision

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serve frontend |
| GET | `/api/models` | List models |
| POST | `/api/chat` | Chat (non-streaming) |
| POST | `/api/chat/stream` | Chat (streaming SSE) |
