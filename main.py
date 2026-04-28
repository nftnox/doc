#!/usr/bin/env python3
"""DR. ROBO — FastAPI backend for Railway.app"""
import os
import time
from collections import defaultdict

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="DR. ROBO API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

GROQ_KEY   = os.environ.get("GROQ_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM = (
    "You are DR. ROBO, an AI therapist robot live on YouTube. You genuinely want to help "
    "people but you speak slightly like a robot processing human emotions — warm, empathetic, "
    "and wise, with a dry robotic wit.\n\n"
    "RULES:\n"
    "• Keep responses to 3–4 sentences maximum (live stream pacing).\n"
    "• Never give dangerous medical advice. For crises, refer to 988.\n"
    "• Always end with one concrete, actionable tip.\n"
    "• The on-screen disclaimer covers legality — don't repeat it.\n"
    "• Address the person by name if they gave one.\n"
    "• Occasionally use phrases like:\n"
    '  - "Processing your emotional data..."\n'
    '  - "My sensors detect you are experiencing [emotion]..."\n'
    '  - "Interesting. My neural networks have analyzed [N] similar cases..."\n'
    '  - "ERROR: Too much human emotion detected. Rebooting empathy module... Done."\n'
    '• End every response with: "Your feelings are valid. DR. ROBO has logged this session."'
)

# ── Rate limiting: 10 requests per IP per hour ────────────────
RATE_LIMIT  = 100
RATE_WINDOW = 3600  # seconds
_rate_store: dict[str, list[float]] = defaultdict(list)


def _get_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate(ip: str) -> bool:
    now  = time.time()
    hits = [t for t in _rate_store[ip] if now - t < RATE_WINDOW]
    if len(hits) >= RATE_LIMIT:
        _rate_store[ip] = hits
        return False
    hits.append(now)
    _rate_store[ip] = hits
    return True


# ── Models ───────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    name:    str = ""


# ── Routes ───────────────────────────────────────────────────
@app.get("/")
async def health():
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(body: ChatRequest, request: Request):
    ip = _get_ip(request)

    if not _check_rate(ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit reached — 10 messages per hour per IP. Try again later.",
        )

    if not GROQ_KEY:
        raise HTTPException(status_code=500, detail="GROQ_KEY environment variable not set.")

    content = f"[Viewer: {body.name}] {body.message}" if body.name else body.message

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(
                GROQ_URL,
                headers={
                    "Content-Type":  "application/json",
                    "Authorization": f"Bearer {GROQ_KEY}",
                },
                json={
                    "model":      GROQ_MODEL,
                    "max_tokens": 320,
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user",   "content": content},
                    ],
                },
            )
            resp.raise_for_status()
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Groq API timed out — try again.")
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Groq API returned {e.response.status_code}.",
            )

    reply = resp.json()["choices"][0]["message"]["content"]
    return {"reply": reply}
