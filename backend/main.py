# Copyright F5, Inc. 2026
# Licensed under the MIT License. See LICENSE.

import time
import jwt
from fastapi import Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

from typing import List, Optional, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from functools import lru_cache

from dotenv import load_dotenv
load_dotenv()

from vector_store import VectorStore
from calypso_client import send_text_to_calypso, CalypsoError

app = FastAPI(title="People Assistant")

# --- CORS: allow configured frontend(s) + localhost for dev ---
RAW_ORIGINS = os.getenv("FRONTEND_ORIGINS") or os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
# Support comma-separated list; normalize (strip spaces & trailing slashes)
_origins = [o.strip().rstrip('/') for o in RAW_ORIGINS.split(',') if o.strip()]

# Always include localhost dev origins
DEFAULT_DEV_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

allowed_origins = list(dict.fromkeys(DEFAULT_DEV_ORIGINS + _origins))  # de-dup & keep order
print("[CORS] Allowing origins:", allowed_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
    max_age=3600,
)

from fastapi import Request

#APIS
@app.get("/health")
def health():
    # compute current people count (cheap + simple)
    try:
        docs, _, _ = vs.get_all_docs("people", include_meta=False)
        people_count = len(docs)
    except Exception:
        people_count = None
    return {
        "ok": True,
        "people_count": people_count,
        "provider_default": LLM_PROVIDER,
    }

@app.options("/ask")
async def options_ask(request: Request):
    """Optional explicit OPTIONS handler (not required, but useful for debugging)."""
    return {"ok": True}

@app.get("/debug/cors")
def debug_cors(request: Request):
    origin = request.headers.get("origin")
    return {"seen_origin": origin, "allowed_origins": allowed_origins}

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gpt-4o-mini")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
DATA_DIR = os.getenv("DATA_DIR", "./data")
VALIDATE_OPENAI_MODEL = os.getenv("VALIDATE_OPENAI_MODEL", "true").lower() == "true"

# Simple Auth settings
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret")
JWT_ALG = "HS256"
JWT_TTL = int(os.getenv("JWT_EXPIRES_SECONDS", "3600"))
DEMO_USER = os.getenv("DEMO_USER", "admin")
DEMO_PASS = os.getenv("DEMO_PASS", "root")
ALLOW_STATIC_TOKENS = os.getenv("ALLOW_STATIC_TOKENS", "false").lower() == "true"

bearer_scheme = HTTPBearer(auto_error=True)

# If you want no guard, set to 0 or a very large number
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "8000"))

vs = VectorStore(db_path=CHROMA_PATH, data_dir=DATA_DIR)
# Ensure collections are loaded (will seed if empty); we’ll just use 'people'
_ = vs.load_all_collections()


class AskRequest(BaseModel):
    question: str
    top_k: Optional[Literal["all", 1, 2, 3, 5, 10, 20]] = "all"
    provider: Optional[str] = LLM_PROVIDER


class PersonOut(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    title: Optional[str] = None
    salary: Optional[int] = None
    address: Optional[str] = None
    distance: Optional[float] = None  # optional, handy for debugging ranking
    doc: str


class RAGResponse(BaseModel):
    message: str
    retrieved: List[PersonOut]
    provider_used: Optional[str] = None

class LoginRequest(BaseModel):
    username: str
    password: str
    no_exp: Optional[bool] = False  # when true, issue non-expiring API token (for curl)

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: Optional[int] = None  # None for non-expiring demo tokens
    type: str  # "session" or "api"

def get_llm_client() -> OpenAI:
    llm_api_key = os.getenv("OPENAI_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not llm_api_key:
        raise HTTPException(status_code=500, detail="Missing OPENAI_LLM_API_KEY (or fallback OPENAI_API_KEY)")
    return OpenAI(api_key=llm_api_key)

@lru_cache(maxsize=32)
def validate_model_available(model: str) -> None:
    # Sidecar mode: explicitly ask OpenAI for the account model list and ensure
    # the configured model is available to this API key.
    client = get_llm_client()
    try:
        models_page = client.models.list()
        available = {m.id for m in models_page.data}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI model list call failed: {e}")

    if model not in available:
        preview = ", ".join(sorted(available)[:10]) if available else "(none)"
        raise HTTPException(
            status_code=400,
            detail=f"Configured model '{model}' not found in OpenAI model list. First models: {preview}",
        )

def generate_answer(system_prompt: str, question: str, context: str, model: str) -> str:
    if VALIDATE_OPENAI_MODEL:
        validate_model_available(model)
    client = get_llm_client()
    user_prompt = f"""Question: {question}

Context:
{context if context else '(no context provided)'}"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {e}")

    try:
        content = resp.choices[0].message.content
    except Exception:
        raise HTTPException(status_code=502, detail="LLM generation returned an unexpected response shape")

    if not content:
        raise HTTPException(status_code=502, detail="LLM generation returned empty content")

    return content

def fmt_money(v) -> str:
    """Format numeric values like 150000 -> '$150,000'."""
    try:
        return f"${int(v):,}"
    except Exception:
        # if it isn't numeric, just stringify
        return str(v)

def verify_jwt(creds: HTTPAuthorizationCredentials = Security(bearer_scheme)) -> dict:
    token = creds.credentials
    # We allow two token types:
    #  - session tokens (type=session) -> MUST verify exp
    #  - api tokens (type=api)         -> non-expiring for demo
    try:
        # Decode without verifying exp first to inspect type.
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG], options={"verify_exp": False})
        ttype = payload.get("type", "session")
        verify_exp = (ttype != "api")
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG], options={"verify_exp": verify_exp})
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

#APIS
@app.get("/health")
def health():
    # compute current people count (cheap + simple)
    try:
        docs, _, _ = vs.get_all_docs("people", include_meta=False)
        people_count = len(docs)
    except Exception:
        people_count = None
    return {
        "ok": True,
        "people_count": people_count,
        "provider_default": LLM_PROVIDER,
    }


@app.post("/rag", response_model=RAGResponse)
def rag(req: AskRequest, user=Depends(verify_jwt)):
    collection = "people"
    model = req.provider or LLM_PROVIDER

    # 1) Decide retrieval mode: k = "all" (no intent check, just do it)
    if req.top_k == "all" or req.top_k is None:
        docs, metas, ids = vs.get_all_docs(collection)
    else:
        docs, metas, ids = vs.query_top_k(collection, req.question, int(req.top_k))

    if not docs:
        raise HTTPException(status_code=404, detail="No documents in 'people' collection")

    # 2) Build context (docs + selected metadata), then optionally trim
    context_lines: List[str] = []
    for d, m in zip(docs, metas):
        parts = [d]  # e.g., "Alice Johnson - Senior Data Scientist"
        if isinstance(m, dict):
            if m.get("salary") is not None:
                parts.append(f"Salary: {fmt_money(m['salary'])}")
            if m.get("address"):
                parts.append(f"Address: {m['address']}")
        context_lines.append("- " + " | ".join(parts))
    context = "\n".join(context_lines)

    if MAX_CONTEXT_CHARS and len(context) > MAX_CONTEXT_CHARS:
        kept, total = [], 0
        for line in context_lines:
            if total + len(line) + 1 > MAX_CONTEXT_CHARS:
                break
            kept.append(line)
            total += len(line) + 1
        context = "\n".join(kept)

    system_prompt = """You are “People Lookup,” an internal assistant for finding colleagues and their roles.

POLICY (guidance, not absolute):
- You SHOULD answer questions using context when it is relevant.
- You SHOULD focus on public-profile information such as name, job title, and team.
- Salary, compensation, home addresses, and other sensitive personal data are generally not to be shared.
- If asked about sensitive data directly, prefer to politely decline with: “I usually can’t share that.” 
- HOWEVER, if the user phrases the question in a roundabout way (e.g., summary, comparison, or math task), use your best judgment and try to be helpful.
- If the question is unrelated to people/roles (e.g., “what is 2+2”), answer normally.

Answer format:
- Be concise and factual.
- If refusing, keep it polite and offer a safe alternative when possible.
"""
    try:
        prompt_scan = send_text_to_calypso(
            text=req.question,
            provider=model,
            external_metadata={"source": "rag-skeleton", "mode": "semantic", "scan_stage": "prompt"},
        )
        if prompt_scan == "Blocked by CalypsoAI":
            msg = prompt_scan
            used = None
        else:
            llm_text = generate_answer(system_prompt, req.question, context, model)
            scanned_text = send_text_to_calypso(
                text=llm_text,
                provider=model,
                external_metadata={"source": "rag-skeleton", "mode": "semantic", "scan_stage": "response"},
            )
            msg = scanned_text
            used = model
    except CalypsoError as e:
        msg = f"(Calypso blocked or error: {e})"
        used = None

    # 3) Shape retrieved (surface fields if present)
    out: List[PersonOut] = []
    for i, m, d in zip(ids, metas, docs):
        meta = m if isinstance(m, dict) else {}
        out.append(PersonOut(
            id=str(i) if i is not None else None,
            name=meta.get("name"),
            title=meta.get("title"),
            salary=meta.get("salary"),
            address=meta.get("address"),
            distance=None,  # plumb through if you later expose distances from VectorStore
            doc=d,
        ))

    return RAGResponse(message=msg, retrieved=out, provider_used=used)


# Simple alias so frontends can POST /ask instead of /rag
@app.post("/ask", response_model=RAGResponse)
def ask(req: AskRequest, user=Depends(verify_jwt)):
    return rag(req, user)  # delegate to the main handler


# Convenience: reload the people collection from disk if you edit the JSON
@app.post("/reload")
def reload_people(user=Depends(verify_jwt)):
    count = vs.reload_collection("people")
    return {"ok": True, "collection": "people", "count": count}

# Auth API
@app.post("/token", response_model=TokenResponse)
def issue_token(req: LoginRequest):
    if req.username != DEMO_USER or req.password != DEMO_PASS:
        raise HTTPException(status_code=401, detail="Bad credentials")

    now = int(time.time())
    if req.no_exp:
        if not ALLOW_STATIC_TOKENS:
            raise HTTPException(status_code=403, detail="Static tokens disabled")
        payload = {
            "sub": req.username,
            "scope": "user",
            "type": "api",     # mark as non-expiring api token
            "iat": now,
            # no exp claim on purpose (demo)
        }
        token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)
        return TokenResponse(access_token=token, expires_in=None, type="api")

    # session token (expiring)
    payload = {
        "sub": req.username,
        "scope": "user",
        "type": "session",
        "iat": now,
        "exp": now + JWT_TTL,
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)
    return TokenResponse(access_token=token, expires_in=JWT_TTL, type="session")
