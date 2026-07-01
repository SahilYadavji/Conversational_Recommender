"""
app.py

FastAPI application for the SHL Assessment Recommender.

Responsibilities
----------------
- Configure FastAPI
- Configure CORS
- Register API routes
- Handle exceptions
- Expose health endpoints
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from chatbot import chat
from models import ChatRequest, ChatResponse

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------

app = FastAPI(
    title="SHL Assessment Recommender API",
    description="""
An AI-powered recommendation engine that suggests the most relevant
SHL assessments using semantic search (FAISS) and Gemini.

Features:
- Semantic search over SHL catalog
- Retrieval-Augmented Generation (RAG)
- Grounded recommendations
- Multi-turn conversations
""",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Logs request processing time.
    """

    start = time.perf_counter()

    logger.info(
        "Incoming %s %s",
        request.method,
        request.url.path,
    )

    response = await call_next(request)

    duration = time.perf_counter() - start

    logger.info(
        "%s completed in %.3f sec",
        request.url.path,
        duration,
    )

    return response


# ---------------------------------------------------------------------
# Exception Handlers
# ---------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception,
):
    logger.exception("Unhandled exception")

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error."
        },
    )


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@app.get(
    "/",
    tags=["System"],
)
async def root():
    """
    Root endpoint.
    """

    return {
        "service": "SHL Assessment Recommender API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get(
    "/health",
    tags=["System"],
)
async def health():
    """
    Health check endpoint.
    """

    return {
        "status": "healthy"
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Chat with the SHL recommendation assistant",
)
async def chat_api(request: ChatRequest):
    """
    Main chatbot endpoint.
    """

    try:

        messages = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in request.messages
        ]

        response = chat(messages)

        return ChatResponse(**response)

    except ValueError as exc:

        logger.warning(str(exc))

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception:

        logger.exception("Chat endpoint failed.")

        raise HTTPException(
            status_code=500,
            detail="Unable to process request.",
        )


# ---------------------------------------------------------------------
# Startup / Shutdown Events
# ---------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    logger.info("=" * 60)
    logger.info("SHL Assessment Recommender started.")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown():
    logger.info("Application shutdown complete.")