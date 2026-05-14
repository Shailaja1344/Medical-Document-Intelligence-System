"""
app.py — FastAPI application entry point for Medical Document Intelligence System
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.api.routes import router
from src.utils.config import API_TITLE, API_VERSION, API_DESCRIPTION
from src.utils.logger import log


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle events."""
    log.info(f"Starting {API_TITLE} v{API_VERSION}")
    log.info("API ready — visit http://localhost:8000/docs for Swagger UI")
    yield
    log.info("Shutting down Medical Document Intelligence API")


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return """
    <html>
      <head><title>Medical Document Intelligence API</title></head>
      <body style="font-family:sans-serif;max-width:700px;margin:60px auto;text-align:center">
        <h1>🏥 Medical Document Intelligence API</h1>
        <p>End-to-end AI system for healthcare document understanding.</p>
        <p>
          <a href="/docs" style="margin:10px;padding:10px 24px;background:#0066cc;color:white;
             border-radius:6px;text-decoration:none">📖 Swagger Docs</a>
          <a href="/redoc" style="margin:10px;padding:10px 24px;background:#444;color:white;
             border-radius:6px;text-decoration:none">📋 ReDoc</a>
        </p>
        <hr/>
        <small>Version: """ + API_VERSION + """</small>
      </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn
    from src.utils.config import API_HOST, API_PORT

    uvicorn.run(
        "app:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info",
    )
