"""
FedShield API Server — Standalone Entrypoint

This is a convenience entrypoint that can be run directly with:
    cd backend
    uvicorn app.api_server:app --reload --host 127.0.0.1 --port 8000

It re-exports the app from app.main, which already includes
all routes (legacy API, bank data, Laya decision model).
"""

from app.main import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api_server:app", host="0.0.0.0", port=8000, reload=True)
