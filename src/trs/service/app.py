from __future__ import annotations


def create_app(service):
    try:
        from fastapi import FastAPI
    except ImportError as exc:
        raise RuntimeError("FastAPI is not installed. Install optional api dependencies first.") from exc
    from trs.service.routes import build_router

    app = FastAPI(title="TRS Service")
    app.include_router(build_router(service))
    return app
