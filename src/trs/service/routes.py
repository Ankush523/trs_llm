from __future__ import annotations

from trs.data.registry import get_dataset_adapter


def ensure_fastapi():
    try:
        from fastapi import APIRouter, FastAPI, HTTPException  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("FastAPI is not installed. Install optional api dependencies first.") from exc


def build_router(service) -> "APIRouter":
    ensure_fastapi()
    from fastapi import APIRouter, HTTPException

    router = APIRouter()

    @router.post("/v1/reason")
    def reason(payload: dict):
        try:
            return service.reason(payload)
        except Exception as exc:  # pragma: no cover - service adapter
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/health")
    def health():
        return {"status": "ok"}

    return router
