from fastapi import APIRouter

from llm_backend.api.endpoints import teams, hitl

api_router = APIRouter()

api_router.include_router(teams.router, prefix="/teams", tags=["teams"])
api_router.include_router(hitl.router, tags=["hitl"])

