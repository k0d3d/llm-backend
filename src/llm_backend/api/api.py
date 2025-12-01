from fastapi import APIRouter

from llm_backend.api.endpoints import teams, hitl, database

api_router = APIRouter()

api_router.include_router(teams.router, prefix="/teams", tags=["teams"])
api_router.include_router(hitl.router, tags=["hitl"])
api_router.include_router(database.router, tags=["database"])

