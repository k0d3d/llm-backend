from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict
from llm_backend.core.security import IPBlocker

router = APIRouter()

@router.get("/blocked-ips", response_model=Dict[str, str])
async def get_blocked_ips():
    """Get all blocked IPs and their reasons."""
    blocker = IPBlocker()
    return blocker.get_blocked_ips()

@router.post("/unblock-ip/{ip}")
async def unblock_ip(ip: str):
    """Unblock a specific IP."""
    blocker = IPBlocker()
    blocker.unblock_ip(ip)
    return {"message": f"IP {ip} unblocked successfully."}

@router.post("/clear-blocked-ips")
async def clear_blocked_ips():
    """Clear all blocked IPs."""
    blocker = IPBlocker()
    ips = blocker.get_blocked_ips()
    for ip in ips:
        blocker.unblock_ip(ip)
    return {"message": "All IPs unblocked successfully.", "count": len(ips)}
