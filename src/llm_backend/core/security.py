import logging
import re
import os
from datetime import datetime
from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware
from redis import Redis

# Patterns derived from the log entries provided
SUSPICIOUS_PATTERNS = [
    r"\.env$",
    r"\.git",
    r"\.php",
    r"\.action",
    r"wp-config",
    r"wp-includes",
    r"xmlrpc\.php",
    r"cgi-bin",
    r"axis2-admin",
    r"axis2/axis2-admin",
    r"java\.lang\.ProcessBuilder",
    r"Runtime\.getRuntime\(\)\.exec",
    r"allow_url_include",
    r"auto_prepend_file",
    r"q=node&destination=node",
    r"wlwmanifest\.xml",
    r"PROPFIND",
    r"robots\.txt",
]

SUSPICIOUS_EXTENSIONS = [
    ".php", ".asp", ".aspx", ".jsp", ".cgi", ".sh", ".sql"
]

# Internal networks or IPs that should never be blocked
WHITELISTED_IPS = ["127.0.0.1", "::1"]

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
BLOCK_LIST_KEY = "blocked_ips"
BLOCK_DETAILS_PREFIX = "blocked_ip_info:"

logger = logging.getLogger(__name__)

class IPBlocker:
    def __init__(self):
        self.redis = Redis.from_url(REDIS_URL, decode_responses=True)

    def is_blocked(self, ip: str) -> bool:
        if ip in WHITELISTED_IPS:
            return False
        return self.redis.sismember(BLOCK_LIST_KEY, ip)

    def block_ip(self, ip: str, reason: str):
        if ip in WHITELISTED_IPS:
            logger.info(f"Skipping block for whitelisted IP: {ip} Reason: {reason}")
            return

        if not self.redis.sismember(BLOCK_LIST_KEY, ip):
            logger.warning(f"Blocking IP: {ip} Reason: {reason}")
            self.redis.sadd(BLOCK_LIST_KEY, ip)
            self.redis.set(
                f"{BLOCK_DETAILS_PREFIX}{ip}",
                f"Blocked at {datetime.now().isoformat()} - Reason: {reason}"
            )

    def get_blocked_ips(self):
        ips = self.redis.smembers(BLOCK_LIST_KEY)
        details = {}
        for ip in ips:
            info = self.redis.get(f"{BLOCK_DETAILS_PREFIX}{ip}")
            details[ip] = info or "No details available"
        return details

    def unblock_ip(self, ip: str):
        self.redis.srem(BLOCK_LIST_KEY, ip)
        self.redis.delete(f"{BLOCK_DETAILS_PREFIX}{ip}")

class SecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            client_host = forwarded_for.split(",")[0].strip()
        else:
            client_host = request.client.host

        # WHITELIST LOGIC:
        # 1. Check for Cloudflare Worker specific headers (like X-Tenant-ID) 
        #    which indicates it's coming from our trusted core-api or llm-invoker.
        # 2. Check for internal gateway IP (172.x.x.x) which is often the proxy.
        
        is_internal = (
            request.headers.get("x-tenant-id") is not None or
            request.headers.get("authorization") is not None or
            client_host.startswith("172.") # Docker internal network
        )

        blocker = IPBlocker()

        if not is_internal and blocker.is_blocked(client_host):
            return Response(
                content="Access Denied",
                status_code=status.HTTP_403_FORBIDDEN
            )

        path = request.url.path
        query = request.url.query
        method = request.method

        # Skip inspection for trusted internal requests to prevent accidental blocking
        if is_internal:
            return await call_next(request)

        if method == "PROPFIND":
            blocker.block_ip(client_host, f"Suspicious method: {method}")
            return Response(content="Access Denied", status_code=status.HTTP_403_FORBIDDEN)

        full_path = f"{path}?{query}" if query else path
        
        if path.startswith("//"):
             blocker.block_ip(client_host, f"Suspicious path format: {path}")
             return Response(content="Access Denied", status_code=status.HTTP_403_FORBIDDEN)

        for pattern in SUSPICIOUS_PATTERNS:
            if re.search(pattern, full_path, re.IGNORECASE):
                blocker.block_ip(client_host, f"Matched suspicious pattern: {pattern} in {full_path}")
                return Response(content="Access Denied", status_code=status.HTTP_403_FORBIDDEN)

        if any(path.endswith(ext) for ext in SUSPICIOUS_EXTENSIONS):
            blocker.block_ip(client_host, f"Suspicious file extension in path: {path}")
            return Response(content="Access Denied", status_code=status.HTTP_403_FORBIDDEN)

        response = await call_next(request)
        return response
