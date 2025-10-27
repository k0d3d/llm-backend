"""Redis connection for RQ"""
import os
from redis import Redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

def get_redis_connection():
    """Get Redis connection for RQ"""
    return Redis.from_url(REDIS_URL)
