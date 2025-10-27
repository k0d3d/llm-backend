"""RQ Worker script"""
import os
import sys
import logging
from dotenv import load_dotenv
from rq import Worker, Queue, Connection

# Load environment variables first
load_dotenv()

from llm_backend.workers.connection import get_redis_connection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def start_worker():
    """Start RQ worker"""
    redis_conn = get_redis_connection()

    with Connection(redis_conn):
        queues = ['default']
        worker = Worker(
            queues,
            connection=redis_conn,
            name=f'worker-{os.getpid()}'
        )
        logger.info(f"Starting RQ worker: {worker.name}")
        logger.info(f"Listening on queues: {queues}")
        worker.work(with_scheduler=False)

if __name__ == '__main__':
    try:
        start_worker()
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Worker error: {e}")
        sys.exit(1)
