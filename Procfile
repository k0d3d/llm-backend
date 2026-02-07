web: poetry run fastapi run src/main.py --host 0.0.0.0 --port $PORT --workers 4
worker: poetry run python -m llm_backend.workers.worker
