#!/bin/bash

# Start the FastAPI server
poetry run fastapi run src/main.py --port $PORT --workers 2 &
api_pid=$!
echo "FastAPI server started with PID: $api_pid"


# Trap the CTRL+C signal
trap "kill $api_pid; wait" INT

# Wait for the processes to finish
wait
