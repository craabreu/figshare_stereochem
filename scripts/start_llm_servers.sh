#!/bin/bash
# Script to start a specified number of Ollama servers, each on a different GPU

# Read number of servers from command line argument; default to 8.
NUM_SERVERS=${1:-8}

echo "Starting $NUM_SERVERS Ollama servers on separate GPUs..."

# Kill any existing ollama processes
pkill ollama 2>/dev/null || true
sleep 2

# Start ollama servers, each on a different GPU and port
for ((gpu=0; gpu<NUM_SERVERS; gpu++)); do
    port=$((11434 + gpu))
    echo "Starting Ollama server on GPU $gpu, port $port"
    CUDA_VISIBLE_DEVICES=$gpu OLLAMA_HOST=127.0.0.1:$port nohup ollama serve > ollama_gpu${gpu}.log 2>&1 &
    sleep 2  # Give each server time to start
    echo "Ollama server on GPU $gpu, port $port started!"
done

echo "All Ollama servers started!"

# Wait for servers to be ready
echo "Waiting for servers to be ready..."
sleep 10

# Test connectivity
for ((gpu=0; gpu<NUM_SERVERS; gpu++)); do
    port=$((11434 + gpu))
    echo "Testing GPU $gpu (port $port)..."
    curl -s http://localhost:$port/api/tags > /dev/null \
        && echo "[✓] GPU $gpu ready" \
        || echo "[✗] GPU $gpu failed"
done
