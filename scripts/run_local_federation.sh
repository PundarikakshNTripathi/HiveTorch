#!/usr/bin/env bash
# Run the gRPC federation on one machine: 1 server + 5 client processes. Exits non-zero if anything fails.
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-50051}"
K=5

uv run python main.py server --address "127.0.0.1:${PORT}" --num-clients "$K" --num-rounds 10 --metrics-port 0 &
SERVER=$!
sleep 2
PIDS=()
for i in $(seq 0 $((K - 1))); do
    CLIENT_INDEX=$i uv run python main.py client --server "127.0.0.1:${PORT}" &
    PIDS+=($!)
done
status=0
for p in "${PIDS[@]}"; do wait "$p" || status=1; done
wait "$SERVER" || status=1
exit $status
