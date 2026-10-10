"""HiveTorch entry point.

    python main.py simulate   # in-process FedAvg on the synthetic dataset (no network)
    python main.py server     # gRPC federation server (see src/fedavg_core/server/grpc_server.py for flags)
    python main.py client     # gRPC federation client  (see src/fedavg_core/client/grpc_client.py for flags)
"""
import sys


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "simulate"
    sys.argv = [f"main.py {command}"] + sys.argv[2:]
    if command == "simulate":
        from scripts.run_pipeline import main as run
    elif command == "server":
        from src.fedavg_core.server.grpc_server import main as run
    elif command == "client":
        from src.fedavg_core.client.grpc_client import main as run
    else:
        raise SystemExit(__doc__)
    run()


if __name__ == "__main__":
    main()
