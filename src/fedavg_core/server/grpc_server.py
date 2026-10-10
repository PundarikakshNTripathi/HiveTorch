"""gRPC federation server: drives synchronous FedAvg rounds over the network.

The aggregation maths is exactly the in-process path (`select_round_clients`, `aggregate_weighted_average`), so a
federation run over gRPC with the same seeds reproduces `run_fedavg` bit for bit; only the transport differs.
"""
import argparse
import threading
import uuid
from concurrent import futures
from typing import Dict, List, Optional, Tuple

import grpc
import torch
from loguru import logger
from prometheus_client import Counter, Gauge, start_http_server

from src.fedavg_core.evaluation.global_evaluator import evaluate_accuracy
from src.fedavg_core.models.architectures import build_mlp_classifier
from src.fedavg_core.monitoring.drift_detector import ConceptDriftDetector
from src.fedavg_core.rpc import federated_pb2 as pb
from src.fedavg_core.rpc import federated_pb2_grpc as pb_grpc
from src.fedavg_core.server.coordinator import initialize_global_state, select_round_clients
from src.fedavg_core.server.secure_aggregator import aggregate_weighted_average
from src.fedavg_core.utils.serialization import bytes_to_state, state_to_bytes
from src.fedavg_core.versioning.model_registry import load_model_state

MAX_MESSAGE_BYTES = 256 * 1024 * 1024

# Prometheus metrics (scraped on --metrics-port; alert rules in deployments/prometheus-rules.yaml)
ROUND_GAUGE = Gauge("federation_round", "Index of the currently open federated round")
ACCURACY_GAUGE = Gauge("federation_global_accuracy", "Test accuracy of the global model after the last round")
DRIFT_FLAG = Gauge("federation_drift_flag", "1 if the last round's accuracy dropped >10% below the best so far")
UPDATES_ACCEPTED = Counter("federation_updates_accepted_total", "Client updates accepted into an aggregation")
UPDATES_REJECTED = Counter("federation_updates_rejected_total", "Client updates rejected (wrong round, bad payload, ...)")


class FederationServicer(pb_grpc.FederatedLearningServicer):
    def __init__(
        self,
        num_clients: int,
        num_rounds: int,
        client_fraction: float,
        model_config: dict,
        seed: int,
        test_features: Optional[torch.Tensor] = None,
        test_labels: Optional[torch.Tensor] = None,
        round_timeout: float = 600.0,
    ):
        self.num_clients, self.num_rounds = num_clients, num_rounds
        self.client_fraction, self.model_config, self.seed = client_fraction, model_config, seed
        self.test_features, self.test_labels = test_features, test_labels
        self.round_timeout = round_timeout

        self.global_state = initialize_global_state(**model_config, seed=seed)
        self.session_id = uuid.uuid4().hex
        self.round = -1                      # -1 until every client has joined
        self.selected: List[int] = []
        self.joined: Dict[int, str] = {}     # client_index -> client_id
        self.updates: Dict[int, Tuple[dict, int]] = {}
        self.accuracies: List[float] = []
        self.cond = threading.Condition()

    # ---- helpers (call with self.cond held) --------------------------------------------------------
    @property
    def finished(self) -> bool:
        return self.round >= self.num_rounds

    def _open_round(self, r: int) -> None:
        self.round = r
        ROUND_GAUGE.set(r)
        self.updates = {}
        if r < self.num_rounds:
            # same cohort selection and seeds as the in-process coordinator
            self.selected = select_round_clients(self.num_clients, self.client_fraction, self.seed + r)
            logger.info(f"round {r}: clients {self.selected}")

    def _aggregate_and_advance(self) -> None:
        order = sorted(self.updates)                      # deterministic aggregation order
        states = [self.updates[i][0] for i in order]
        counts = [self.updates[i][1] for i in order]
        self.global_state = aggregate_weighted_average(states, counts)
        if self.test_features is not None and self.test_labels is not None:
            model = build_mlp_classifier(**self.model_config)
            load_model_state(model, self.global_state)
            acc = evaluate_accuracy(model, self.test_features, self.test_labels)
            best_before = max(self.accuracies, default=acc)
            self.accuracies.append(acc)
            ACCURACY_GAUGE.set(acc)
            DRIFT_FLAG.set(1.0 if ConceptDriftDetector(threshold=0.10).check_accuracy_drift(best_before, acc) else 0.0)
            logger.info(f"round {self.round}: test accuracy {acc:.4f}")
        self._open_round(self.round + 1)
        self.cond.notify_all()

    @staticmethod
    def _reject(message):
        UPDATES_REJECTED.inc()
        return pb.ServerAck(success=False, message=message)

    # ---- RPCs --------------------------------------------------------------------------------------
    def Join(self, request, context):
        with self.cond:
            idx = request.client_index
            if not 0 <= idx < self.num_clients:
                return pb.JoinResponse(accepted=False, message=f"client_index must be in [0, {self.num_clients})")
            if idx in self.joined and self.joined[idx] != request.client_id:
                return pb.JoinResponse(accepted=False, message=f"client_index {idx} already taken")
            self.joined[idx] = request.client_id
            if len(self.joined) == self.num_clients and self.round == -1:
                self._open_round(0)
                self.cond.notify_all()
            return pb.JoinResponse(accepted=True, session_id=self.session_id, num_rounds=self.num_rounds)

    def GetModel(self, request, context):
        with self.cond:
            if request.session_id != self.session_id:
                context.abort(grpc.StatusCode.PERMISSION_DENIED, "unknown session")
            idx = next((i for i, cid in self.joined.items() if cid == request.client_id), None)
            if idx is None:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, "join first")
            ok = self.cond.wait_for(lambda: self.round >= request.round, timeout=self.round_timeout)
            if not ok:
                context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, f"round {request.round} did not open")
            if self.finished:
                return pb.ModelResponse(current_round=self.round, participate=False, finished=True)
            if request.round < self.round:
                # a slow client asking about a round that has already closed: point it at the open round
                return pb.ModelResponse(current_round=self.round, participate=False, finished=False)
            participate = idx in self.selected
            return pb.ModelResponse(
                current_round=self.round,
                participate=participate,
                finished=False,
                global_weights=state_to_bytes(self.global_state) if participate else b"",
            )

    def SendUpdate(self, request, context):
        with self.cond:
            if request.session_id != self.session_id:
                return self._reject("unknown session")
            idx = next((i for i, cid in self.joined.items() if cid == request.client_id), None)
            if idx is None or request.round != self.round or idx not in self.selected:
                return self._reject("update not expected from this client in this round")
            if idx in self.updates:
                return self._reject("duplicate update")
            if request.num_samples <= 0:
                return self._reject("num_samples must be positive")
            try:
                state = bytes_to_state(request.weights)
            except Exception as e:  # noqa: BLE001 - reject malformed payloads without taking the server down
                return self._reject(f"bad weights payload: {e}")
            ref = self.global_state
            if state.keys() != ref.keys() or any(state[k].shape != ref[k].shape or state[k].dtype != ref[k].dtype for k in ref):
                return self._reject("weights do not match the global architecture")
            self.updates[idx] = (state, request.num_samples)
            UPDATES_ACCEPTED.inc()
            if len(self.updates) == len(self.selected):
                self._aggregate_and_advance()
            return pb.ServerAck(success=True)

    def GetStatus(self, request, context):
        with self.cond:
            return pb.StatusResponse(current_round=max(self.round, 0), num_rounds=self.num_rounds,
                                     finished=self.finished, round_accuracies=self.accuracies)

    def wait_finished(self, timeout: Optional[float] = None) -> bool:
        with self.cond:
            return self.cond.wait_for(lambda: self.finished, timeout=timeout)


def serve(servicer: FederationServicer, address: str = "[::]:50051") -> Tuple[grpc.Server, int]:
    """Start the server; returns (server, bound_port). `address` may use port 0 to pick a free port."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=servicer.num_clients + 4),
        options=[("grpc.max_send_message_length", MAX_MESSAGE_BYTES), ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES)],
    )
    pb_grpc.add_FederatedLearningServicer_to_server(servicer, server)
    port = server.add_insecure_port(address)
    server.start()
    return server, port


def main() -> None:
    from src.fedavg_core.data.dataset_downloader import build_synthetic_dataset
    from src.fedavg_core.data.processing_pipeline import train_test_split_dataset
    from src.fedavg_core.utils.configuration_factory import load_config

    ap = argparse.ArgumentParser(description="HiveTorch federation server")
    ap.add_argument("--address", default="[::]:50051")
    ap.add_argument("--num-clients", type=int)
    ap.add_argument("--num-rounds", type=int)
    ap.add_argument("--client-fraction", type=float)
    ap.add_argument("--metrics-port", type=int, default=8000, help="Prometheus /metrics port (0 disables)")
    a = ap.parse_args()

    cfg = load_config()
    num_clients = a.num_clients or cfg.federated.num_clients
    servicer = _servicer_from_config(cfg, num_clients, a.num_rounds or cfg.federated.num_rounds,
                                     a.client_fraction or cfg.federated.client_fraction,
                                     build_synthetic_dataset, train_test_split_dataset)
    if a.metrics_port:
        start_http_server(a.metrics_port)
    server, port = serve(servicer, a.address)
    logger.info(f"federation server listening on port {port}; waiting for {num_clients} clients")
    servicer.wait_finished()
    logger.info(f"federation finished; accuracies per round: {[round(x, 4) for x in servicer.accuracies]}")
    server.stop(grace=5).wait()


def _servicer_from_config(cfg, num_clients, num_rounds, client_fraction, build_dataset, split):
    m, t = cfg.model, cfg.training
    features, labels = build_dataset(1000, m.input_size, m.num_classes, t.seed)
    _, _, test_f, test_l = split(features, labels, 0.2, t.seed)
    return FederationServicer(num_clients, num_rounds, client_fraction, dict(m), t.seed, test_f, test_l)


if __name__ == "__main__":
    main()
