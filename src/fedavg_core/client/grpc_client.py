"""gRPC federation client: owns one local data partition, trains when sampled, never uploads raw data."""
import argparse
import os
import time
from typing import Optional

import grpc
import torch
from loguru import logger

from src.fedavg_core.client.edge_trainer import train_client_local
from src.fedavg_core.models.architectures import build_mlp_classifier
from src.fedavg_core.rpc import federated_pb2 as pb
from src.fedavg_core.rpc import federated_pb2_grpc as pb_grpc
from src.fedavg_core.utils.serialization import bytes_to_state, state_to_bytes
from src.fedavg_core.versioning.model_registry import load_model_state

MAX_MESSAGE_BYTES = 256 * 1024 * 1024


class FederatedClient:
    def __init__(self, client_index: int, features: torch.Tensor, labels: torch.Tensor, model_config: dict,
                 local_epochs: int, batch_size: int, learning_rate: float, seed: int, address: str,
                 connect_timeout: float = 60.0):
        self.index, self.features, self.labels = client_index, features, labels
        self.model_config, self.local_epochs = model_config, local_epochs
        self.batch_size, self.learning_rate, self.seed = batch_size, learning_rate, seed
        self.client_id = f"client-{client_index}"
        self.channel = grpc.insecure_channel(address, options=[
            ("grpc.max_send_message_length", MAX_MESSAGE_BYTES), ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES)])
        grpc.channel_ready_future(self.channel).result(timeout=connect_timeout)
        self.stub = pb_grpc.FederatedLearningStub(self.channel)

    def run(self) -> int:
        """Participate until the server reports the federation finished. Returns how many rounds this client trained."""
        join = self.stub.Join(pb.JoinRequest(client_id=self.client_id, client_index=self.index))
        if not join.accepted:
            raise RuntimeError(f"join rejected: {join.message}")
        trained = 0
        r = 0
        while r < join.num_rounds:
            resp = self.stub.GetModel(pb.ModelRequest(client_id=self.client_id, session_id=join.session_id, round=r))
            if resp.finished:
                break
            if resp.current_round != r:        # fell behind: that round already closed, rejoin at the open one
                r = resp.current_round
                continue
            if not resp.participate:
                r += 1
                continue
            model = build_mlp_classifier(**self.model_config)
            load_model_state(model, bytes_to_state(resp.global_weights))
            trained_state = train_client_local(
                model=model, client_features=self.features, client_labels=self.labels,
                local_epochs=self.local_epochs, batch_size=self.batch_size, learning_rate=self.learning_rate,
                seed=self.seed + r + self.index,           # identical to the in-process coordinator's client seed
            )
            ack = self.stub.SendUpdate(pb.ClientUpdate(
                client_id=self.client_id, session_id=join.session_id, round=r,
                weights=state_to_bytes(trained_state), num_samples=int(self.features.shape[0])))
            if not ack.success:
                raise RuntimeError(f"update rejected in round {r}: {ack.message}")
            trained += 1
            r += 1
        self.channel.close()
        return trained


def main() -> None:
    from src.fedavg_core.data.dataset_downloader import build_synthetic_dataset
    from src.fedavg_core.data.non_iid_sharder import partition_data_iid
    from src.fedavg_core.data.processing_pipeline import train_test_split_dataset
    from src.fedavg_core.utils.configuration_factory import load_config

    ap = argparse.ArgumentParser(description="HiveTorch federation client")
    ap.add_argument("--server", default=os.environ.get("SERVER_ADDRESS", f"{os.environ.get('SERVER_HOST', 'localhost')}:50051"))
    ap.add_argument("--index", type=int, default=None,
                    help="client index; defaults to $CLIENT_INDEX or the Kubernetes Indexed-Job completion index")
    a = ap.parse_args()
    index: Optional[int] = a.index
    if index is None:
        index = int(os.environ.get("CLIENT_INDEX", os.environ.get("JOB_COMPLETION_INDEX", "0")))

    cfg = load_config()
    m, t, f = cfg.model, cfg.training, cfg.federated
    # Every client rebuilds the same synthetic pool from the shared seed and keeps only its own shard,
    # standing in for data that physically lives on the device.
    features, labels = build_synthetic_dataset(1000, m.input_size, m.num_classes, t.seed)
    train_f, train_l, _, _ = train_test_split_dataset(features, labels, 0.2, t.seed)
    part_f, part_l = partition_data_iid(train_f, train_l, f.num_clients, t.seed)[index]
    logger.info(f"client {index}: {part_f.shape[0]} local samples, server {a.server}")
    started = time.time()
    n = FederatedClient(index, part_f, part_l, dict(m), t.local_epochs, t.batch_size, t.learning_rate, t.seed, a.server).run()
    logger.info(f"client {index}: trained in {n} rounds ({time.time() - started:.1f}s)")


if __name__ == "__main__":
    main()
