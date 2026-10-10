"""FedAvg over a real gRPC channel must reproduce the in-process FedAvg result."""
import threading

import grpc
import pytest
import torch

from src.fedavg_core.client.grpc_client import FederatedClient
from src.fedavg_core.data.dataset_downloader import build_synthetic_dataset
from src.fedavg_core.data.non_iid_sharder import partition_data_dirichlet, partition_data_iid
from src.fedavg_core.data.processing_pipeline import train_test_split_dataset
from src.fedavg_core.rpc import federated_pb2 as pb
from src.fedavg_core.rpc import federated_pb2_grpc as pb_grpc
from src.fedavg_core.server.coordinator import run_fedavg
from src.fedavg_core.server.grpc_server import FederationServicer, serve
from src.fedavg_core.utils.serialization import bytes_to_state, state_to_bytes

MODEL = {"input_size": 8, "hidden_size": 16, "num_classes": 3}
HP = dict(local_epochs=2, batch_size=16, learning_rate=0.05, seed=7)
K, ROUNDS, FRACTION = 5, 4, 0.6


def _data():
    feats, labels = build_synthetic_dataset(600, MODEL["input_size"], MODEL["num_classes"], HP["seed"])
    tr_f, tr_l, te_f, te_l = train_test_split_dataset(feats, labels, 0.2, HP["seed"])
    return tr_f, tr_l, te_f, te_l


def _run_grpc(partitions, te_f, te_l):
    servicer = FederationServicer(K, ROUNDS, FRACTION, MODEL, HP["seed"], te_f, te_l)
    server, port = serve(servicer, "127.0.0.1:0")
    results, errors = {}, []

    def work(i):
        try:
            c = FederatedClient(i, *partitions[i], MODEL, HP["local_epochs"], HP["batch_size"], HP["learning_rate"],
                                HP["seed"], f"127.0.0.1:{port}")
            results[i] = c.run()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(K)]
    for t in threads:
        t.start()
    assert servicer.wait_finished(timeout=120)
    for t in threads:
        t.join(timeout=30)
    server.stop(grace=1).wait()
    assert not errors, errors
    return servicer, results


@pytest.mark.parametrize("partitioner", ["iid", "dirichlet"])
def test_grpc_federation_matches_in_process_fedavg(partitioner):
    tr_f, tr_l, te_f, te_l = _data()
    parts = (partition_data_iid(tr_f, tr_l, K, HP["seed"]) if partitioner == "iid"
             else partition_data_dirichlet(tr_f, tr_l, K, alpha=0.5, seed=HP["seed"], min_client_size=5))

    servicer, results = _run_grpc(parts, te_f, te_l)
    _, ref_acc = run_fedavg(parts, te_f, te_l, MODEL, ROUNDS, FRACTION, HP["local_epochs"], HP["batch_size"],
                            HP["learning_rate"], HP["seed"])

    assert len(servicer.accuracies) == ROUNDS
    assert servicer.accuracies == pytest.approx(ref_acc, abs=1e-6)
    assert sum(results.values()) == ROUNDS * max(1, round(FRACTION * K))   # every sampled client trained exactly once per round


def test_server_rejects_bad_requests():
    servicer = FederationServicer(2, 2, 1.0, MODEL, 0)
    server, port = serve(servicer, "127.0.0.1:0")
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as ch:
            stub = pb_grpc.FederatedLearningStub(ch)
            assert not stub.Join(pb.JoinRequest(client_id="x", client_index=9)).accepted          # index out of range
            a = stub.Join(pb.JoinRequest(client_id="a", client_index=0))
            assert a.accepted
            assert not stub.Join(pb.JoinRequest(client_id="intruder", client_index=0)).accepted   # index already taken
            stub.Join(pb.JoinRequest(client_id="b", client_index=1))                              # round 0 opens
            resp = stub.GetModel(pb.ModelRequest(client_id="a", session_id=a.session_id, round=0))
            assert resp.participate and not resp.finished
            good = bytes_to_state(resp.global_weights)

            def send(state, **kw):
                return stub.SendUpdate(pb.ClientUpdate(client_id="a", session_id=a.session_id, round=0,
                                                       weights=state_to_bytes(state), num_samples=10, **kw))

            assert not stub.SendUpdate(pb.ClientUpdate(client_id="a", session_id="bogus", round=0)).success
            wrong = {k: v[:1] for k, v in good.items()}
            assert not send(wrong).success                                                         # wrong shapes
            assert not stub.SendUpdate(pb.ClientUpdate(client_id="a", session_id=a.session_id, round=0,
                                                       weights=b"not safetensors", num_samples=10)).success
            assert not stub.SendUpdate(pb.ClientUpdate(client_id="a", session_id=a.session_id, round=1,
                                                       weights=state_to_bytes(good), num_samples=10)).success   # wrong round
            assert send(good).success
            assert not send(good).success                                                          # duplicate
    finally:
        server.stop(grace=0).wait()
    assert torch.is_tensor(next(iter(good.values())))
