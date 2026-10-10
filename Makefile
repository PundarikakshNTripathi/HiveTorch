.PHONY: help setup proto lint test simulate experiments federation-local compose-up docker-build cluster-up deploy monitor-up clean all

help:
	@echo "make setup             install dependencies (uv)"
	@echo "make proto             regenerate gRPC stubs into src/fedavg_core/rpc"
	@echo "make lint              ruff + mypy + stub freshness check"
	@echo "make test              unit tests + gRPC federation-vs-in-process equivalence tests"
	@echo "make simulate          in-process FedAvg on the synthetic dataset"
	@echo "make experiments       IID vs Dirichlet vs shards on MNIST (regenerates the README table and docs/results/)"
	@echo "make federation-local  gRPC server + 5 client processes on this machine"
	@echo "make compose-up        the same federation in Docker Compose"
	@echo "make cluster-up deploy run it on a local Minikube cluster (Indexed Job)"

all: setup proto lint test simulate

setup:
	uv sync

proto:
	uv run python scripts/gen_proto.py

lint:
	uv run ruff check .
	uv run mypy src scripts main.py
	uv run python scripts/gen_proto.py --check

test:
	uv run pytest tests/ -v

simulate:
	uv run python main.py simulate

experiments:
	uv run python scripts/run_heterogeneity_experiment.py

federation-local:
	bash scripts/run_local_federation.sh

compose-up:
	docker compose -f docker/docker-compose.yml up --build --abort-on-container-exit --exit-code-from server

docker-build:
	docker build -t hivetorch:latest -f docker/Dockerfile .

cluster-up:
	minikube status || minikube start --cpus=4 --memory=4096
	minikube image load hivetorch:latest

deploy:
	kubectl apply -f deployments/k8s-simulation-job.yaml
	kubectl wait --for=condition=complete job/hivetorch-server --timeout=600s
	kubectl logs job/hivetorch-server | tail -20

monitor-up:
	helm repo add prometheus-community https://prometheus-community.github.io/helm-charts || true
	helm repo update
	helm upgrade --install prometheus prometheus-community/kube-prometheus-stack
	kubectl apply -f deployments/prometheus-rules.yaml

clean:
	kubectl delete -f deployments/k8s-simulation-job.yaml --ignore-not-found
