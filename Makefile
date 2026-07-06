.PHONY: setup dvc-init proto lint test docker-build cluster-up deploy monitor-up wandb-login all clean

# Load environment variables if .env exists
-include .env
export

all: setup dvc-init proto lint test docker-build cluster-up monitor-up deploy wandb-login run-pipeline run-tuning run-benchmarks

setup:
	@echo "Installing dependencies..."
	uv sync

dvc-init:
	@echo "Setting up local DVC storage..."
	@mkdir -p .dvc-storage
	@grep -q ".dvc-storage" .gitignore || echo ".dvc-storage" >> .gitignore
	@dvc init -f || true
	@dvc remote add -d local_storage .dvc-storage -f
	@git add .dvc/config .dvcignore .gitignore
	@git commit -m "chore: configure local dvc storage" || true

proto:
	@echo "Compiling protobufs..."
	python -m grpc_tools.protoc -I proto --python_out=src/fedavg_core --grpc_python_out=src/fedavg_core proto/federated.proto

lint:
	@echo "Running linters..."
	uv run ruff check .
	uv run mypy .

test:
	@echo "Running tests..."
	uv run pytest tests/

docker-build:
	@echo "Building Docker image..."
	docker build -t hivetorch:latest -f docker/Dockerfile .

cluster-up:
	@echo "Starting Minikube..."
	minikube status || minikube start --cpus=4 --memory=8192
	minikube image load hivetorch:latest

deploy:
	@echo "Deploying to Kubernetes..."
	kubectl apply -f deployments/k8s-simulation-job.yaml
	kubectl apply -f deployments/prometheus-rules.yaml

monitor-up:
	@echo "Setting up Prometheus and Grafana..."
	helm repo add prometheus-community https://prometheus-community.github.io/helm-charts || true
	helm repo update
	helm upgrade --install prometheus prometheus-community/prometheus
	helm upgrade --install grafana grafana/grafana

wandb-login:
	@echo "Logging into Weights & Biases..."
	@if [ -z "$(WANDB_API_KEY)" ]; then \
		echo "WANDB_API_KEY not found in .env"; \
		exit 1; \
	fi
	wandb login $(WANDB_API_KEY)

run-pipeline:
	@echo "Executing End-to-End Federated Pipeline..."
	uv run python scripts/run_pipeline.py

run-tuning:
	@echo "Executing Optuna Hyperparameter Sweep..."
	uv run python scripts/tune_hyperparameters.py

run-benchmarks:
	@echo "Executing Benchmarks & Generating Visualizations..."
	uv run python scripts/run_benchmarks.py

clean:
	@echo "Cleaning up..."
	minikube delete || true
	rm -rf .dvc-storage
