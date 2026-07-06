.PHONY: proto lint test build-docker

proto:
	python -m grpc_tools.protoc -I proto --python_out=src/fedavg_core --grpc_python_out=src/fedavg_core proto/federated.proto

lint:
	ruff check .
	mypy .

test:
	pytest tests/

build-docker:
	docker build -t hivetorch:latest -f docker/Dockerfile .
