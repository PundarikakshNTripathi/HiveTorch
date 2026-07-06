git checkout feature/models-and-data
git add src/fedavg_core/models src/fedavg_core/data
git commit -m "feat: data and model initialization"
git checkout develop
git merge feature/models-and-data

git checkout feature/client-training
git add src/fedavg_core/client
git commit -m "feat: client training loop"
git checkout develop
git merge feature/client-training

git checkout feature/server-aggregation
git add src/fedavg_core/server
git commit -m "feat: server aggregation logic"
git checkout develop
git merge feature/server-aggregation

git checkout feature/federated-orchestration
git add scripts/orchestrate_simulation.py src/fedavg_core/versioning
git commit -m "feat: orchestration and versioning"
git checkout develop
git merge feature/federated-orchestration

git checkout feature/baselines-and-probes
git add src/fedavg_core/evaluation scripts/analyze_heterogeneity.py
git commit -m "feat: global baselines and heterogeneity analysis"
git checkout develop
git merge feature/baselines-and-probes

git checkout feature/infra-setup
git add Makefile deployments/ .github/ tests/ .config/ src/fedavg_core/monitoring src/fedavg_core/optimization src/fedavg_core/utils pyproject.toml uv.lock
git commit -m "feat: mlops infrastructure setup"
git checkout develop
git merge feature/infra-setup

git checkout feature/benchmarking-and-docs
git add .
git commit -m "docs: comprehensive readme and benchmarks"
git checkout develop
git merge feature/benchmarking-and-docs

git checkout main
git merge develop

git branch -D feature/models-and-data feature/client-training feature/server-aggregation feature/federated-orchestration feature/baselines-and-probes feature/infra-setup feature/benchmarking-and-docs

git push origin develop main
