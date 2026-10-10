"""FedAvg under label skew: IID vs Dirichlet(alpha) vs pathological shards, on MNIST.

20 clients, 50% sampled per round, 1 local epoch, MLP 784-128-10, SGD(lr=0.05), 40 rounds, 3 seeds. Test accuracy of
the global model is the mean over seeds; the table reports final accuracy and the number of rounds to reach 90% (or "-").

    uv run python scripts/run_heterogeneity_experiment.py            # writes docs/results/ and updates the README table
    uv run python scripts/run_heterogeneity_experiment.py --quick    # tiny smoke run
"""
import argparse
import json
import os
import re
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.fedavg_core.data.non_iid_sharder import (  # noqa: E402
    partition_data_dirichlet,
    partition_data_iid,
    partition_data_non_iid,
)
from src.fedavg_core.server.coordinator import run_fedavg  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "docs", "results")


def load_mnist(quick: bool):
    from torchvision import datasets

    root = os.path.join(ROOT, "data")
    tr = datasets.MNIST(root, train=True, download=True)
    te = datasets.MNIST(root, train=False, download=True)

    def prep(ds, n=None):
        x = ds.data.float().div(255.0).sub(0.1307).div(0.3081).reshape(len(ds), -1)
        y = ds.targets.long()
        return (x[:n], y[:n]) if n else (x, y)

    return (*prep(tr, 6000 if quick else None), *prep(te, 2000 if quick else None))


def inject_readme(readme, summary):
    start, end = "<!-- HETEROGENEITY_START -->", "<!-- HETEROGENEITY_END -->"
    if not os.path.exists(readme):
        return
    text = open(readme, encoding="utf-8").read()
    if start not in text:
        print("README markers not found; skipping")
        return
    lines = ["| Partition | Final accuracy (mean ± std) | Best-round accuracy | Rounds to 90% |", "|---|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['setting']} | {100 * r['final_mean']:.2f}% ± {100 * r['final_std']:.2f} | {100 * r['best_mean']:.2f}% | {r['rounds_to_90pct'] or '-'} |")
    body = start + "\n" + "\n".join(lines) + "\n" + end
    text = re.sub(re.escape(start) + r".*?" + re.escape(end), lambda _: body, text, flags=re.S)
    open(readme, "w", encoding="utf-8").write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--readme-only", action="store_true", help="re-inject the table from docs/results/heterogeneity.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--clients", type=int, default=20)
    ap.add_argument("--readme", default=os.path.join(ROOT, "README.md"))
    a = ap.parse_args()
    if a.readme_only:
        with open(os.path.join(OUT, "heterogeneity.json")) as f:
            inject_readme(a.readme, json.load(f)["summary"])
        return
    if a.quick:
        a.rounds, a.seeds = 3, [0]
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    train_x, train_y, test_x, test_y = load_mnist(a.quick)
    model_cfg = {"input_size": 784, "hidden_size": 128, "num_classes": 10}
    K = a.clients
    settings = {
        "IID": lambda s: partition_data_iid(train_x, train_y, K, s),
        "Dirichlet a=10": lambda s: partition_data_dirichlet(train_x, train_y, K, 10.0, s),
        "Dirichlet a=1": lambda s: partition_data_dirichlet(train_x, train_y, K, 1.0, s),
        "Dirichlet a=0.3": lambda s: partition_data_dirichlet(train_x, train_y, K, 0.3, s),
        "Dirichlet a=0.1": lambda s: partition_data_dirichlet(train_x, train_y, K, 0.1, s),
        "Shards (2 per client)": lambda s: partition_data_non_iid(train_x, train_y, K, 2, s),
    }

    curves, t0 = {}, time.time()
    for name, make in settings.items():
        runs = []
        for seed in a.seeds:
            parts = make(seed)
            _, accs = run_fedavg(parts, test_x, test_y, model_cfg, a.rounds, 0.5, 1, 32, 0.05, seed)
            runs.append(accs)
            print(f"{name:24s} seed {seed}: final {accs[-1]:.4f}  ({time.time() - t0:.0f}s)")
        curves[name] = np.array(runs)

    os.makedirs(OUT, exist_ok=True)
    summary = []
    for name, arr in curves.items():
        mean = arr.mean(0)
        reach = next((i + 1 for i, v in enumerate(mean) if v >= 0.90), None)
        summary.append({"setting": name, "final_mean": float(mean[-1]), "final_std": float(arr[:, -1].std()),
                        "best_mean": float(mean.max()), "rounds_to_90pct": reach})
    with open(os.path.join(OUT, "heterogeneity.json"), "w") as f:
        json.dump({"config": {"clients": K, "client_fraction": 0.5, "local_epochs": 1, "rounds": a.rounds, "seeds": a.seeds,
                              "model": model_cfg, "lr": 0.05, "batch_size": 32}, "summary": summary,
                   "curves": {k: v.mean(0).tolist() for k, v in curves.items()}}, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for (name, arr), color in zip(curves.items(), ["#2f9e44", "#74c0fc", "#4c6ef5", "#f08c00", "#e03131", "#862e9c"]):
        m, s = arr.mean(0), arr.std(0)
        x = np.arange(1, len(m) + 1)
        ax.plot(x, m, label=name, color=color)
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.12)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Global test accuracy (MNIST)")
    ax.set_title(f"FedAvg under label skew ({K} clients, 50% sampled, mean of {len(a.seeds)} seeds)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "convergence.png"), dpi=200)

    if not a.quick:
        inject_readme(a.readme, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
