"""Generate the gRPC stubs into src/fedavg_core/rpc/ and make their imports package-relative.

    python scripts/gen_proto.py            # (re)generate
    python scripts/gen_proto.py --check    # fail if the committed stubs are out of date (used in CI)
"""
import argparse
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROTO_DIR = ROOT / "proto"
OUT = ROOT / "src" / "fedavg_core" / "rpc"


def generate(out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "grpc_tools.protoc", f"-I{PROTO_DIR}", f"--python_out={out_dir}", f"--pyi_out={out_dir}",
         f"--grpc_python_out={out_dir}", str(PROTO_DIR / "federated.proto")],
        check=True,
    )
    grpc_file = out_dir / "federated_pb2_grpc.py"
    text = grpc_file.read_text()
    text = re.sub(r"^import federated_pb2 as federated__pb2$", "from . import federated_pb2 as federated__pb2", text, flags=re.M)
    grpc_file.write_text(text)
    (out_dir / "__init__.py").write_text('"""Generated gRPC stubs (scripts/gen_proto.py). Do not edit by hand."""\n')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if not args.check:
        generate(OUT)
        print(f"generated stubs in {OUT}")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            generate(pathlib.Path(tmp))
            stale = [n for n in ("federated_pb2.py", "federated_pb2.pyi", "federated_pb2_grpc.py")
                     if (pathlib.Path(tmp) / n).read_text() != (OUT / n).read_text()]
        if stale:
            sys.exit(f"stale generated files: {stale}; run scripts/gen_proto.py")
        print("generated stubs are up to date")
