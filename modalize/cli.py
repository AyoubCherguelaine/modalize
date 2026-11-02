from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .runner import run

TEMPLATE = """\
# modalize config: everything about how your project runs on Modal
command: python train.py          # what to run (from this folder, in the container)

# compute
gpu: L40S                         # T4 | L40S | A100-80GB | H100 | H100:2 | null for CPU only
cpu: 8
memory: 32768                     # MiB
timeout: 21600                    # seconds (max 24h)

# environment
python_version: "3.11"
requirements: requirements.txt
apt_packages: []
env:
  PYTHONUNBUFFERED: "1"
secrets: []                       # Modal secret names, e.g. [huggingface]
volumes: {}                       # container path -> Modal volume, e.g. {/data: my-volume}
# modal_environment: main
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="modalize", description="Run this project on Modal GPUs/CPUs")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="create a config.yaml in the current folder")

    r = sub.add_parser("run", help="run the project on Modal")
    r.add_argument("config", nargs="?", default="config.yaml")
    r.add_argument("--gpu")
    r.add_argument("--cpu", type=float)
    r.add_argument("--memory", type=int, help="MiB")
    r.add_argument("--timeout", type=int, help="seconds")
    r.add_argument("--command", help="override the command")
    r.add_argument("--detach", action="store_true", help="start the job and return immediately")
    args = p.parse_args(argv)

    if args.cmd == "init":
        target = Path("config.yaml")
        if target.exists():
            print("config.yaml already exists")
            return 1
        target.write_text(TEMPLATE)
        print("created config.yaml")
        return 0

    if not Path(args.config).exists():
        print(f"{args.config} not found. Run `modalize init` first.")
        return 1
    overrides = {k: getattr(args, k) for k in ("gpu", "cpu", "memory", "timeout", "command")}
    result = run(load_config(args.config, **overrides), detach=args.detach)
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    sys.exit(main())
