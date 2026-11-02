from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import yaml


@dataclass
class RunConfig:
    name: str
    command: str = "python train.py"
    # compute
    gpu: str | None = None  # "T4", "L40S", "A100-80GB", "H100", "H100:2" ...
    cpu: float | None = None
    memory: int | None = None  # MiB
    timeout: int = 6 * 3600  # seconds, Modal max is 24h
    # environment
    python_version: str = "3.11"
    requirements: str | None = "requirements.txt"
    apt_packages: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    secrets: list[str] = field(default_factory=list)  # names of Modal Secrets
    # code + storage
    project_dir: str = "."  # (relative to config.yaml) uploaded to `workdir` in the container
    workdir: str = "/workspace"
    ignore: list[str] = field(default_factory=lambda: [
        ".git", "__pycache__", "*.pyc", ".venv", "venv", "data", "outputs", "*.ipynb_checkpoints",
    ])
    volumes: dict[str, str] = field(default_factory=dict)  # mount path -> Modal Volume name
    modal_environment: str | None = None  # e.g. "main"


def load_config(path: str | Path, **overrides) -> RunConfig:
    """Load a YAML config. Relative paths inside it resolve against the file's folder."""
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text()) or {}
    known = {f.name for f in fields(RunConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown config keys in {path.name}: {sorted(unknown)}")
    raw.setdefault("name", path.stem)
    cfg = RunConfig(**raw)

    base = path.parent
    cfg.project_dir = str((base / cfg.project_dir).resolve())
    if cfg.requirements:
        req = (base / cfg.requirements).resolve()
        cfg.requirements = str(req) if req.exists() else None

    overrides = {k: v for k, v in overrides.items() if v is not None}
    return replace(cfg, **overrides)
