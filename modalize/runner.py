from __future__ import annotations

import shlex
import sys

import modal

from .config import RunConfig, load_config


def build_image(cfg: RunConfig) -> modal.Image:
    image = modal.Image.debian_slim(python_version=cfg.python_version)
    if cfg.apt_packages:
        image = image.apt_install(*cfg.apt_packages)
    if cfg.requirements:
        image = image.pip_install_from_requirements(cfg.requirements)
    if cfg.env:
        image = image.env(cfg.env)
    # added last so code changes don't invalidate the cached dependency layers
    return image.add_local_dir(cfg.project_dir, cfg.workdir, ignore=cfg.ignore)


def _wrapped_command(cfg: RunConfig) -> str:
    # merge stderr into stdout, keep the exit code, and flush volumes to storage at the end
    syncs = "; ".join(f"sync {shlex.quote(p)}" for p in cfg.volumes)
    return f"exec 2>&1; {cfg.command}\ncode=$?\n{syncs}\nexit $code"


def run(config: RunConfig | str, *, detach: bool = False, **overrides) -> int | str:
    """Run `config.command` in a Modal container.

    Streams logs and returns the exit code. With detach=True it returns the sandbox id
    immediately and the job keeps running on Modal.
    """
    cfg = load_config(config, **overrides) if isinstance(config, str) else config
    app = modal.App.lookup(cfg.name, create_if_missing=True, environment_name=cfg.modal_environment)

    volumes = {
        path: modal.Volume.from_name(name, create_if_missing=True, environment_name=cfg.modal_environment)
        for path, name in cfg.volumes.items()
    }
    secrets = [modal.Secret.from_name(s, environment_name=cfg.modal_environment) for s in cfg.secrets]

    print(f"[modalize] building image + starting {cfg.name!r} (gpu={cfg.gpu}, cpu={cfg.cpu}, mem={cfg.memory})")
    sb = modal.Sandbox.create(
        "bash", "-c", _wrapped_command(cfg),
        app=app,
        image=build_image(cfg),
        gpu=cfg.gpu,
        cpu=cfg.cpu,
        memory=cfg.memory,
        timeout=cfg.timeout,
        workdir=cfg.workdir,
        volumes=volumes,
        secrets=secrets,
    )
    if detach:
        print(f"[modalize] detached, sandbox id: {sb.object_id}")
        return sb.object_id

    try:
        for line in sb.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
        sb.wait()
    except KeyboardInterrupt:
        print("\n[modalize] interrupted, terminating sandbox")
        sb.terminate()
        return 130
    print(f"[modalize] finished with exit code {sb.returncode}")
    return sb.returncode
