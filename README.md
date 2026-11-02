# modalize

Run any local training or fine-tuning project on [Modal](https://modal.com) GPUs and CPUs, driven by a single `config.yaml`.

Drop `config.yaml` next to your `train.py` and `requirements.txt`, then:

```bash
modalize run
```

Your code is uploaded, a container image is built from your requirements, and your command runs on the GPU you picked. Logs stream to your terminal or notebook. No Modal decorators, no changes to your training script.

## Why

| Without modalize | With modalize |
|---|---|
| Rewrite your script with `@app.function(...)` | Your script stays untouched |
| Re-learn Modal's image and volume APIs per project | One YAML file, same in every project |
| Hardcode GPU choice in code | Change `gpu:` in YAML or pass `--gpu H100` |

## Install

```bash
pip install git+https://github.com/<your-user>/modalize.git
modal setup        # one-time Modal login
```

Requires Python 3.10+ and a Modal account.

## Quickstart

```
my-project/
├── train.py
├── requirements.txt
└── config.yaml
```

```bash
cd my-project
modalize init     # writes a config.yaml template
modalize run
```

A minimal `config.yaml`:

```yaml
command: python train.py
gpu: L40S
```

A full one:

```yaml
command: python train.py --epochs 3 --out /data/output

# compute
gpu: H100                 # T4 | L40S | A100-40GB | A100-80GB | H100 | H100:2 | null (CPU only)
cpu: 8
memory: 32768             # MiB
timeout: 21600            # seconds, Modal max is 24h

# environment
python_version: "3.11"
requirements: requirements.txt
apt_packages: [git]
env:
  HF_HOME: /data/hf
  PYTHONUNBUFFERED: "1"
secrets: [huggingface]    # Modal secrets injected as env vars
volumes:
  /data: my-volume        # container path -> Modal Volume (created if missing)
```

## Configuration reference

| Key | Default | Description |
|---|---|---|
| `command` | `python train.py` | Shell command run in the container |
| `name` | config file name | Modal app name |
| `gpu` | none (CPU) | GPU type, optionally with count, e.g. `A100-80GB:4` |
| `cpu` | Modal default | CPU cores |
| `memory` | Modal default | RAM in MiB |
| `timeout` | `21600` | Max runtime in seconds |
| `python_version` | `3.11` | Python in the container |
| `requirements` | `requirements.txt` | Installed with pip; skipped if the file is missing |
| `apt_packages` | `[]` | System packages |
| `env` | `{}` | Environment variables |
| `secrets` | `[]` | Names of Modal Secrets |
| `volumes` | `{}` | Mount path to Modal Volume name |
| `project_dir` | `.` | Folder uploaded to the container, relative to `config.yaml` |
| `workdir` | `/workspace` | Where the project lands and the command runs |
| `ignore` | `.git`, `__pycache__`, `.venv`, `data`, `outputs`, ... | Patterns not uploaded |
| `modal_environment` | Modal default | Modal environment, e.g. `main` |

Relative paths in the config resolve against the config file's folder.

## CLI

```bash
modalize init                              # create config.yaml
modalize run                               # run ./config.yaml
modalize run other.yaml                    # use another config
modalize run --gpu A100-80GB --cpu 16 --memory 65536
modalize run --command "python train.py --lr 1e-5"
modalize run --timeout 7200
modalize run --detach                      # start and return; job keeps running on Modal
```

CLI flags override the YAML. The exit code of your command is the exit code of `modalize`. Ctrl+C terminates the remote job.

## Use from a notebook

Keep your notebook local and offload the heavy step:

```python
from modalize import run

run("config.yaml", gpu="H100")        # logs stream into the cell
job_id = run("config.yaml", detach=True)
```

## Data, checkpoints and secrets

```bash
# secrets (e.g. Hugging Face token)
modal secret create huggingface HF_TOKEN=hf_xxx

# upload data to a volume
modal volume put my-volume ./local_data /data

# download results
modal volume get my-volume /output ./output
```

Write checkpoints to a mounted volume path (e.g. `/data/output`). `modalize` flushes every volume when the command ends, so files persist. Setting `HF_HOME` inside a volume caches model downloads between runs.

## How it works

1. Reads `config.yaml` and applies CLI overrides.
2. Builds a Modal image: Debian + your Python, apt packages, `requirements.txt`, env vars. The project folder is added last, so code edits don't rebuild the dependency layers.
3. Starts a [Modal Sandbox](https://modal.com/docs/guide/sandbox) with the requested GPU, CPU, memory, volumes and secrets.
4. Runs your command with stderr merged into stdout, streams the output, runs `sync` on each volume, and returns the exit code.

## Troubleshooting

- **`config.yaml not found`**: run `modalize init` or pass the config path.
- **Large upload or slow start**: add big folders to `ignore`; datasets belong in a volume.
- **Missing package at runtime**: add it to `requirements.txt`; `apt_packages` for system libraries.
- **Gated model download fails**: create the `huggingface` secret and list it under `secrets`.
- **Job exceeds 24h**: Modal limits a sandbox to 24h; save checkpoints to a volume and resume.

## Development

```bash
git clone https://github.com/<your-user>/modalize.git
cd modalize
pip install -e ".[dev]"
pytest
```

A runnable sample lives in [example_project/](example_project/).

## License

MIT, see [LICENSE](LICENSE).
