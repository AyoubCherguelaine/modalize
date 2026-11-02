import pytest

from modalize import load_config


def test_defaults_and_relative_paths(tmp_path):
    (tmp_path / "requirements.txt").write_text("torch\n")
    (tmp_path / "config.yaml").write_text("gpu: T4\n")
    cfg = load_config(tmp_path / "config.yaml")
    assert cfg.command == "python train.py"
    assert cfg.gpu == "T4"
    assert cfg.project_dir == str(tmp_path)
    assert cfg.requirements == str(tmp_path / "requirements.txt")


def test_missing_requirements_is_skipped(tmp_path):
    (tmp_path / "config.yaml").write_text("gpu: T4\n")
    assert load_config(tmp_path / "config.yaml").requirements is None


def test_overrides_ignore_none(tmp_path):
    (tmp_path / "config.yaml").write_text("gpu: T4\ncpu: 2\n")
    cfg = load_config(tmp_path / "config.yaml", gpu="H100", cpu=None)
    assert (cfg.gpu, cfg.cpu) == ("H100", 2)


def test_unknown_key_rejected(tmp_path):
    (tmp_path / "config.yaml").write_text("gpuu: T4\n")
    with pytest.raises(ValueError, match="gpuu"):
        load_config(tmp_path / "config.yaml")
