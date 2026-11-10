import json

import pytest

from g1_imitation.core.constants import CONFIG_PATH, load_config


def test_local_config_selection_and_explicit_precedence(monkeypatch, tmp_path):
    local = tmp_path / "local.json"
    local.write_text(json.dumps({"runtime": {"python": "local-python"}}), encoding="utf-8")
    monkeypatch.setenv("G1_IMITATION_CONFIG", str(local))
    assert load_config()["runtime"]["python"] == "local-python"
    assert load_config(CONFIG_PATH) == json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    monkeypatch.setenv("G1_IMITATION_CONFIG", str(tmp_path / "missing.json"))
    with pytest.raises(FileNotFoundError):
        load_config()
