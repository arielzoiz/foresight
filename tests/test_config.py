"""Config loading fails loudly at startup, not on the first request."""

from __future__ import annotations

import pytest

from foresight.config import ForesightConfig, Runtime
from foresight.errors import ConfigError

BASE = {
    "models": {
        "target": {"served_name": "target-model", "model": "fake-target", "base_url": "http://x/v1"},
        "aux": {"served_name": "aux-model", "model": "fake-aux", "base_url": "http://x/v1"},
    },
    "aux_prompt": "{{ prompt }}",
    "builder": {"name": "template", "template": "{{ aux }}\n{{ prompt }}"},
}


@pytest.mark.asyncio
async def test_valid_config_builds_a_runtime():
    config = ForesightConfig(**BASE)
    runtime = Runtime(config)
    try:
        assert runtime.adapter.name == "generic"
        assert runtime.builder.name == "template"
        assert set(runtime.by_served_name) == {"target-model", "aux-model"}
    finally:
        await runtime.aclose()


@pytest.mark.asyncio
async def test_unknown_adapter_raises_at_construction():
    config = ForesightConfig(**{**BASE, "adapter": {"name": "does_not_exist"}})
    with pytest.raises(ConfigError, match="unknown adapter"):
        Runtime(config)


@pytest.mark.asyncio
async def test_unknown_builder_raises_at_construction():
    bad = {**BASE, "builder": {"name": "does_not_exist", "template": "{{ prompt }}"}}
    config = ForesightConfig(**bad)
    with pytest.raises(ConfigError, match="unknown builder"):
        Runtime(config)


@pytest.mark.asyncio
async def test_unknown_backend_raises_at_construction():
    bad = dict(BASE)
    bad["models"] = {
        **BASE["models"],
        "target": {**BASE["models"]["target"], "backend": "does_not_exist"},
    }
    config = ForesightConfig(**bad)
    with pytest.raises(ConfigError, match="unknown backend"):
        Runtime(config)


@pytest.mark.asyncio
async def test_target_and_aux_sharing_served_name_raises():
    bad = dict(BASE)
    bad["models"] = {
        "target": {"served_name": "same-name", "model": "a", "base_url": "http://x/v1"},
        "aux": {"served_name": "same-name", "model": "b", "base_url": "http://x/v1"},
    }
    config = ForesightConfig(**bad)
    with pytest.raises(ConfigError, match="share served_name"):
        Runtime(config)


def test_missing_model_role_raises():
    bad = {**BASE, "models": {"target": BASE["models"]["target"]}}  # no aux
    config = ForesightConfig(**bad)
    with pytest.raises(ConfigError, match="models.aux"):
        Runtime(config)


def test_bad_aux_prompt_template_raises():
    bad = {**BASE, "aux_prompt": "{{ nonexistent }}"}
    with pytest.raises(ConfigError):
        Runtime(ForesightConfig(**bad))


def test_bad_builder_template_raises():
    bad = {**BASE, "builder": {"name": "template", "template": "{{ nonexistent }}"}}
    with pytest.raises(ConfigError):
        Runtime(ForesightConfig(**bad))


def test_real_shipped_configs_load_without_error():
    for name in ("naive", "control", "canary"):
        from foresight.config import load_config

        config = load_config(f"configs/{name}.yaml")
        assert config.models["target"].served_name == "target-model"
