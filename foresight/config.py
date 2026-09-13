"""YAML -> objects. Everything is wired here, once, at startup.

Two principles:

* **Fail loudly at startup.** An unknown adapter, builder or backend raises
  before the server binds a port, not on the first request. Misconfiguration
  should be a five-second failure, not an experiment that quietly ran the wrong
  arm for six hours.
* **Injection, not lookup.** The adapter receives its backend, spec and prompt
  here; nothing reaches for globals later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import httpx
import yaml
from jinja2 import StrictUndefined, Template, TemplateSyntaxError
from pydantic import BaseModel, Field, ValidationError

from .adapters import ADAPTERS, CallerAdapter
from .backends import BACKENDS, Backend
from .builders import BUILDERS, Builder
from .errors import ConfigError
from .llm import ModelSpec
from .pipeline import Pipeline, SessionStore
from .stages import AuxStage, Stage
from .trace import TraceWriter


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class AdapterConfig(BaseModel):
    name: str = "generic"


class BuilderConfig(BaseModel):
    name: str = "template"
    template: str = "{{ prompt }}"


class SessionConfig(BaseModel):
    ttl_s: float = 3600.0


class TraceConfig(BaseModel):
    """Where to append trace records. ``path: null`` (the default) disables it.

    Opting out is a config value rather than a NoopWriter class, same as
    ``guard: null``.
    """

    path: str | None = None


class ForesightConfig(BaseModel):
    """The whole configuration, validated before anything is constructed."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    models: dict[str, ModelSpec]
    adapter: AdapterConfig = Field(default_factory=AdapterConfig)
    aux_prompt: str = "{{ prompt }}"
    builder: BuilderConfig = Field(default_factory=BuilderConfig)
    stages: list[Literal["aux"]] = Field(default_factory=lambda: ["aux"])
    session: SessionConfig = Field(default_factory=SessionConfig)
    trace: TraceConfig = Field(default_factory=TraceConfig)
    guard: Any = None  # milestone 1 ships no implementation; see guards.py


class Runtime:
    """Live objects built from a config: what the server actually uses."""

    def __init__(self, config: ForesightConfig, http_client: httpx.AsyncClient | None = None) -> None:
        self.config = config
        self.target_spec = _require_model(config, "target")
        self.aux_spec = _require_model(config, "aux")

        # Injectable so tests can run the whole pipeline over ASGITransport,
        # in-process, with no real sockets -- see tests/test_e2e.py.
        self._client = http_client or httpx.AsyncClient()
        self.backends: dict[str, Backend] = {
            name: cls(self._client) for name, cls in BACKENDS.items()
        }
        self.target_backend = self._backend_for(self.target_spec)
        self.aux_backend = self._backend_for(self.aux_spec)

        self.adapter = _build_adapter(config, self.aux_backend, self.aux_spec)
        self.builder = _build_builder(config)
        self.tracer = _build_tracer(config)
        self.store = SessionStore(ttl_s=config.session.ttl_s)
        self.pipeline = Pipeline(
            adapter=self.adapter,
            stages=_build_stages(config, self.adapter, self.aux_spec),
            builder=self.builder,
            store=self.store,
        )

        # Routing is by served model name, so two specs may not claim the same
        # one -- the request would be ambiguous and one arm would silently win.
        if self.target_spec.served_name == self.aux_spec.served_name:
            raise ConfigError(
                f"target and aux share served_name {self.target_spec.served_name!r}; "
                "routing is by served name, so they must differ"
            )

        self.by_served_name: dict[str, tuple[str, ModelSpec, Backend]] = {
            self.target_spec.served_name: ("target", self.target_spec, self.target_backend),
            self.aux_spec.served_name: ("aux", self.aux_spec, self.aux_backend),
        }

    def _backend_for(self, spec: ModelSpec) -> Backend:
        try:
            return self.backends[spec.backend]
        except KeyError:
            raise ConfigError(
                f"unknown backend {spec.backend!r} for model {spec.served_name!r}; "
                f"known: {sorted(BACKENDS)}"
            ) from None

    async def aclose(self) -> None:
        await self._client.aclose()


def load_config(path: str | Path) -> ForesightConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    try:
        return ForesightConfig(**raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid config {path}: {exc}") from exc


def build_runtime(path: str | Path, http_client: httpx.AsyncClient | None = None) -> Runtime:
    return Runtime(load_config(path), http_client=http_client)


def _require_model(config: ForesightConfig, role: str) -> ModelSpec:
    try:
        return config.models[role]
    except KeyError:
        raise ConfigError(
            f"config must define models.{role}; got {sorted(config.models)}"
        ) from None


def _build_adapter(
    config: ForesightConfig, aux_backend: Backend, aux_spec: ModelSpec
) -> CallerAdapter:
    try:
        cls = ADAPTERS[config.adapter.name]
    except KeyError:
        raise ConfigError(
            f"unknown adapter {config.adapter.name!r}; known: {sorted(ADAPTERS)}"
        ) from None

    _check_template(config.aux_prompt, "aux_prompt", {"prompt", "system"})
    return cls(aux_backend=aux_backend, aux_spec=aux_spec, aux_prompt=config.aux_prompt)


def _build_builder(config: ForesightConfig) -> Builder:
    try:
        cls = BUILDERS[config.builder.name]
    except KeyError:
        raise ConfigError(
            f"unknown builder {config.builder.name!r}; known: {sorted(BUILDERS)}"
        ) from None

    if cls.name == "template":
        _check_template(config.builder.template, "builder.template", {"prompt", "aux"})
        return cls(config.builder.template)
    return cls()


def _build_stages(
    config: ForesightConfig, adapter: CallerAdapter, aux_spec: ModelSpec
) -> list[Stage]:
    return [AuxStage(adapter, aux_spec) for name in config.stages if name == "aux"]


def _build_tracer(config: ForesightConfig) -> TraceWriter | None:
    """None when tracing is off, otherwise a writer proven to work.

    The preflight matters: a failed trace write fails the request (see
    errors.TraceFailure), so an unwritable path must surface here -- as a
    five-second startup failure -- rather than killing the first request of a
    six-hour run.
    """
    path = config.trace.path
    if not path:
        return None

    writer = TraceWriter(path)
    try:
        writer.preflight()
    except OSError as exc:
        raise ConfigError(f"trace.path {path!r} is not writable: {exc}") from exc
    return writer


def _check_template(source: str, where: str, allowed: set[str]) -> None:
    """Render once with dummy values so a bad template fails at startup.

    StrictUndefined turns a typo like {{ promt }} into an error here rather than
    a silently empty substitution on every request for the rest of the run.
    """
    try:
        Template(source, undefined=StrictUndefined).render(**{k: "x" for k in allowed})
    except TemplateSyntaxError as exc:
        raise ConfigError(f"{where} is not valid Jinja: {exc}") from exc
    except Exception as exc:  # undefined variable, etc.
        raise ConfigError(
            f"{where} failed to render (available variables: {sorted(allowed)}): {exc}"
        ) from exc
