import logging
from dataclasses import dataclass
from typing import Dict, Optional

from src.constants import (
    CLAUDE_MODELS,
    DEFAULT_BACKEND,
    DEFAULT_CLAUDE_MODEL,
    SUPPORTED_BACKENDS,
)
from src.llm_adapters import (
    ClaudeCLIAdapter,
    CodexCLIAdapter,
    GeminiCLIAdapter,
    LLMCLIAdapter,
    PromptPayload,
    resolve_cli_path,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackendSelection:
    backend: str
    model: Optional[str]


class PromptCacheHooks:
    """Hook plumbing for prompt caching integrations."""

    def before_send(self, prompt: PromptPayload, session: Dict[str, Optional[str]]) -> None:
        return None

    def after_receive(self, chunk: Dict[str, str], session: Dict[str, Optional[str]]) -> None:
        return None


def resolve_backend(model: Optional[str]) -> BackendSelection:
    if not model:
        backend = DEFAULT_BACKEND
        resolved_model = DEFAULT_CLAUDE_MODEL if backend == "claude" else None
        return BackendSelection(backend=backend, model=resolved_model)

    if model in SUPPORTED_BACKENDS:
        backend = model
        resolved_model = DEFAULT_CLAUDE_MODEL if backend == "claude" else None
        return BackendSelection(backend=backend, model=resolved_model)

    if model in CLAUDE_MODELS or model.startswith("claude-"):
        return BackendSelection(backend="claude", model=model)

    raise ValueError(
        f"Unknown model '{model}'. Supported backends: {', '.join(SUPPORTED_BACKENDS)}."
    )


class LLMGateway:
    def __init__(self, hooks: Optional[PromptCacheHooks] = None):
        self._adapters: Dict[str, LLMCLIAdapter] = {}
        self.hooks = hooks or PromptCacheHooks()

    def get_adapter(self, backend: str) -> LLMCLIAdapter:
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend '{backend}'. Supported backends: {SUPPORTED_BACKENDS}."
            )

        if backend not in self._adapters:
            cli_path = resolve_cli_path(backend)
            adapter = self._create_adapter(backend, cli_path)
            self._adapters[backend] = adapter

        return self._adapters[backend]

    def _create_adapter(self, backend: str, cli_path: str) -> LLMCLIAdapter:
        if backend == "claude":
            return ClaudeCLIAdapter(cli_path=cli_path)
        if backend == "gemini":
            return GeminiCLIAdapter(cli_path=cli_path)
        if backend == "codex":
            return CodexCLIAdapter(cli_path=cli_path)
        raise ValueError(f"Unsupported backend: {backend}")

    def backend_status(self) -> Dict[str, str]:
        status = {}
        for backend in SUPPORTED_BACKENDS:
            try:
                resolve_cli_path(backend)
                status[backend] = "available"
            except FileNotFoundError as exc:
                status[backend] = f"missing: {exc}"
        return status
