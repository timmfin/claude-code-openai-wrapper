import asyncio
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Iterable, Optional, Protocol

from src.auth import auth_manager
from src.constants import BACKEND_CLI_COMMANDS, BACKEND_CLI_ENV_VARS

logger = logging.getLogger(__name__)


StreamChunk = Dict[str, str]


@dataclass
class PromptPayload:
    prompt: str
    system_prompt: Optional[str]
    model: Optional[str]
    session_id: Optional[str]


class LLMCLIAdapter(Protocol):
    async def start_session(self, session_id: Optional[str] = None) -> None:
        ...

    async def resume_session(self, session_id: str) -> None:
        ...

    async def send_prompt(self, prompt: PromptPayload) -> AsyncIterator[StreamChunk]:
        ...

    async def close(self) -> None:
        ...


def resolve_cli_path(backend: str) -> str:
    env_var = BACKEND_CLI_ENV_VARS.get(backend)
    if not env_var:
        raise ValueError(f"Unsupported backend: {backend}")

    explicit_path = os.getenv(env_var)
    if explicit_path:
        if not os.path.exists(explicit_path):
            raise FileNotFoundError(
                f"{backend} CLI not found at {explicit_path}. Set {env_var} to a valid path."
            )
        return explicit_path

    command = BACKEND_CLI_COMMANDS.get(backend, backend)
    resolved = shutil.which(command)
    if not resolved:
        raise FileNotFoundError(
            f"{backend} CLI not found in PATH. Install '{command}' or set {env_var}."
        )
    return resolved


class BaseCLIAdapter:
    backend: str = "base"
    noise_prefixes: Iterable[str] = ()
    noise_patterns: Iterable[re.Pattern[str]] = (
        re.compile(r"^\s*\[[0-9:. TZ-]+\]\s+\w+"),
        re.compile(r"^\s*(INFO|DEBUG|WARN|WARNING|ERROR)\b"),
    )

    def __init__(self, cli_path: str, cwd: Optional[str] = None):
        self.cli_path = cli_path
        self.cwd = cwd

    async def start_session(self, session_id: Optional[str] = None) -> None:
        return None

    async def resume_session(self, session_id: str) -> None:
        return None

    async def close(self) -> None:
        return None

    def build_args(self, payload: PromptPayload) -> Iterable[str]:
        return []

    def build_env(self) -> Dict[str, str]:
        return dict(os.environ)

    def format_prompt(self, payload: PromptPayload) -> str:
        if payload.system_prompt:
            return f"{payload.system_prompt}\n\n{payload.prompt}"
        return payload.prompt

    async def send_prompt(self, prompt: PromptPayload) -> AsyncIterator[StreamChunk]:
        args = [self.cli_path, *self.build_args(prompt)]
        env = self.build_env()
        formatted_prompt = self.format_prompt(prompt)

        logger.info("Launching %s CLI: %s", self.backend, " ".join(args))
        process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd,
            env=env,
        )

        if process.stdin:
            process.stdin.write(formatted_prompt.encode("utf-8"))
            await process.stdin.drain()
            process.stdin.close()

        stderr_task = asyncio.create_task(self._collect_stream(process.stderr))

        async for chunk in self._stream_stdout(process.stdout):
            yield chunk

        await process.wait()
        stderr_output = await stderr_task
        if stderr_output:
            yield {"type": "error", "content": stderr_output}
        yield {"type": "done", "content": ""}

    async def _collect_stream(self, stream: Optional[asyncio.StreamReader]) -> str:
        if not stream:
            return ""
        data = await stream.read()
        return data.decode("utf-8", errors="ignore").strip()

    async def _stream_stdout(
        self, stream: Optional[asyncio.StreamReader]
    ) -> AsyncIterator[StreamChunk]:
        if not stream:
            return

        buffer = ""
        while True:
            data = await stream.read(1024)
            if not data:
                break
            buffer += data.decode("utf-8", errors="ignore")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                for chunk in self._normalize_line(line):
                    yield chunk

        if buffer.strip():
            for chunk in self._normalize_line(buffer):
                yield chunk

    def _normalize_line(self, line: str) -> Iterable[StreamChunk]:
        stripped = line.strip()
        if not stripped:
            return []

        if any(stripped.startswith(prefix) for prefix in self.noise_prefixes):
            return []

        for pattern in self.noise_patterns:
            if pattern.search(stripped):
                return []

        if stripped.startswith("data:"):
            stripped = stripped[len("data:") :].strip()

        if stripped.startswith("{") or stripped.startswith("["):
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                return [{"type": "delta", "content": stripped}]
            return list(self._normalize_json(payload))

        return [{"type": "delta", "content": stripped}]

    def _normalize_json(self, payload: object) -> Iterable[StreamChunk]:
        if isinstance(payload, list):
            for item in payload:
                yield from self._normalize_json(item)
            return

        if not isinstance(payload, dict):
            return [{"type": "delta", "content": str(payload)}]

        if payload.get("type") in {"done", "end"} or payload.get("event") == "done":
            return [{"type": "done", "content": ""}]

        if "error" in payload:
            error = payload["error"]
            if isinstance(error, dict):
                message = error.get("message") or json.dumps(error)
            else:
                message = str(error)
            return [{"type": "error", "content": message}]

        if payload.get("type") in {"tool", "tool_use"}:
            content = payload.get("content") or payload.get("name") or json.dumps(payload)
            return [{"type": "tool", "content": str(content)}]

        text = self._extract_text(payload)
        if text:
            return [{"type": "delta", "content": text}]

        return []

    def _extract_text(self, payload: Dict[str, object]) -> str:
        if "delta" in payload and isinstance(payload["delta"], dict):
            delta = payload["delta"]
            if "content" in delta:
                return str(delta["content"])

        if "content" in payload:
            content = payload["content"]
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        parts.append(part)
                return "".join(parts)

        if "text" in payload:
            return str(payload["text"])

        if "message" in payload and isinstance(payload["message"], dict):
            return self._extract_text(payload["message"])

        if "completion" in payload:
            return str(payload["completion"])

        return ""


class ClaudeCLIAdapter(BaseCLIAdapter):
    backend = "claude"

    def build_args(self, payload: PromptPayload) -> Iterable[str]:
        args = ["--print"]
        if payload.model:
            # Assumes Claude Code CLI supports --model <name>
            args += ["--model", payload.model]
        return args

    def build_env(self) -> Dict[str, str]:
        env = dict(os.environ)
        env.update(auth_manager.get_claude_code_env_vars())
        return env


class GeminiCLIAdapter(BaseCLIAdapter):
    backend = "gemini"
    noise_prefixes = ("gemini:",)


class CodexCLIAdapter(BaseCLIAdapter):
    backend = "codex"
