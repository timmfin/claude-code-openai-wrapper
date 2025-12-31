# Multi-CLI LLM Gateway Architecture

## Adapter design

The gateway introduces a small adapter layer (`src/llm_adapters.py`) that normalizes CLI-backed
LLM providers behind a single interface:

```
LLMCLIAdapter.start_session(session_id?: str) -> None
LLMCLIAdapter.resume_session(session_id: str) -> None
LLMCLIAdapter.send_prompt(prompt: PromptPayload) -> AsyncIterator[StreamChunk]
LLMCLIAdapter.close() -> None
```

Concrete adapters:

* `ClaudeCLIAdapter`
* `GeminiCLIAdapter`
* `CodexCLIAdapter`

Each adapter spawns its CLI via `asyncio.create_subprocess_exec` and **streams stdout incrementally**.
No adapter buffers the full response. The gateway routes requests by resolving the incoming
`model` field:

* `"claude" | "gemini" | "codex"` selects the backend explicitly.
* Any value that looks like a Claude model (e.g., `claude-*`) is treated as a Claude backend
  override and preserves backward compatibility.

The `src/llm_gateway.py` `LLMGateway` lazily creates adapters and validates CLI availability by
resolving CLI paths from either `$PATH` or explicit environment variables:

* `CLAUDE_CLI_PATH`
* `GEMINI_CLI_PATH`
* `CODEX_CLI_PATH`

> **Assumptions (documented inline in code):**
> * Claude Code CLI supports `claude --print` and an optional `--model <name>` flag.
> * Gemini/Codex CLIs accept prompts on STDIN and stream results to STDOUT.
> * Session continuity is handled at the gateway layer by replaying full conversation context,
>   unless a CLI exposes its own session mechanism later.

## Streaming normalization

CLI output is normalized into a common internal stream shape:

```
{
  "type": "delta" | "tool" | "error" | "done",
  "content": "string"
}
```

`BaseCLIAdapter` reads stdout in 1KB chunks, splits on newlines, and attempts to parse JSON
per line. JSON lines are mapped using heuristics for `content`, `text`, `delta`, `message`, and
`completion` fields. Non-JSON lines are treated as `delta` content. Noise (timestamps and
log-level prefixes) is stripped conservatively.

The OpenAI-compatible SSE stream is emitted from these normalized chunks (one `delta` per
event). Tool messages are currently logged and ignored, and errors are sent as SSE error payloads.

## Session management

Sessions are stored in `src/session_manager.py` with minimal metadata:

* `backend` (claude/gemini/codex)
* `cli_session_id` (reserved for future use)

Cross-backend reuse is prevented by validating the backend on session access. `conversation_id`
is accepted as an alias to `session_id`, and both map to the same session store.

## Prompt caching hooks

Hook plumbing lives in `PromptCacheHooks` (`src/llm_gateway.py`):

* `before_send(prompt, session)`
* `after_receive(chunk, session)`

These are no-ops by default and provide extension points for caching or analytics without forcing
any storage implementation.

## Adding another CLI

1. Add a new backend name in `src/constants.py`:
   * `SUPPORTED_BACKENDS`
   * `BACKEND_CLI_ENV_VARS`
   * `BACKEND_CLI_COMMANDS`
2. Create a new adapter in `src/llm_adapters.py` inheriting from `BaseCLIAdapter`.
   * Override `build_args`, `build_env`, or `noise_prefixes` as needed.
3. Register the adapter in `LLMGateway._create_adapter`.
4. Update any documentation or examples as desired.

## Example curl requests

Claude (default backend):

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude",
    "messages": [{"role": "user", "content": "Hello from Claude"}],
    "stream": true
  }'
```

Gemini:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini",
    "messages": [{"role": "user", "content": "Hello from Gemini"}],
    "stream": true
  }'
```

Codex:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "codex",
    "messages": [{"role": "user", "content": "Write a bash snippet to list files"}],
    "stream": true
  }'
```
