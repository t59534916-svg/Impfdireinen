"""LLM client abstraction for the insight layer.

* :class:`LLMClient` — the minimal protocol the generator depends on.
* :class:`AnthropicClient` — the real client (official ``anthropic`` SDK, Claude).
* :class:`MockLLMClient` — deterministic stub for tests / offline use.

Keeping the generator behind this protocol means the whole insight layer is
testable without a network call, and that the honesty guardrails can be exercised
against an adversarial mock that *tries* to overclaim.
"""
from __future__ import annotations

from typing import Callable, Optional, Protocol, runtime_checkable


class InsightLLMError(RuntimeError):
    """Raised when the LLM backend fails (network, auth, refusal)."""


@runtime_checkable
class LLMClient(Protocol):
    """Anything that can turn a (system, user) prompt pair into text."""

    def complete(self, system: str, user: str) -> str: ...


class AnthropicClient:
    """Real LLM backend using the official Anthropic SDK (Claude).

    Defaults to ``claude-opus-4-8``. Credentials and base URL resolve from the
    environment (``ANTHROPIC_API_KEY`` / ``ANTHROPIC_BASE_URL`` / a profile), so
    no secret is hard-coded.
    """

    def __init__(
        self,
        *,
        model: str = "claude-opus-4-8",
        max_tokens: int = 1500,
        effort: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.max_tokens = int(max_tokens)
        self.effort = effort
        self.timeout = float(timeout)

    def complete(self, system: str, user: str) -> str:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise InsightLLMError(
                "The 'anthropic' package is required for AnthropicClient. "
                "Install it (`pip install anthropic`) or pass a different LLMClient."
            ) from exc

        client = anthropic.Anthropic()
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self.effort is not None:
            kwargs["output_config"] = {"effort": self.effort}
        try:
            resp = client.with_options(timeout=self.timeout).messages.create(**kwargs)
        except anthropic.APIError as exc:
            raise InsightLLMError(f"Anthropic API error: {exc}") from exc

        if getattr(resp, "stop_reason", None) == "refusal":
            raise InsightLLMError("Model declined to respond (stop_reason='refusal').")
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        if not text.strip():
            raise InsightLLMError("Model returned empty text.")
        return text


class MockLLMClient:
    """Deterministic test/offline client.

    Pass a fixed string, or a callable ``(system, user) -> str`` to simulate
    behavior (including an adversarial model that tries to overclaim).
    """

    def __init__(self, response: "str | Callable[[str, str], str]") -> None:
        self._response = response

    def complete(self, system: str, user: str) -> str:
        if callable(self._response):
            return self._response(system, user)
        return self._response
