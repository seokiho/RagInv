"""
LLM generator — Ollama only (no LangChain, no cloud API cost).

Talks to Ollama's HTTP API directly with httpx.
Supports both sync and async usage.

Quick test:
    python -c "from core.generator import OllamaGenerator; g=OllamaGenerator(); print(g.chat([{'role':'user','content':'안녕'}]))"
"""
from __future__ import annotations

from typing import AsyncGenerator, List
import httpx


class OllamaGenerator:
    """
    Thin wrapper around Ollama /api/chat endpoint.
    Model must already be pulled:  ollama pull exaone3.5:7.8b
    # ollama pull llama3.1:8b
    """

    _STOP_TOKENS = {
        "exaone": ["[|endofturn|]", "</s>", "[|system|]", "[|user|]"],
        "qwen3":  ["<|im_end|>", "<|endoftext|>"],
        "llama":  ["<|eot_id|>", "</s>"],
    }

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "exaone3.5:7.8b",
        timeout: float = 180.0,
        max_tokens: int = 2048,
        num_ctx: int = 8192,
        repeat_penalty: float = 1.15,
        repeat_last_n: int = 128,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        self.repeat_penalty = repeat_penalty
        self.repeat_last_n = repeat_last_n

        m = model.lower()
        if "qwen3" in m:
            self._stop = self._STOP_TOKENS["qwen3"]
            self._think = False   # Qwen3 thinking 모드 비활성화
        elif "exaone" in m:
            self._stop = self._STOP_TOKENS["exaone"]
            self._think = None
        else:
            self._stop = self._STOP_TOKENS["llama"]
            self._think = None

    def _options(self, temperature: float) -> dict:
        opts = {
            "temperature": temperature,
            "num_ctx": self.num_ctx,
            "num_predict": self.max_tokens,
            "repeat_penalty": self.repeat_penalty,
            "repeat_last_n": self.repeat_last_n,
            "stop": self._stop,
        }
        if self._think is not None:
            opts["think"] = self._think
        return opts

    # ── Sync ────────────────────────────────────────────────────────────
    def chat(self, messages: List[dict], temperature: float = 0.0) -> str:
        """
        Synchronous chat completion.
        messages = [{"role": "system"|"user"|"assistant", "content": "..."}]
        """
        resp = httpx.post(
            f"{self.host}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False,
                  "options": self._options(temperature)},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def is_alive(self) -> bool:
        """Check if Ollama server is running."""
        try:
            r = httpx.get(f"{self.host}/api/tags", timeout=3.0)
            return r.status_code == 200
        except Exception:
            return False

    # ── Async (for FastAPI streaming) ───────────────────────────────────
    async def astream(
        self, messages: List[dict], temperature: float = 0.0
    ) -> AsyncGenerator[str, None]:
        """
        Async streaming generator — yields token strings one by one.
        Use with FastAPI StreamingResponse.
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.host}/api/chat",
                json={"model": self.model, "messages": messages, "stream": True,
                      "options": self._options(temperature)},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    import json
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break

    async def achat(self, messages: List[dict], temperature: float = 0.0) -> str:
        """Async non-streaming chat (collects full response)."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.host}/api/chat",
                json={"model": self.model, "messages": messages, "stream": False,
                      "options": self._options(temperature)},
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
