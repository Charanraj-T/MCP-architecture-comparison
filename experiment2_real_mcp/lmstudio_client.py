import time
from dataclasses import dataclass, field

@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

@dataclass
class LLMResponse:
    content: str
    usage: TokenUsage
    latency_ms: float = 0.0
    raw: dict = field(default_factory=dict)

class LMStudioClient:
    def __init__(self, base_url: str = "http://localhost:1234/v1", model: str = "qwen/qwen3-8b"):
        self.base_url = base_url
        self.model = model
        self._client = None

    async def _ensure_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(base_url=self.base_url, api_key="not-needed")

    def _inject_no_think(self, messages: list[dict]) -> list[dict]:
        result = []
        no_think_system = {"role": "system", "content": "/no_think"}
        has_no_think = False
        for m in messages:
            if m.get("role") == "system" and "/no_think" in (m.get("content", "") or ""):
                has_no_think = True
            result.append(m)
        if not has_no_think:
            result.insert(0, no_think_system)
        return result

    async def chat(self, messages: list[dict], temperature: float = 0.1, max_tokens: int = 4096, **kwargs) -> LLMResponse:
        await self._ensure_client()
        messages = self._inject_no_think(messages)
        start = time.monotonic()
        response = await self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens, **kwargs,
        )
        elapsed = (time.monotonic() - start) * 1000
        usage = response.usage
        reasoning = 0
        if usage and usage.completion_tokens_details:
            reasoning = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
        token_usage = TokenUsage(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            reasoning_tokens=reasoning,
            total_tokens=usage.total_tokens if usage else 0,
        )
        return LLMResponse(
            content=response.choices[0].message.content or "",
            usage=token_usage, latency_ms=elapsed,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )
