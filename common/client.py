import os
import sys
import time
from dataclasses import dataclass, field


MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "qwen/qwen3-8b": 40960,
    "llama-3.1-70b-versatile": 128000,
    "llama-3.1-8b-instant": 128000,
}


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
    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str = "qwen/qwen3-8b",
        api_key: str = "not-needed",
        inject_no_think: bool = True,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.inject_no_think = inject_no_think
        self._client = None

    async def _ensure_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                base_url=self.base_url, api_key=self.api_key
            )

    def _inject_no_think(self, messages: list[dict]) -> list[dict]:
        if not self.inject_no_think:
            return messages
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

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs,
    ) -> LLMResponse:
        await self._ensure_client()
        messages = self._inject_no_think(messages)
        start = time.monotonic()
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
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
            usage=token_usage,
            latency_ms=elapsed,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )


def select_provider():
    from dotenv import load_dotenv
    load_dotenv()
    from rich.console import Console
    console = Console()
    console.print("[bold]Select provider:[/bold]")
    console.print("  [L] Local LM Studio (qwen/qwen3-8b, 40k context)")
    console.print("  [G] Groq API (llama-3.1-70b-versatile, 128k context)")
    choice = input("Choice [L/G]: ").strip().lower()
    if choice == "g":
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            console.print("[red]Error: GROQ_API_KEY not set in environment[/red]")
            sys.exit(1)
        model = os.environ.get("GROQ_MODEL", "llama-3.1-70b-versatile")
        client = LMStudioClient(
            base_url="https://api.groq.com/openai/v1",
            model=model,
            api_key=api_key,
            inject_no_think=False,
        )
        console.print(f"[green]Groq: {model}[/green]\n")
        return client, model
    client = LMStudioClient()
    console.print(f"[green]Local: qwen/qwen3-8b[/green]\n")
    return client, "qwen/qwen3-8b"


def get_token_budget(model_name: str, safety_margin: int = 2000) -> int:
    limit = MODEL_CONTEXT_LIMITS.get(model_name, 40960)
    return limit - safety_margin
