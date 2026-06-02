import asyncio
import os
import sys
import time
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "qwen/qwen3-8b": 32768,
    "llama-3.3-70b-versatile": 131072,
    "llama-3.1-8b-instant": 131072,
    "mixtral-8x7b-32768": 32768,
    "gemma2-9b-it": 8192,
    "llama3.1:8b": 131072,
    "llama3.1:70b": 131072,
    "llama3.2:1b": 131072,
    "llama3.2:3b": 131072,
    "qwen2.5:7b": 32768,
    "qwen2.5:14b": 32768,
    "qwen2.5:32b": 32768,
    "qwen2.5:72b": 32768,
    "mistral:7b": 32768,
    "mixtral:8x7b": 32768,
    "phi3:mini": 131072,
    "phi3:medium": 131072,
    "deepseek-coder:6.7b": 16384,
    "deepseek-coder:33b": 16384,
    "codellama:7b": 16384,
    "codellama:34b": 16384,
    "gemma2:2b": 8192,
    "gemma2:9b": 8192,
    "nemotron-mini:4b": 4096,
    "orca-mini:3b": 8192,
    "orca-mini:7b": 8192,
    "starcoder2:7b": 16384,
    "codegemma:7b": 8192,
    "command-r:35b": 131072,
    "command-r-plus:104b": 131072,
    "llava:7b": 4096,
    "llava:13b": 4096,
    "llava:34b": 4096,
    "llama-3.2-90b-vision-preview": 131072,
    "llama-3.2-11b-vision-preview": 131072,
    "google/gemini-2.0-flash-001": 1048576,
    "google/gemini-2.0-flash-lite-preview": 1048576,
    "meta-llama/llama-3.3-70b-instruct": 131072,
    "mistralai/mistral-small-3.1-24b-instruct": 131072,
    "deepseek/deepseek-chat": 65536,
    "qwen/qwen-2.5-72b-instruct": 32768,
    "gpt-4o-mini": 131072,
    "gpt-4o": 131072,
    "openai/gpt-4o": 128000,
    "openai/gpt-4o-mini": 128000,
    "gpt-5.3-codex": 128000,
    "claude-sonnet-4-5": 200000,
}


PROVIDERS = {
    "L": {
        "name": "Local LM Studio",
        "base_url": "http://localhost:1234/v1",
        "api_key": "not-needed",
        "inject_no_think": True,
        "models": ["qwen/qwen3-8b"],
    },
}


def _get_env_config():
    if os.environ.get("LLM_BASE_URL"):
        api_key = (
            os.environ.get("LLM_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("GROQ_API_KEY")
            or os.environ.get("GITHUB_TOKEN")
            or ""
        )
        return {
            "base_url": os.environ["LLM_BASE_URL"],
            "api_key": api_key,
            "model": os.environ.get("LLM_MODEL", "default"),
            "inject_no_think": os.environ.get("LLM_INJECT_NO_THINK", "").lower() in ("1", "true"),
        }
    return None


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
    error: str = ""


class OpenAIClient:
    # Token bucket rate limiter (class-level, shared across all instances)
    _rate_limit_window: list[tuple[float, int]] = []  # (timestamp, prompt_tokens)
    _rate_limit_max_tokens = 5500  # per 60s sliding window
    _rate_limit_window_secs = 60

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

    async def _rate_limit(self, estimated_prompt_tokens: int = 0):
        now = time.monotonic()
        cutoff = now - self._rate_limit_window_secs
        # Prune old entries
        OpenAIClient._rate_limit_window = [
            (ts, tok) for ts, tok in OpenAIClient._rate_limit_window if ts > cutoff
        ]
        window_tokens = sum(tok for _, tok in OpenAIClient._rate_limit_window)
        needed = window_tokens + estimated_prompt_tokens
        if needed > self._rate_limit_max_tokens:
            # Need to wait until enough tokens fall out of the window
            # Sort entries by timestamp (should already be sorted, but be safe)
            entries = sorted(OpenAIClient._rate_limit_window)
            # Find how many seconds until enough tokens drain
            drain_needed = needed - self._rate_limit_max_tokens
            drained = 0
            wait = 0
            for ts, tok in entries:
                drained += tok
                if drained >= drain_needed:
                    wait = (ts + self._rate_limit_window_secs) - now
                    break
            if wait > 0:
                wait = min(wait, self._rate_limit_window_secs)  # cap at window size
                await asyncio.sleep(wait)

    @staticmethod
    def _parse_retry_delay(err_str: str) -> float:
        import re
        # Match "Please retry in Xs" or "retryDelay": "Xs"
        m = re.search(r'retry in\s*([\d.]+)\s*s', err_str, re.I)
        if m:
            return float(m.group(1))
        m = re.search(r'retryDelay["\']?\s*:\s*["\']?([\d.]+)s?', err_str)
        if m:
            return float(m.group(1))
        return 0.0

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 4096,
        request_timeout: int = 120,

        **kwargs,
    ) -> LLMResponse:
        await self._ensure_client()
        messages = self._inject_no_think(messages)
        # Estimate prompt tokens for rate limiting (rough: 4 chars per token)
        est_prompt = sum(len(m.get("content", "")) for m in messages) // 4
        await self._rate_limit(est_prompt)
        for retry in range(5):
            start = time.monotonic()
            try:
                response = await asyncio.wait_for(
                    self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    ),
                    timeout=request_timeout,
                )
                break  # success
            except asyncio.TimeoutError:
                elapsed = (time.monotonic() - start) * 1000
                return LLMResponse(content="", usage=TokenUsage(), latency_ms=elapsed, error="Request timed out")
            except Exception as e:
                err_str = f"{type(e).__name__}: {e}"
                if "429" in err_str or "RateLimit" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    delay = self._parse_retry_delay(err_str)
                    if delay <= 0:
                        delay = 10.0 * (retry + 1)
                    await asyncio.sleep(delay)
                    continue
                elapsed = (time.monotonic() - start) * 1000
                return LLMResponse(content="", usage=TokenUsage(), latency_ms=elapsed, error=err_str)
        else:
            elapsed = (time.monotonic() - start) * 1000
            return LLMResponse(content="", usage=TokenUsage(), latency_ms=elapsed, error=f"Rate limited after 5 retries: {err_str}")
        elapsed = (time.monotonic() - start) * 1000
        if not response.choices:
            return LLMResponse(content="", usage=TokenUsage(), latency_ms=elapsed, error="Empty response: no choices returned")
        usage = response.usage
        # Track actual prompt tokens for rate limiting
        if usage and usage.prompt_tokens:
            now = time.monotonic()
            OpenAIClient._rate_limit_window.append((now, usage.prompt_tokens))
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


LMStudioClient = OpenAIClient


def _discover_ollama_models() -> list[str]:
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return []
        lines = result.stdout.strip().split("\n")
        if len(lines) < 2:
            return []
        models = []
        for line in lines[1:]:
            parts = line.split()
            if parts:
                models.append(parts[0])
        return models
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def _pick_ollama_model(console, models: list[str]) -> str:
    if len(models) == 1:
        return models[0]
    console.print("[bold]Available Ollama models:[/bold]")
    for i, m in enumerate(models, 1):
        size_info = ""
        try:
            result = subprocess.run(
                ["ollama", "show", m], capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.split("\n"):
                if "size" in line.lower():
                    size_info = f" ({line.split()[-1]})"
                    break
        except Exception:
            pass
        console.print(f"  [{i}] {m}{size_info}")
    choice = input(f"Select model [1-{len(models)}]: ").strip()
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            return models[idx]
    except ValueError:
        pass
    return models[0]


def select_provider():
    from dotenv import load_dotenv
    load_dotenv()
    from rich.console import Console
    console = Console()

    env_config = _get_env_config()
    if env_config:
        provider_info = env_config.get("provider", "generic")
        model = env_config.get("model", "unknown")
        
        # Handle Azure clients (LangChain-based)
        if "client" in env_config and provider_info.startswith("azure"):
            console.print(f"[green]LLM from env ({provider_info}): {model}[/green]")
            return env_config["client"], model
        
        # Handle generic OpenAI-compatible clients
        api_key = env_config.get("api_key") or "not-needed"
        base_url = env_config.get("base_url", "")
        if base_url:
            console.print(f"[green]LLM from env ({provider_info}): {base_url} / {model}[/green]")
            return OpenAIClient(
                base_url=base_url,
                model=model,
                api_key=api_key,
                inject_no_think=env_config.get("inject_no_think", False),
            ), model

    console.print("[bold]Select provider:[/bold]")
    console.print("  [L] Local LM Studio  (qwen/qwen3-8b, http://localhost:1234/v1)")
    console.print("  [O] Local Ollama     (auto-detect models, http://localhost:11434/v1)")
    console.print("  [A] Azure OpenAI     (needs AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT)")
    console.print("  [B] Azure Anthropic  (needs AZURE_ANTHROPIC_API_KEY + AZURE_ANTHROPIC_ENDPOINT)")
    console.print("  [G] Groq API         (llama-3.3-70b-versatile, needs GROQ_API_KEY)")
    console.print("  [R] OpenRouter       (auto model selection, needs OPENROUTER_API_KEY)")
    console.print("  [H] GitHub Models    (gpt-4o-mini, needs GITHUB_TOKEN)")
    console.print("  [C] Cloudflare AI    (free tier, needs CF_API_KEY + CF_ACCOUNT_ID)")
    console.print("  [M] Custom URL       (any OpenAI-compatible endpoint)")

    choice = input("Choice [L/O/A/B/G/R/H/C/M]: ").strip().lower()

    if choice == "o":
        models = _discover_ollama_models()
        if not models:
            console.print("[yellow]Ollama not found or no models. Check 'ollama list'.[/yellow]")
            console.print("[yellow]Falling back to default model: llama3.2:3b[/yellow]")
            models = ["llama3.2:3b"]
        model = _pick_ollama_model(console, models)
        client = OpenAIClient(
            base_url="http://localhost:11434/v1",
            model=model,
            api_key="not-needed",
            inject_no_think=False,
        )
        console.print(f"[green]Ollama: {model}[/green]\n")
        return client, model

    if choice == "a":
        from common.langchain_clients import LangChainAzureOpenAIClient
        
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.3-codex")
        
        # Prompt if not in env, or always prompt when called again with addon model
        if not api_key or not endpoint:
            from getpass import getpass
            console.print("[bold]Azure OpenAI Configuration:[/bold]")
            endpoint = input("AZURE_OPENAI_ENDPOINT (e.g., https://xxxx.openai.azure.com): ").strip() or endpoint
            api_key = getpass("AZURE_OPENAI_API_KEY: ").strip() or api_key
        deployment = input(f"AZURE_OPENAI_DEPLOYMENT_NAME (e.g., gpt-5.3-codex, Ministral-3B) [{deployment}]: ").strip() or deployment
        
        if not api_key or not endpoint:
            console.print("[red]Error: Azure OpenAI credentials required[/red]")
            sys.exit(1)
        
        # Store credentials on the client instance, not in os.environ
        os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = deployment
        
        client = LangChainAzureOpenAIClient()
        console.print(f"[green]Azure OpenAI: {deployment}[/green]\n")
        return client, deployment

    if choice == "b":
        from common.langchain_clients import LangChainAzureAnthropicClient
        
        api_key = os.environ.get("AZURE_ANTHROPIC_API_KEY")
        endpoint = os.environ.get("AZURE_ANTHROPIC_ENDPOINT")
        deployment = os.environ.get("AZURE_ANTHROPIC_DEPLOYMENT_NAME", "claude-sonnet-4-5")
        
        # Prompt if not in env
        if not api_key or not endpoint:
            from getpass import getpass
            console.print("[bold]Azure Anthropic Configuration:[/bold]")
            endpoint = input("AZURE_ANTHROPIC_ENDPOINT (e.g., https://xxxx.services.ai.azure.com): ").strip() or endpoint
            api_key = getpass("AZURE_ANTHROPIC_API_KEY: ").strip() or api_key
        deployment = input(f"AZURE_ANTHROPIC_DEPLOYMENT_NAME [{deployment}]: ").strip() or deployment
        
        if not api_key or not endpoint:
            console.print("[red]Error: Azure Anthropic credentials required[/red]")
            sys.exit(1)
        
        os.environ["AZURE_ANTHROPIC_DEPLOYMENT_NAME"] = deployment
        
        client = LangChainAzureAnthropicClient()
        console.print(f"[green]Azure Anthropic: {deployment}[/green]\n")
        return client, deployment

    if choice == "g":
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            console.print("[red]Error: GROQ_API_KEY not set in environment[/red]")
            sys.exit(1)
        model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        client = OpenAIClient(
            base_url="https://api.groq.com/openai/v1",
            model=model,
            api_key=api_key,
            inject_no_think=False,
        )
        console.print(f"[green]Groq: {model}[/green]\n")
        return client, model

    if choice == "r":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            console.print("[red]Error: OPENROUTER_API_KEY not set in environment[/red]")
            sys.exit(1)
        console.print("[bold]Select OpenRouter model:[/bold]")
        console.print("  [1] google/gemini-2.0-flash-001        (free, 1M context)")
        console.print("  [2] meta-llama/llama-3.3-70b-instruct   (paid, but cheap)")
        console.print("  [3] openai/gpt-4o                       (paid, best format compliance)")
        console.print("  [4] mistralai/mistral-small-3.1-24b     (paid, but cheap)")
        console.print("  [5] deepseek/deepseek-chat              (paid, but cheap)")
        console.print("  [6] custom OpenRouter model slug")
        model_choice = input("Choice [1-6]: ").strip()
        model_map = {
            "1": "google/gemini-2.0-flash-001",
            "2": "meta-llama/llama-3.3-70b-instruct",
            "3": "openai/gpt-4o",
            "4": "mistralai/mistral-small-3.1-24b-instruct",
            "5": "deepseek/deepseek-chat",
        }
        model = model_map.get(model_choice)
        if not model:
            model = input("Enter model slug (e.g. google/gemini-2.0-flash-001): ").strip()
        client = OpenAIClient(
            base_url="https://openrouter.ai/api/v1",
            model=model,
            api_key=api_key,
            inject_no_think=False,
        )
        console.print(f"[green]OpenRouter: {model}[/green]\n")
        return client, model

    if choice == "h":
        api_key = os.environ.get("GITHUB_TOKEN")
        if not api_key:
            console.print("[red]Error: GITHUB_TOKEN not set in environment[/red]")
            sys.exit(1)
        model = os.environ.get("GH_MODEL", "gpt-4o-mini")
        client = OpenAIClient(
            base_url="https://models.inference.ai.azure.com",
            model=model,
            api_key=api_key,
            inject_no_think=False,
        )
        console.print(f"[green]GitHub Models: {model}[/green]\n")
        return client, model

    if choice == "c":
        account_id = os.environ.get("CF_ACCOUNT_ID")
        api_key = os.environ.get("CF_API_KEY")
        if not account_id or not api_key:
            console.print("[red]Error: CF_ACCOUNT_ID and CF_API_KEY must be set[/red]")
            sys.exit(1)
        model = os.environ.get("CF_MODEL", "@cf/meta/llama-3.2-3b-instruct")
        client = OpenAIClient(
            base_url=f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
            model=model,
            api_key=api_key,
            inject_no_think=False,
        )
        console.print(f"[green]Cloudflare AI: {model}[/green]\n")
        return client, model

    if choice == "m":
        from urllib.parse import urlparse
        base_url = input("Base URL (e.g. http://localhost:1234/v1): ").strip()
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https"):
            console.print("[red]Error: URL must start with http:// or https://[/red]")
            sys.exit(1)
        model = input("Model name: ").strip()
        if not model:
            console.print("[red]Error: Model name is required[/red]")
            sys.exit(1)
        from getpass import getpass
        api_key = getpass("API key (leave blank if none): ").strip()
        inject = input("Inject /no_think? (y/N): ").strip().lower() == "y"
        client = OpenAIClient(
            base_url=base_url,
            model=model,
            api_key=api_key or "not-needed",
            inject_no_think=inject,
        )
        console.print(f"[green]Custom: {model} at {base_url}[/green]\n")
        return client, model

    client = OpenAIClient()
    console.print(f"[green]Local LM Studio: qwen/qwen3-8b[/green]\n")
    return client, "qwen/qwen3-8b"


def get_token_budget(model_name: str, safety_margin: int = 4000) -> int:
    # Map Azure deployment names to their actual context limits
    azure_mappings = {
        "gpt-5.3-codex": 128000,
        "gpt-4": 8192,
        "gpt-4-turbo": 128000,
        "gpt-4-turbo-preview": 128000,
        "gpt-35-turbo": 4096,
        "claude-sonnet-4-5": 200000,
        "claude-opus": 200000,
        "claude-haiku": 100000,
        "Ministral-3B": 32768,
    }
    
    # Check exact match in MODEL_CONTEXT_LIMITS first
    if model_name in MODEL_CONTEXT_LIMITS:
        limit = MODEL_CONTEXT_LIMITS[model_name]
    # Check Azure mappings
    elif model_name in azure_mappings:
        limit = azure_mappings[model_name]
    # Check if model_name contains common patterns
    elif "gpt-4-turbo" in model_name.lower() or "gpt4-turbo" in model_name.lower():
        limit = 128000
    elif "gpt-4" in model_name.lower() or "gpt4" in model_name.lower():
        limit = 8192
    elif "gpt-35" in model_name.lower() or "gpt35" in model_name.lower():
        limit = 4096
    elif "claude-3-opus" in model_name.lower():
        limit = 200000
    elif "claude" in model_name.lower():
        limit = 200000  # Default for Claude models
    else:
        # Default fallback
        limit = 40960
    
    # Use 10% safety margin for large models, 20% for smaller ones
    safety = int(limit * 0.1) if limit > 50000 else int(limit * 0.2)
    return max(limit - safety, 20000)  # Ensure minimum budget


def estimate_tokens_per_model(model_name: str, token_count: int) -> float:
    limits = {
        "llama-3.3-70b-versatile": 0.59,
    }
    return limits.get(model_name, 0)
