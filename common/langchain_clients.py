"""
LangChain-based LLM clients for Azure and other providers.
"""

from dataclasses import dataclass, field
import os
import time


def _ensure_content_string(content) -> str:
    """Ensure content is a string, handling list/dict/other types."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Join list items (e.g., content blocks)
        # Each item might be a dict with 'text' key (Anthropic format)
        parts = []
        for item in content:
            if isinstance(item, dict) and 'text' in item:
                parts.append(item['text'])
            elif isinstance(item, str):
                parts.append(item)
            else:
                # Try to get 'text' attribute or str() fallback
                if hasattr(item, 'text'):
                    parts.append(item.text)
                else:
                    parts.append(str(item))
        return "".join(parts)
    if isinstance(content, dict):
        # If it's a dict with 'text' key, use that
        return content.get("text", str(content))
    # Try to extract text attribute for objects with .text property
    if hasattr(content, 'text'):
        return content.text
    return str(content)


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


class LangChainAzureOpenAIClient:
    """LangChain-based Azure OpenAI client."""
    
    def __init__(self):
        self.model = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.3-codex")
        self.base_url = os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        self.llm = None
        self._initialize()
    
    def _initialize(self):
        try:
            from langchain_openai import ChatOpenAI
            
            self.llm = ChatOpenAI(
                model=self.model,
                base_url=self.base_url,
                api_key=self.api_key,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Azure OpenAI client: {e}")
    
    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a chat message."""
        if not self.llm:
            return LLMResponse(
                content="",
                usage=TokenUsage(),
                error="Client not initialized"
            )
        
        start = time.monotonic()
        try:
            response = await self.llm.ainvoke(
                messages,
                config={"temperature": temperature, "max_tokens": max_tokens}
            )
            elapsed = (time.monotonic() - start) * 1000
            
            # Extract token usage from response
            prompt_tokens = 0
            completion_tokens = 0
            reasoning_tokens = 0
            total_tokens = 0
            
            # LangChain responses may have usage_metadata attribute
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                prompt_tokens = usage.get('input_tokens', 0)
                completion_tokens = usage.get('output_tokens', 0)
                reasoning_tokens = usage.get('reasoning_tokens', 0)
                total_tokens = usage.get('total_tokens', 0) or (prompt_tokens + completion_tokens)
            
            # Fallback: check response attributes
            if total_tokens == 0 and hasattr(response, 'response_metadata'):
                metadata = response.response_metadata
                if isinstance(metadata, dict) and 'usage' in metadata:
                    usage = metadata['usage']
                    prompt_tokens = usage.get('prompt_tokens', prompt_tokens)
                    completion_tokens = usage.get('completion_tokens', completion_tokens)
                    reasoning_tokens = usage.get('reasoning_tokens', reasoning_tokens)
                    total_tokens = usage.get('total_tokens', 0) or (prompt_tokens + completion_tokens)
            
            return LLMResponse(
                content=_ensure_content_string(response.content) if hasattr(response, 'content') else str(response),
                usage=TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    reasoning_tokens=reasoning_tokens,
                    total_tokens=total_tokens,
                ),
                latency_ms=elapsed,
                raw=response.response_metadata if hasattr(response, 'response_metadata') else {},
            )
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            return LLMResponse(
                content="",
                usage=TokenUsage(),
                latency_ms=elapsed,
                error=f"{type(e).__name__}: {str(e)[:200]}"
            )


class LangChainAzureAnthropicClient:
    """LangChain-based Azure Anthropic client."""
    
    def __init__(self):
        self.model = os.environ.get("AZURE_ANTHROPIC_DEPLOYMENT_NAME", "claude-sonnet-4-5")
        self.api_url = os.environ.get("AZURE_ANTHROPIC_ENDPOINT")
        self.api_key = os.environ.get("AZURE_ANTHROPIC_API_KEY")
        self.llm = None
        self._initialize()
    
    def _initialize(self):
        try:
            from langchain_anthropic import ChatAnthropic
            
            self.llm = ChatAnthropic(
                model=self.model,
                anthropic_api_url=self.api_url,
                anthropic_api_key=self.api_key,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Azure Anthropic client: {e}")
    
    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a chat message."""
        if not self.llm:
            return LLMResponse(
                content="",
                usage=TokenUsage(),
                error="Client not initialized"
            )
        
        start = time.monotonic()
        try:
            response = await self.llm.ainvoke(
                messages,
                config={"temperature": temperature, "max_tokens": max_tokens}
            )
            elapsed = (time.monotonic() - start) * 1000
            
            # Extract token usage from response
            prompt_tokens = 0
            completion_tokens = 0
            reasoning_tokens = 0
            total_tokens = 0
            
            # LangChain responses may have usage_metadata attribute
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                prompt_tokens = usage.get('input_tokens', 0)
                completion_tokens = usage.get('output_tokens', 0)
                reasoning_tokens = usage.get('reasoning_tokens', 0)
                total_tokens = usage.get('total_tokens', 0) or (prompt_tokens + completion_tokens)
            
            # Fallback: check response attributes
            if total_tokens == 0 and hasattr(response, 'response_metadata'):
                metadata = response.response_metadata
                if isinstance(metadata, dict) and 'usage' in metadata:
                    usage = metadata['usage']
                    prompt_tokens = usage.get('prompt_tokens', prompt_tokens)
                    completion_tokens = usage.get('completion_tokens', completion_tokens)
                    reasoning_tokens = usage.get('reasoning_tokens', reasoning_tokens)
                    total_tokens = usage.get('total_tokens', 0) or (prompt_tokens + completion_tokens)
            
            return LLMResponse(
                content=_ensure_content_string(response.content) if hasattr(response, 'content') else str(response),
                usage=TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    reasoning_tokens=reasoning_tokens,
                    total_tokens=total_tokens,
                ),
                latency_ms=elapsed,
                raw=response.response_metadata if hasattr(response, 'response_metadata') else {},
            )
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            return LLMResponse(
                content="",
                usage=TokenUsage(),
                latency_ms=elapsed,
                error=f"{type(e).__name__}: {str(e)[:200]}"
            )


def get_azure_openai_client() -> LangChainAzureOpenAIClient:
    """Create a fresh Azure OpenAI LangChain client."""
    return LangChainAzureOpenAIClient()


def get_azure_anthropic_client() -> LangChainAzureAnthropicClient:
    """Create a fresh Azure Anthropic LangChain client."""
    return LangChainAzureAnthropicClient()
