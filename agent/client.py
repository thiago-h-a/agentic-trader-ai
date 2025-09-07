import os
from typing import Optional, Literal, Any


def _env_truthy(v: Optional[str]) -> bool:
    """
    Accept a non-empty, non-placeholder string as a "present" env var.
    """
    return bool(
        v
        and v.strip()
        and v.strip().lower()
        not in {"", "none", "null", "your_azure_api_key_here", "your_google_api_key_here"}
    )


class _DummyResp:
    def __init__(self, content: str) -> None:
        self.content = content


class _DummyLLM:
    """
    Minimal stand-in client when no provider SDK/API key is available.
    It exposes .invoke(prompt) and returns an object with a .content string,
    mirroring the shape of real clients used here.
    """

    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def invoke(self, prompt: str) -> _DummyResp:
        # Conservative HOLD response as a safe default
        return _DummyResp(
            '{"action":"HOLD","confidence":0.5,"reasoning":"fallback","risk_level":"medium","expected_outcome":"no_trade"}'
        )


def get_llm_client(
    provider: Optional[Literal["azure", "gemini"]] = None,
    temperature: float = 0.1,
    max_tokens: int = 1000,
    **kwargs: Any,
):
    """
    Return an LLM client exposing `.invoke(prompt) -> ResponseLike(content=str)`.
    Preference order:
      1) Azure OpenAI if its key/envs are present
      2) Google Gemini if its key is present
      3) Fallback dummy client
    """
    # Auto-detect provider if not set
    if provider is None:
        azure_ok = _env_truthy(os.getenv("AZURE_OPENAI_API_KEY"))
        gemini_ok = _env_truthy(os.getenv("GOOGLE_API_KEY"))
        if azure_ok:
            provider = "azure"
        elif gemini_ok:
            provider = "gemini"
        else:
            provider = "azure"  # default (will fall back to _DummyLLM on import failure)

    if provider == "azure":
        try:
            from langchain_openai import AzureChatOpenAI

            return AzureChatOpenAI(
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        except Exception:
            return _DummyLLM()
    else:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                temperature=temperature,
                max_output_tokens=max_tokens,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                **kwargs,
            )
        except Exception:
            return _DummyLLM()
