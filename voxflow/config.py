import os
from dataclasses import dataclass, field


def _env(name, default, cast=str):
    return cast(os.getenv(name, default))


@dataclass
class Settings:
    vllm_url: str = field(default_factory=lambda: _env("VLLM_URL", "http://localhost:8001/v1"))
    llm_model: str = field(default_factory=lambda: _env("VLLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", "EMPTY"))
    whisper_device: str = field(default_factory=lambda: _env("WHISPER_DEVICE", "cpu"))  # cpu | cuda
    tts_voice: str = field(default_factory=lambda: _env("TTS_VOICE", "en-US-AriaNeural"))
    whisper_model: str = field(default_factory=lambda: _env("WHISPER_MODEL", "base.en"))
    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", ""))
    database_url: str = field(default_factory=lambda: _env("DATABASE_URL", ""))
    # resilience
    llm_first_token_timeout: float = field(default_factory=lambda: _env("LLM_TTFT_TIMEOUT", "3.0", float))
    llm_retries: int = field(default_factory=lambda: _env("LLM_RETRIES", "2", int))
    tts_timeout: float = field(default_factory=lambda: _env("TTS_TIMEOUT", "4.0", float))
    plan_timeout: float = field(default_factory=lambda: _env("PLAN_TIMEOUT", "1.0", float))
