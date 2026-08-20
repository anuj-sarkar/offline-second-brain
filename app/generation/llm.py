"""
app/generation/llm.py

Local LLM generation interface via Ollama.
Supports both synchronous single-response generation and streaming token generation.
"""

from collections.abc import Generator
import logging

import ollama

from config.model_config import DEFAULT_LLM_CONFIG, LLMConfig

logger = logging.getLogger(__name__)


def generate(prompt: str, config: LLMConfig = DEFAULT_LLM_CONFIG) -> str:
    """Send prompt to local Ollama model and return complete text response."""
    logger.info(f"Generating with model={config.model_name}, temperature={config.temperature}")

    response = ollama.chat(
        model=config.model_name,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": config.temperature,
            "top_p": config.top_p,
            "num_predict": config.max_tokens,
        },
    )
    return response["message"]["content"]


def generate_stream(
    prompt: str,
    config: LLMConfig = DEFAULT_LLM_CONFIG,
) -> Generator[str, None, None]:
    """
    Stream generated tokens from local Ollama model.
    Yields string chunks suitable for st.write_stream or terminal streaming.
    """
    logger.info(f"Streaming with model={config.model_name}, temperature={config.temperature}")

    stream = ollama.chat(
        model=config.model_name,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": config.temperature,
            "top_p": config.top_p,
            "num_predict": config.max_tokens,
        },
        stream=True,
    )

    for chunk in stream:
        yield chunk["message"]["content"]