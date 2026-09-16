"""
app/generation/llm.py

Local LLM generation interface via LangChain's ChatOllama.
Supports both synchronous single-response generation and streaming token generation.
"""

from collections.abc import Generator
import logging

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from config.model_config import DEFAULT_LLM_CONFIG, LLMConfig

logger = logging.getLogger(__name__)

# Cache ChatOllama instances keyed by (model_name, temperature, top_p, max_tokens)
# so we don't reconstruct on every call.
_llm_cache: dict[tuple, ChatOllama] = {}


def _get_llm(config: LLMConfig) -> ChatOllama:
    key = (config.model_name, config.temperature, config.top_p, config.max_tokens)
    if key not in _llm_cache:
        _llm_cache[key] = ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            top_p=config.top_p,
            num_predict=config.max_tokens,
        )
        logger.info(f"Initialized ChatOllama: model={config.model_name}, temp={config.temperature}")
    return _llm_cache[key]


def generate(prompt: str, config: LLMConfig = DEFAULT_LLM_CONFIG) -> str:
    """Send prompt to local Ollama model and return complete text response."""
    logger.info(f"Generating with model={config.model_name}, temperature={config.temperature}")
    llm = _get_llm(config)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def generate_stream(
    prompt: str,
    config: LLMConfig = DEFAULT_LLM_CONFIG,
) -> Generator[str, None, None]:
    """
    Stream generated tokens from local Ollama model.
    Yields string chunks suitable for st.write_stream or terminal streaming.
    """
    logger.info(f"Streaming with model={config.model_name}, temperature={config.temperature}")
    llm = _get_llm(config)
    for chunk in llm.stream([HumanMessage(content=prompt)]):
        yield chunk.content