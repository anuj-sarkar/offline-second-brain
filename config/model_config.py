"""
config/model_config.py

Centralized model configuration - kept separate from app logic so
you can change models/parameters in exactly one place, without
hunting through generation code. This also makes future experiments
(Phase 11: comparing different LLMs) straightforward - swap values
here rather than editing function calls scattered across the codebase.
"""

from dataclasses import dataclass


@dataclass
class LLMConfig:
    model_name: str = "llama3.2:3b"
    temperature: float = 0.1       # low = consistent, factual, less "creative" -
                                     # what we want for RAG, which should faithfully
                                     # reflect retrieved context, not improvise
    top_p: float = 0.9              # nucleus sampling - restricts token choices to
                                     # the smallest set whose cumulative probability
                                     # exceeds top_p; works alongside temperature
    max_tokens: int = 512           # cap on response length, prevents runaway
                                     # generation and keeps answers focused


@dataclass
class EmbeddingConfig:
    model_name: str = "nomic-embed-text"


DEFAULT_LLM_CONFIG = LLMConfig()
DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig()