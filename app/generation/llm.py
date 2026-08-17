"""
app/generation/llm.py

Phase 6 deliverable: a minimal, isolated interface to the local LLM
via Ollama. Deliberately has NO awareness of retrieval, chunks, or
RAG - this module's only job is "accept a prompt, return a response."
Combining this with retrieval is Phase 7's job, not this module's.

Testing generation in isolation like this means that if the full RAG
pipeline misbehaves later, you already know generation-by-itself
works correctly - narrowing down where a bug lives.
"""

import logging

import ollama

from config.model_config import LLMConfig, DEFAULT_LLM_CONFIG

logger = logging.getLogger(__name__)


def generate(prompt: str, config: LLMConfig = DEFAULT_LLM_CONFIG) -> str:
    """
    Send a prompt to the local Ollama model and return its response.

    Input:  a prompt string, an optional LLMConfig (uses sane defaults
            if not provided)
    Output: the model's generated text response

    This is a single blocking call - Ollama loads the model (if not
    already warm in memory) and streams tokens internally, but the
    Python client here waits for the full response before returning.
    Streaming token-by-token to the user is a nice later improvement,
    not needed for this phase.
    """
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    test_prompts = [
        "What is 2 + 2? Answer in one short sentence.",
        "Explain what a genetic algorithm is, in two sentences.",
    ]

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        response = generate(prompt)
        print(f"Response: {response}")
        print("-" * 70)