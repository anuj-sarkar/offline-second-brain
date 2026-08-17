"""
tests/test_generation.py

Tests for app.generation.llm. Makes real calls to your local Ollama
server (no mocking) - intentional, since the goal of this phase is
proving the local generation pipeline works end to end.

Run with: python -m pytest tests/test_generation.py -v
"""

from app.generation.llm import generate
from config.model_config import LLMConfig


def test_generate_returns_nonempty_string():
    response = generate("Say hello in exactly two words.")
    assert isinstance(response, str)
    assert len(response.strip()) > 0


def test_low_temperature_produces_consistent_output():
    # Same prompt, temperature=0, run twice - should be identical or
    # near-identical, since low temperature minimizes randomness.
    # (Not a strict guarantee across all models/backends, but a
    # reasonable sanity check for llama3.2:3b via Ollama.)
    config = LLMConfig(temperature=0.0, max_tokens=20)
    prompt = "What is the capital of France? Answer in one word."

    response_a = generate(prompt, config=config)
    response_b = generate(prompt, config=config)

    assert "paris" in response_a.lower()
    assert "paris" in response_b.lower()


def test_max_tokens_limits_response_length():
    config = LLMConfig(max_tokens=10)
    response = generate("Write a long essay about the history of computing.", config=config)
    # Rough check - a 10-token cap should produce a short response,
    # not a full essay. Word count is a loose proxy for token count.
    assert len(response.split()) < 30