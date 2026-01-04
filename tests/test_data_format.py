import importlib.util
from pathlib import Path

from scripts.prepare_dataset import format_sft_prompt


def test_format_sft_prompt():
    human = "  Hello there  "
    assistant = "I am fine. "
    out = format_sft_prompt(human, assistant)
    assert out == "Human: Hello there\nAssistant: I am fine."