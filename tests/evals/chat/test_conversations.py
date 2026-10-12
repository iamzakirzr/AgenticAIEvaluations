"""Multi-turn chat: memory, role adherence, completeness, per-turn relevancy.

The golden supplies the user's turns; the bot under test produces the
assistant turns live, so the conversation scored is the one it really had.
"""

import pytest

from evalkit.builders import run_conversation
from evalkit.config import get_settings
from evalkit.datasets import case_id, load_conversations
from evalkit.metrics import suites
from evalkit.runner import run_metrics

pytestmark = pytest.mark.llm

GOLDENS = load_conversations(get_settings().goldens_dir / "conversations.json")


@pytest.mark.parametrize("golden", GOLDENS, ids=case_id)
def test_conversation(golden, chat_app, judge, thresholds, record):
    test_case = run_conversation(chat_app, golden)
    record(run_metrics("chat", case_id(golden), test_case, suites.chat_judged(judge, thresholds)))
