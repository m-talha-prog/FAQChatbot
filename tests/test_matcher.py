"""Tests for the FAQ matcher.

Run from the project root with:
    python -m pytest
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make `import matcher` work no matter where pytest is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from matcher import FALLBACK_MESSAGE, MAX_RELATED, FAQMatcher  # noqa: E402


@pytest.fixture(scope="module")
def matcher() -> FAQMatcher:
    """Build the matcher once for all tests (fitting TF-IDF is not free)."""
    return FAQMatcher()


# (what the user types, the FAQ question we expect to be matched, why it's a good test)
CASES = [
    ("How long does shipping take?", "How long does shipping take?", "exact question"),
    ("how many days until my delivery arrives", "How long does shipping take?", "rephrased wording"),
    ("how do i reset my pasword", "I forgot my password. How do I reset it?", "typo: pasword"),
    ("do you accept paypall", "What payment methods do you accept?", "typo: paypall"),
    ("how do i track my parcel", "How can I track my order?", "synonym: parcel"),
]


@pytest.mark.parametrize("query, expected, _why", CASES)
def test_matches_expected_faq(matcher: FAQMatcher, query: str, expected: str, _why: str) -> None:
    """Exact, rephrased and typo'd questions should reach the right FAQ confidently."""
    response = matcher.ask(query)
    assert response.confident, f"score {response.match.score:.2f} is below the threshold"
    assert response.match.question == expected


def test_out_of_scope_question_gets_fallback(matcher: FAQMatcher) -> None:
    """A question unrelated to the store must not get a (wrong) FAQ answer."""
    response = matcher.ask("what is the meaning of life")
    assert not response.confident
    assert response.answer == FALLBACK_MESSAGE


def test_related_questions_are_valid(matcher: FAQMatcher) -> None:
    """Related suggestions: at most MAX_RELATED, no duplicates, never the answer itself."""
    response = matcher.ask("how long does shipping take")
    questions = [m.question for m in response.related]

    assert 0 < len(questions) <= MAX_RELATED
    assert len(set(questions)) == len(questions)
    assert response.match.question not in questions