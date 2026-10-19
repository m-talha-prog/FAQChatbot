"""FAQ matching using TF-IDF vectors and cosine similarity.

How it works, in short:
    1. Each FAQ question is converted into a vector of numbers (TF-IDF).
    2. The user's question is converted into a vector the same way.
    3. We measure the angle between the user's vector and every FAQ vector
       (cosine similarity) and pick the closest one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import preprocess

DEFAULT_FAQ_PATH = Path(__file__).parent / "faqs.json"

# Below this cosine similarity we do NOT trust the match. 0.25 is a starting
# point found by trying real questions: tune it up if the bot answers wrongly,
# down if it refuses questions it should understand.
CONFIDENCE_THRESHOLD = 0.25
MAX_RELATED = 2  # how many "related questions" to suggest
MIN_RELATED_SCORE = 0.05  # ignore suggestions that share almost nothing with the query
FALLBACK_MESSAGE = "I'm not sure I understood that. Could you try rephrasing your question?"


@dataclass(frozen=True)
class Match:
    """One FAQ entry together with how similar it is to the user's question."""

    index: int
    question: str
    answer: str
    category: str
    score: float  # cosine similarity, from 0.0 (nothing in common) to 1.0 (identical)


@dataclass(frozen=True)
class Response:
    """What the chatbot says back to the user."""

    answer: str  # the FAQ answer, or the polite fallback message
    confident: bool  # True if the best score reached the threshold
    match: Match  # best candidate (kept even when not confident, for debugging/UI)
    related: list[Match]  # up to MAX_RELATED suggested questions


class FAQMatcher:
    """Finds the FAQ entry that best matches a free-text user question."""

    def __init__(
        self,
        faq_path: str | Path = DEFAULT_FAQ_PATH,
        threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        """Load the FAQs and fit the TF-IDF vectorizer on their questions.

        Args:
            faq_path: Path to a JSON file containing a list of FAQ objects with
                "question", "answer" and "category" keys (and optional "keywords").
            threshold: Minimum cosine similarity needed to trust a match.
        """
        self.threshold = threshold
        with open(faq_path, encoding="utf-8") as f:
            self.faqs: list[dict] = json.load(f)

        # TF-IDF = Term Frequency x Inverse Document Frequency.
        #   - Term Frequency: a word that appears often in a question matters more.
        #   - Inverse Document Frequency: a word that appears in MANY questions
        #     (e.g. "order") is less useful for telling them apart, so it gets a
        #     lower weight, while a rare word like "refund" gets a higher one.
        # We pass our own `preprocess` function as the analyzer so the vectorizer
        # uses our NLTK pipeline instead of its built-in tokenizer.
        self.vectorizer = TfidfVectorizer(analyzer=preprocess)

        # We vectorize the question PLUS its optional keywords. Keywords hold
        # words users might type that the question itself doesn't contain
        # (e.g. "paypal", "parcel"), which helps with vocabulary mismatch.
        # The question is repeated twice so its words weigh more than the
        # keywords; otherwise a long keyword list "dilutes" the question.
        # fit_transform learns the vocabulary + IDF weights from these texts
        # and returns a matrix: one row per FAQ, one column per vocabulary word.
        documents = [
            f"{f['question']} {f['question']} {f.get('keywords', '')}" for f in self.faqs
        ]
        self.faq_vectors = self.vectorizer.fit_transform(documents)

    def _scores(self, query: str) -> np.ndarray:
        """Return the cosine similarity between `query` and every FAQ question."""
        # transform (not fit_transform!): reuse the vocabulary learned from the
        # FAQs. Words the vectorizer has never seen are simply ignored.
        query_vector = self.vectorizer.transform([query])

        # Cosine similarity = (A . B) / (|A| x |B|).
        # It compares the *direction* of two vectors, not their length, so a
        # short question and a long one can still match if they use the same
        # important words. If the query has no known words, every score is 0.
        return cosine_similarity(query_vector, self.faq_vectors).ravel()

    def _to_match(self, index: int, score: float) -> Match:
        """Build a Match object for the FAQ at `index`."""
        faq = self.faqs[index]
        return Match(
            index=index,
            question=faq["question"],
            answer=faq["answer"],
            category=faq.get("category", ""),
            score=float(score),
        )

    def best_match(self, query: str) -> Match:
        """Return the single most similar FAQ entry (no threshold applied yet)."""
        scores = self._scores(query)
        best = int(np.argmax(scores))
        return self._to_match(best, scores[best])

    def top_matches(self, query: str, k: int = 3) -> list[Match]:
        """Return the `k` most similar FAQ entries, best first."""
        scores = self._scores(query)
        # argsort sorts ascending, so we negate the scores to get best first.
        order = np.argsort(-scores, kind="stable")[:k]
        return [self._to_match(int(i), scores[i]) for i in order]

    def ask(self, query: str) -> Response:
        """Answer a user question, or fall back politely if we are unsure.

        The confidence threshold is the safety net: a wrong answer is worse
        than "I'm not sure", so low-scoring matches are not shown as answers.
        """
        ranked = self.top_matches(query, k=1 + MAX_RELATED)
        best, others = ranked[0], ranked[1:]

        if best.score >= self.threshold:
            related = [m for m in others if m.score >= MIN_RELATED_SCORE]
            return Response(best.answer, True, best, related)

        # Not confident: don't answer, but still offer the closest questions.
        suggestions = [m for m in ranked if m.score >= MIN_RELATED_SCORE][:MAX_RELATED]
        return Response(FALLBACK_MESSAGE, False, best, suggestions)