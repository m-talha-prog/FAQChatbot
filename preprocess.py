"""Text preprocessing for the FAQ chatbot.

Pipeline (in order):
    lowercase -> remove punctuation -> tokenize -> remove stopwords -> lemmatize

Why preprocess at all? A computer sees "Returns?" and "return" as different
strings. Preprocessing squashes surface differences (case, punctuation,
word endings) so that texts with the same *meaning* end up with the same
*tokens*, which makes matching far more reliable.
"""

from __future__ import annotations

import string
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# NLTK ships its language data separately from the code. Maps the package
# name to the path NLTK uses to check whether it is already installed.
_NLTK_RESOURCES: dict[str, str] = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",  # needed by newer NLTK versions
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
}

# Negations carry meaning ("can't log in" vs "can log in"), so we keep them
# even though NLTK's default stopword list includes them.
_KEEP_WORDS: set[str] = {"no", "not", "nor"}

# str.translate table that deletes every ASCII punctuation character.
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def ensure_nltk_data() -> None:
    """Download any missing NLTK data (only runs the download once)."""
    for package, path in _NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


ensure_nltk_data()


@lru_cache(maxsize=1)
def _stop_words() -> frozenset[str]:
    """Return the English stopword set (cached so we build it only once)."""
    return frozenset(set(stopwords.words("english")) - _KEEP_WORDS)


@lru_cache(maxsize=1)
def _lemmatizer() -> WordNetLemmatizer:
    """Return a shared WordNet lemmatizer instance."""
    return WordNetLemmatizer()


def preprocess(text: str) -> list[str]:
    """Turn raw text into a list of clean, normalized tokens.

    Example:
        >>> preprocess("How long does shipping take?")
        ['long', 'ship', 'take']

    Args:
        text: Any raw string, such as a user question.

    Returns:
        A list of lowercase lemma tokens with stopwords removed.
    """
    # 1. Lowercase: "Returns" and "returns" should count as the same word.
    text = text.lower()

    # 2. Remove punctuation: "refund?" and "refund" should match.
    text = text.translate(_PUNCT_TABLE)

    # 3. Tokenize: split the string into individual words.
    tokens = word_tokenize(text)

    # 4. Remove stopwords: very common words ("the", "is", "my", "do") appear
    #    in almost every sentence, so they add noise rather than meaning.
    stop_words = _stop_words()
    tokens = [t for t in tokens if t not in stop_words]

    # 5. Lemmatize: reduce words to their dictionary base form so that
    #    "returns", "returned", "returning" all become "return".
    #    WordNet needs to know the part of speech, and we don't run a POS
    #    tagger, so we try verb first, then noun. This handles most
    #    FAQ-style words ("shipping" -> "ship", "cards" -> "card").
    lemmatizer = _lemmatizer()
    return [lemmatizer.lemmatize(lemmatizer.lemmatize(t, pos="v"), pos="n") for t in tokens]


def preprocess_to_string(text: str) -> str:
    """Same as :func:`preprocess` but joins the tokens into one string."""
    return " ".join(preprocess(text))


if __name__ == "__main__":
    # Quick demo: python preprocess.py
    for sample in ["How long does shipping take?", "Can't log in!! My account is locked."]:
        print(f"{sample!r} -> {preprocess(sample)}")