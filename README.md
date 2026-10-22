# FAQChatbot

A small FAQ chatbot for an e-commerce store, built with classic NLP: **NLTK** for text cleaning, **scikit-learn** (TF-IDF + cosine similarity) for matching, and **Streamlit** for the chat UI. It answers questions about shipping, returns, payments, order tracking and account issues, and says "I'm not sure" instead of guessing when it doesn't understand.

No API keys, no GPU, no external services. Everything runs locally.

## Features

- Matches a free-text question to the closest of 29 FAQs
- Handles rephrased wording and small typos
- **Confidence threshold:** low-scoring matches get a polite fallback, not a wrong answer
- Suggests up to 2 related questions
- Streamlit chat UI with a sidebar showing the matched question and similarity score
- CLI mode for quick testing
- Unit tests with pytest

## Project structure

```
FAQChatbot/
├── faqs.json            # The knowledge base: questions, answers, categories, keywords
├── preprocess.py        # NLTK text cleaning pipeline
├── matcher.py           # TF-IDF vectorizer + cosine similarity + threshold logic
├── app.py               # Streamlit chat UI
├── cli.py               # Command-line chat for quick testing
├── tests/
│   └── test_matcher.py  # pytest tests (exact, rephrased, typos, fallback, related)
├── requirements.txt
└── README.md
```

## Setup

You need Python 3.9 or newer.

```bash
# 1. Clone the repo and enter it
git clone https://github.com/<m-talha-prog>/FAQChatbot.git
cd FAQChatbot

# 2. (Recommended) create a virtual environment
python -m venv .venv
# Windows:       .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate

# 3. Install dependencies
python -m pip install -r requirements.txt
```

NLTK downloads the small language data it needs (tokenizer, stopwords, WordNet) automatically the first time you run the project, so you need an internet connection for that first run.

## Usage

```bash
# Chat in the terminal
python cli.py

# Chat in the browser (opens at http://localhost:8501)
python -m streamlit run app.py

# Run the tests
python -m pytest
```

Using `python -m streamlit` instead of plain `streamlit` avoids a common Windows problem where the `streamlit` command isn't on your PATH.

Try questions like:

| You type | What happens |
|---|---|
| `how long does shipping take` | Matches the shipping-time FAQ |
| `how do i reset my pasword` | Typo, but still finds the password FAQ |
| `how do i track my parcel` | "parcel" is a keyword for the tracking FAQ |
| `what is the meaning of life` | Below the threshold, so the bot asks you to rephrase |

## How the matching works (in plain language)

Think of the chatbot as a very fast librarian with an index card for every FAQ. When you ask something, it finds the card whose wording overlaps most with yours. There are four steps.

### 1. Clean the text (`preprocess.py`)

Computers see `"Returns?"` and `"return"` as different strings, so we first make them look alike:

1. **Lowercase:** `Returns` becomes `returns`
2. **Remove punctuation:** `refund?` becomes `refund`
3. **Tokenize:** split the sentence into individual words
4. **Remove stopwords:** drop filler words like "the", "is", "my", "do". They appear in almost every sentence, so they say nothing about *what* is being asked. (We keep "no" and "not" because they flip meaning.)
5. **Lemmatize:** reduce each word to its dictionary form, so `returned`, `returning` and `returns` all become `return`

Example: `"How long does shipping take?"` becomes `['long', 'ship', 'take']`.

### 2. Turn text into numbers (TF-IDF)

Algorithms can't compare sentences directly, so each FAQ becomes a list of numbers (a *vector*), with one number per word in our vocabulary. The number is the word's **TF-IDF** weight:

- **TF (term frequency):** a word used more in a sentence matters more to it.
- **IDF (inverse document frequency):** a word that appears in *many* FAQs (like "order") is a weak clue, while a rare word (like "refund" or "paypal") is a strong clue, so it gets a higher weight.

Analogy: if you're guessing which book a sentence came from, the word "the" tells you nothing, but "Hogwarts" tells you a lot. IDF is how the math knows that.

Each FAQ is vectorized as its question (counted twice, so the question's own words matter most) plus its optional `keywords` (extra words people might type, like "parcel" or "paypal").

### 3. Compare with cosine similarity

Your question is turned into a vector the same way. Then we measure how closely it points in the same direction as each FAQ vector:

```
cosine similarity = (A · B) / (|A| × |B|)
```

Picture two arrows drawn from the same point. If they point the same way, the score is close to 1.0. If they point in unrelated directions, it's 0. Because it compares *direction* and not *length*, a short question and a long one can still match if they use the same important words.

### 4. Decide whether to answer

- If the best score is **at least 0.25** (`CONFIDENCE_THRESHOLD` in `matcher.py`), the bot shows that FAQ's answer, plus up to 2 related questions (the next-best matches).
- If it's **below 0.25**, the bot says it isn't sure and shows its closest guesses as "Did you mean...". A wrong answer is worse than admitting uncertainty.

### Tuning the threshold

0.25 is a starting point found by trying realistic questions. If the bot gives wrong answers, raise it. If it refuses questions it should understand, lower it. Use the sidebar in the Streamlit app to see real scores while you experiment.

## Adding or editing FAQs

Edit `faqs.json`. Each entry looks like this:

```json
{
  "id": 30,
  "category": "payments",
  "question": "Do you offer installment plans?",
  "answer": "Yes, orders over $100 can be split into 3 monthly payments at checkout.",
  "keywords": "installment, pay later, split payment, monthly"
}
```

Keep keywords short (about 3-8 terms) with words and phrasings users might actually type. A very long keyword list dilutes the weight of each word and can lower scores.

## Limitations

This is a **bag-of-words** system: it counts shared words and cannot understand meaning.

- **Synonyms must be listed.** It doesn't know "parcel" means "package" unless a keyword says so.
- **Word order and context are ignored.** "Do you take PayPal?" can wrongly match "How long does shipping take?" because both contain "take". "Do you accept PayPal?" works.
- **Typos only help partially.** A typo is simply an unknown word and is ignored. It works if the other words still match (`"refnd policy"` works because "policy" survives), but a question that is only typos scores 0.
- **Very short questions are ambiguous.** "Where is my package?" could mean tracking or a missing parcel.
- **One question, one answer.** It can't handle two questions in one message or follow-ups like "and what about international?"
- **Fixed knowledge base.** It only knows what is in `faqs.json`, and the threshold needs manual tuning.
- **English only.**
- **Small test set.** The tests cover representative cases, not every possible phrasing.

## Upgrading to embeddings

Embeddings fix the biggest limitation above: they capture **meaning**, not just shared words. A sentence-embedding model (such as `all-MiniLM-L6-v2` from the `sentence-transformers` library) turns each sentence into a vector where similar meanings sit close together. "Do you take PayPal?" and "Is PayPal accepted?" end up near each other even though they share almost no words, and "parcel" lands near "package" without a keyword list.

The change is small, because the rest of the pipeline (cosine similarity, threshold, related questions) stays the same:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Replace the TF-IDF step in FAQMatcher.__init__ ...
faq_vectors = model.encode(questions, normalize_embeddings=True)

# ... and the query step in FAQMatcher._scores
query_vector = model.encode([query], normalize_embeddings=True)
scores = (faq_vectors @ query_vector.T).ravel()  # cosine similarity
```

Things to know before switching:

- You no longer need the NLTK preprocessing; embedding models work best on the raw sentence.
- Similarity scores sit on a different scale, so **re-tune the threshold** (often somewhere around 0.4-0.6).
- It adds a larger dependency (PyTorch) and a model download of roughly 90 MB.
- Matching is a bit slower, though still fast for a few hundred FAQs.

A good next step is to keep both versions and compare them on the same test questions.

## Tech stack

Python · NLTK · scikit-learn · Streamlit · pytest