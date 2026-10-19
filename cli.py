"""Command-line interface for quickly testing the FAQ matcher.

Usage:
    python cli.py
Type a question and press Enter. Type 'quit' or 'exit' (or press Ctrl+C) to leave.
"""

from __future__ import annotations

from matcher import MIN_RELATED_SCORE, FAQMatcher

EXIT_WORDS = {"quit", "exit", "q"}


def main() -> None:
    """Run the interactive chat loop."""
    matcher = FAQMatcher()
    print(f"FAQ Chatbot (type 'quit' to exit) | confidence threshold: {matcher.threshold}\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in EXIT_WORDS:
            print("Bye!")
            break

        response = matcher.ask(query)
        print(f"\nBot: {response.answer}")

        score = response.match.score
        if response.confident:
            print(f"     [matched: \"{response.match.question}\" | score: {score:.2f}]")
        elif score >= MIN_RELATED_SCORE:
            print(f"     [closest (below threshold): \"{response.match.question}\" | score: {score:.2f}]")
        else:
            print(f"     [no similar FAQ found | score: {score:.2f}]")

        if response.related:
            heading = "You might also ask" if response.confident else "Did you mean"
            print(f"     {heading}:")
            for related in response.related:
                print(f"       - {related.question} ({related.score:.2f})")
        print()


if __name__ == "__main__":
    main()