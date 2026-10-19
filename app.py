"""Streamlit chat UI for the FAQ chatbot.

Run with:
    python -m streamlit run app.py

The sidebar shows which FAQ was matched and its similarity score, so the
matching is visible while you chat (handy for demos).
"""

from __future__ import annotations

import streamlit as st

from matcher import MIN_RELATED_SCORE, FAQMatcher, Response

GREETING = "Hi! Ask me about shipping, returns, payments, order tracking, or your account."


@st.cache_resource
def load_matcher() -> FAQMatcher:
    """Build the matcher once and reuse it across reruns.

    Streamlit re-runs this whole script on every interaction, so without
    caching we would re-fit the TF-IDF vectorizer on each message.
    """
    return FAQMatcher()


def format_reply(response: Response) -> str:
    """Turn a Response into the markdown text shown in the chat bubble."""
    text = response.answer
    if response.related:
        heading = "You might also want to ask" if response.confident else "Did you mean"
        suggestions = "\n".join(f"- {m.question}" for m in response.related)
        text += f"\n\n**{heading}:**\n{suggestions}"
    return text


def init_state() -> None:
    """Create the chat history and last-response slots on first load."""
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": GREETING}]
    if "last_response" not in st.session_state:
        st.session_state.last_response = None


def render_sidebar(response: Response | None, threshold: float) -> None:
    """Show the matched FAQ and similarity score in the sidebar."""
    with st.sidebar:
        st.header("Match details")
        if response is None:
            st.caption("Ask a question to see which FAQ it matches.")
        else:
            match = response.match
            if response.confident or match.score >= MIN_RELATED_SCORE:
                st.markdown("**Closest FAQ**")
                st.write(match.question)
                st.markdown(f"**Category:** {match.category}")
            else:
                st.markdown("**Closest FAQ**")
                st.write("No similar FAQ found")
            st.markdown(f"**Similarity score:** {match.score:.2f}")
            # Cosine similarity is already between 0 and 1, so it fits a progress bar.
            st.progress(min(max(match.score, 0.0), 1.0))
            if response.confident:
                st.success("Confident match: answer shown")
            else:
                st.warning("Below threshold: fallback shown")
        st.caption(f"Confidence threshold: {threshold}")

        st.divider()
        if st.button("Clear chat"):
            st.session_state.messages = [{"role": "assistant", "content": GREETING}]
            st.session_state.last_response = None
            st.rerun()


def main() -> None:
    """Draw the chat page and handle new user messages."""
    st.set_page_config(page_title="FAQ Chatbot", page_icon="💬")
    st.title("💬 Store FAQ Chatbot")

    init_state()
    matcher = load_matcher()

    # chat_input is pinned to the bottom of the page by Streamlit.
    query = st.chat_input("Type your question...")
    if query:
        response = matcher.ask(query)
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append({"role": "assistant", "content": format_reply(response)})
        st.session_state.last_response = response

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    render_sidebar(st.session_state.last_response, matcher.threshold)


if __name__ == "__main__":
    main()