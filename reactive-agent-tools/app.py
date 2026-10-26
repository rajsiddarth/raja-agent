"""Streamlit chat UI for the Finance Agent."""

import streamlit as st
from dotenv import load_dotenv
from agent import run_agent

load_dotenv()

st.set_page_config(
    page_title="Finance Agent",
    page_icon="📈",
    layout="centered",
)

st.title("📈 Finance Agent")
st.caption("Claude · yfinance · Real-time market data")

if "history" not in st.session_state:
    st.session_state.history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tools"):
            with st.expander(f"{len(msg['tools'])} tool(s) called", expanded=False):
                for t in msg["tools"]:
                    st.caption(f"→ {t}")

SUGGESTIONS = [
    "What is Apple's current price?",
    "Show me Tesla fundamentals",
    "What's the latest news on NVDA?",
    "How has Microsoft performed over the last month?",
]

if not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(2)
    for i, s in enumerate(SUGGESTIONS):
        if cols[i % 2].button(s, use_container_width=True):
            st.session_state["pending"] = s
            st.rerun()

prompt = st.chat_input("Ask about your portfolio, quotes, or market data...")

if "pending" in st.session_state:
    prompt = st.session_state.pop("pending")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    tools_called: list[str] = []

    with st.chat_message("assistant"):
        status = st.status("Fetching market data...", expanded=False)
        try:
            reply, st.session_state.history = run_agent(
                prompt,
                st.session_state.history,
                on_tool_call=lambda name: (
                    tools_called.append(name),
                    status.update(label=f"→ {name}"),
                ),
            )
            label = f"Done · {len(tools_called)} tool(s) called" if tools_called else "Done"
            status.update(label=label, state="complete", expanded=False)
        except Exception as exc:
            reply = f"Something went wrong: {type(exc).__name__}: {exc}"
            status.update(label="Error", state="error")

        st.markdown(reply)
        if tools_called:
            with st.expander(f"{len(tools_called)} tool(s) called", expanded=False):
                for t in tools_called:
                    st.caption(f"→ {t}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "tools": tools_called,
    })
