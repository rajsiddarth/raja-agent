"""Streamlit chat UI for the ReAct Finance Agent."""

import streamlit as st
from dotenv import load_dotenv
from agent import run_agent

load_dotenv()

st.set_page_config(
    page_title="ReAct Finance Agent",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 ReAct Finance Agent")
st.caption("Claude · Reason + Act · yfinance · Real-time market data")

if "history" not in st.session_state:
    st.session_state.history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("thoughts"):
            with st.expander("💭 Reasoning", expanded=False):
                for thought in msg["thoughts"]:
                    st.markdown(thought)
        st.markdown(msg["content"])
        if msg.get("tools"):
            with st.expander(f"{len(msg['tools'])} tool(s) called", expanded=False):
                for t in msg["tools"]:
                    st.caption(f"→ {t}")

SUGGESTIONS = [
    "Get Apple and Microsoft fundamentals",
    "Get Tesla and Ford fundamentals",
    "Get recent NVDA news headlines",
    "Get S&P 500 performance over the last month",
]

if not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(2)
    for i, s in enumerate(SUGGESTIONS):
        if cols[i % 2].button(s, use_container_width=True):
            st.session_state["pending"] = s
            st.rerun()

prompt = st.chat_input("Ask anything about markets, stocks, or your portfolio...")

if "pending" in st.session_state:
    prompt = st.session_state.pop("pending")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    tools_called: list[str] = []
    thoughts: list[str] = []

    with st.chat_message("assistant"):
        status = st.status("Thinking...", expanded=False)

        try:
            reply, st.session_state.history = run_agent(
                prompt,
                st.session_state.history,
                on_tool_call=lambda name: (
                    tools_called.append(name),
                    status.update(label=f"→ {name}"),
                ),
                on_think=lambda reasoning: (
                    thoughts.append(reasoning),
                    status.update(label="💭 Reasoning..."),
                ),
            )
            parts = []
            if thoughts:
                parts.append(f"{len(thoughts)} reasoning step(s)")
            if tools_called:
                parts.append(f"{len(tools_called)} tool(s) called")
            label = "Done · " + ", ".join(parts) if parts else "Done"
            status.update(label=label, state="complete", expanded=False)
        except Exception as exc:
            reply = f"Something went wrong: {type(exc).__name__}: {exc}"
            status.update(label="Error", state="error")

        if thoughts:
            with st.expander("💭 Reasoning", expanded=True):
                for thought in thoughts:
                    st.markdown(thought)

        st.markdown(reply)

        if tools_called:
            with st.expander(f"{len(tools_called)} tool(s) called", expanded=False):
                for t in tools_called:
                    st.caption(f"→ {t}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "tools": tools_called,
        "thoughts": thoughts,
    })
