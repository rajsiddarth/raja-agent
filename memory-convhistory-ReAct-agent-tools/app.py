"""Streamlit chat UI for the ReAct Finance Agent with conversation memory."""

import streamlit as st
from dotenv import load_dotenv
from agent import run_agent
from memory import load_history, save_history, list_sessions, delete_session

load_dotenv()

st.set_page_config(
    page_title="ReAct Finance Agent",
    page_icon="🧠",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Sidebar — session management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("💾 Sessions")

    sessions = list_sessions()
    session_names = [s["name"] for s in sessions]

    if sessions:
        st.caption("Saved sessions:")
        for s in sessions:
            st.markdown(f"**{s['name']}** · {s['turns']} turn(s)")

    st.divider()
    new_session_name = st.text_input("Session name", value="default", key="session_input")

    col1, col2 = st.columns(2)
    load_btn = col1.button("Load", use_container_width=True)
    clear_btn = col2.button("New", use_container_width=True)

    if load_btn and new_session_name:
        st.session_state.session_name = new_session_name
        st.session_state.history = load_history(new_session_name)
        st.session_state.messages = []
        st.rerun()

    if clear_btn:
        st.session_state.session_name = new_session_name
        st.session_state.history = []
        st.session_state.messages = []
        st.rerun()

    if sessions and st.button("🗑 Delete this session", use_container_width=True):
        delete_session(st.session_state.get("session_name", "default"))
        st.session_state.history = []
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "session_name" not in st.session_state:
    st.session_state.session_name = "default"
if "history" not in st.session_state:
    st.session_state.history = load_history(st.session_state.session_name)
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🧠 ReAct Finance Agent")
st.caption(
    f"Claude · Reason + Act · yfinance · Real-time market data · "
    f"Session: **{st.session_state.session_name}**"
)

prior_turns = sum(1 for m in st.session_state.history if isinstance(m, dict) and m.get("role") == "user")
if prior_turns > 0 and not st.session_state.messages:
    st.info(f"Resumed session '{st.session_state.session_name}' — {prior_turns} prior turn(s) in memory.")

# ---------------------------------------------------------------------------
# Render existing conversation
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Suggestion buttons (only on empty chat)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
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
            save_history(st.session_state.session_name, st.session_state.history)

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
