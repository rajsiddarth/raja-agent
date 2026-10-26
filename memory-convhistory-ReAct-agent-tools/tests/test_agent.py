"""Unit tests for agent.py — mocks the Anthropic client, tests the agent loop."""

from unittest.mock import MagicMock, patch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent import run_agent


def make_text_response(text: str):
    """Simulate a final Claude response (end_turn)."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [block]
    return response


def make_tool_response(tool_name: str, tool_input: dict, tool_id: str = "tu_123"):
    """Simulate Claude requesting a tool call."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.input = tool_input
    block.id = tool_id
    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [block]
    return response


# ---------------------------------------------------------------------------
# Basic loop behaviour
# ---------------------------------------------------------------------------

@patch("agent.client")
def test_end_turn_returns_reply(mock_client):
    mock_client.messages.create.return_value = make_text_response("Apple is at $150.")
    reply, history = run_agent("What is Apple's price?", [])
    assert reply == "Apple is at $150."
    assert history[-1]["role"] == "assistant"


@patch("agent.client")
def test_history_grows_with_each_turn(mock_client):
    mock_client.messages.create.return_value = make_text_response("Hello!")
    _, history = run_agent("Hi", [])
    assert len(history) == 2  # user + assistant


@patch("agent.client")
def test_tool_use_then_end_turn(mock_client):
    """Claude calls get_quote, then produces a final answer."""
    mock_client.messages.create.side_effect = [
        make_tool_response("get_quote", {"symbol": "AAPL"}),
        make_text_response("Apple is at $150."),
    ]
    with patch("agent.TOOL_HANDLERS", {"get_quote": lambda _: {"price": 150.0}}):
        reply, history = run_agent("What is Apple's price?", [])

    assert reply == "Apple is at $150."
    assert mock_client.messages.create.call_count == 2


@patch("agent.client")
def test_unknown_tool_returns_error_result(mock_client):
    """An unknown tool name sends an error tool_result back to Claude."""
    mock_client.messages.create.side_effect = [
        make_tool_response("nonexistent_tool", {}),
        make_text_response("I could not complete that request."),
    ]
    with patch("agent.TOOL_HANDLERS", {}):
        reply, _ = run_agent("Do something", [])

    tool_result_msg = mock_client.messages.create.call_args_list[1][1]["messages"][-1]
    content = tool_result_msg["content"]
    assert content[0]["is_error"] is True


@patch("agent.client")
def test_on_tool_call_callback_fires(mock_client):
    """on_tool_call is invoked with the tool name."""
    mock_client.messages.create.side_effect = [
        make_tool_response("get_quote", {"symbol": "TSLA"}),
        make_text_response("Tesla is at $200."),
    ]
    calls = []
    with patch("agent.TOOL_HANDLERS", {"get_quote": lambda _: {"price": 200.0}}):
        run_agent("Tesla price?", [], on_tool_call=lambda name: calls.append(name))

    assert calls == ["get_quote"]


@patch("agent.client")
def test_existing_history_is_preserved(mock_client):
    """Prior history is passed through and extended."""
    prior = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]
    mock_client.messages.create.return_value = make_text_response("Sure.")
    _, history = run_agent("Follow up", prior)
    assert history[0] == prior[0]
    assert history[1] == prior[1]
    assert len(history) == 4


# ---------------------------------------------------------------------------
# ReAct-specific: think tool
# ---------------------------------------------------------------------------

@patch("agent.client")
def test_think_tool_fires_on_think_callback(mock_client):
    """on_think is called with the reasoning text when Claude uses the think tool."""
    mock_client.messages.create.side_effect = [
        make_tool_response("think", {"reasoning": "I need AAPL and MSFT data."}, "tu_think"),
        make_text_response("Here is the comparison."),
    ]
    thoughts = []
    run_agent("Compare Apple and Microsoft", [], on_think=lambda r: thoughts.append(r))
    assert thoughts == ["I need AAPL and MSFT data."]


@patch("agent.client")
def test_think_tool_does_not_call_data_handler(mock_client):
    """The think tool is a no-op — it must not invoke any TOOL_HANDLERS entry."""
    mock_client.messages.create.side_effect = [
        make_tool_response("think", {"reasoning": "Let me think."}, "tu_think"),
        make_text_response("Done."),
    ]
    handler_called = []
    with patch("agent.TOOL_HANDLERS", {"think": lambda _: handler_called.append(True)}):
        run_agent("Think then answer", [])
    # think is handled inline in the loop, TOOL_HANDLERS["think"] must never be called
    assert handler_called == []


@patch("agent.client")
def test_refusal_returns_friendly_message(mock_client):
    """stop_reason == 'refusal' returns a user-friendly string instead of crashing."""
    response = MagicMock()
    response.stop_reason = "refusal"
    response.content = []
    mock_client.messages.create.return_value = response
    reply, _ = run_agent("Some refused question", [])
    assert "market data" in reply.lower()
    assert "(stopped: refusal)" not in reply
