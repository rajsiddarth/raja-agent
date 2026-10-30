"""LLM judge for the answer eval — grades what code cannot check.

For one agent answer it returns, as schema-validated JSON:
  - response_type : answered / abstained / partial (abstention and false-abstention metrics)
  - claims        : each factual claim in the answer, and whether the evidence supports it (faithfulness)
  - key_facts     : whether each expected fact is covered (correctness)
  - citations     : whether each section citation is supported by a passage from that section

The judge is a different, stronger model than the agent (claude-sonnet-4-6), so the agent is not graded by
itself. It only sees the evidence the agent's tools returned that turn, never the full filing, so
"supported" means "supported by what the agent actually read".
"""

import json

import anthropic

JUDGE_MODEL = "claude-opus-5-5"

_client = None

JUDGE_SYSTEM = """You grade answers from a finance research assistant that answers questions from SEC filings \
(10-K and 10-Q) and market data tools. You receive the user's question, the assistant's final answer, the \
evidence (everything the assistant's tools returned during this turn), and, when the question is answerable, \
the key facts a complete answer should contain.

Grade strictly and literally. Judge only against the evidence provided, not your own knowledge of the company.

1. response_type
   - "answered": the answer gives the requested information.
   - "abstained": the answer says the filing does not contain or support the requested information, or \
corrects a false premise in the question (e.g. "Apple did not cut its dividend").
   - "partial": the answer gives some of the requested information and says other parts are not available.

2. claims: list every factual claim the answer makes about the company, its filings, its business or market \
data. Skip restatements of the question, generic disclaimers, offers of further help, and the citations \
themselves. A claim is supported only if the evidence states it or directly implies it; numbers, dates and \
names must match. For each claim give a short quote or pointer to the supporting evidence, or explain what \
is missing.

3. key_facts: for each key fact given, covered is true only if the answer states it, paraphrase allowed, \
including every essential part (numbers, names, direction of change). Return the facts in the given order. \
If no key facts are given, return an empty list.

4. citations: for each place the answer attributes information to a specific section (e.g. "Item 1A"), say \
whether a passage from that section in the evidence supports the attributed information. If the answer cites \
no sections, return an empty list."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "response_type": {"type": "string", "enum": ["answered", "abstained", "partial"]},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "supported": {"type": "boolean"},
                    "evidence": {"type": "string"},
                },
                "required": ["claim", "supported", "evidence"],
                "additionalProperties": False,
            },
        },
        "key_facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string"},
                    "covered": {"type": "boolean"},
                },
                "required": ["fact", "covered"],
                "additionalProperties": False,
            },
        },
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "citation": {"type": "string"},
                    "attributed_claim": {"type": "string"},
                    "supported_by_cited_section": {"type": "boolean"},
                },
                "required": ["citation", "attributed_claim", "supported_by_cited_section"],
                "additionalProperties": False,
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["response_type", "claims", "key_facts", "citations", "notes"],
    "additionalProperties": False,
}


class JudgeError(Exception):
    """The judge could not produce a grade (refusal, truncation or invalid output)."""


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def build_prompt(question: str, answer: str, evidence: str, key_facts: list[str]) -> str:
    facts = "\n".join(f"- {f}" for f in key_facts) if key_facts else "(none: grade response_type, claims and citations)"
    return (
        f"<question>\n{question}\n</question>\n\n"
        f"<answer>\n{answer}\n</answer>\n\n"
        f"<evidence>\n{evidence}\n</evidence>\n\n"
        f"<key_facts>\n{facts}\n</key_facts>"
    )


def judge_answer(question: str, answer: str, evidence: str, key_facts: list[str], model: str = JUDGE_MODEL) -> dict:
    """Grade one answer. Returns the parsed grade plus {"usage": {...}, "model": ...}; raises JudgeError."""
    response = _get_client().beta.messages.create(
        model=model,
        max_tokens=16000,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": build_prompt(question, answer, evidence, key_facts)}],
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        # on a safety decline the API re-runs the request on Anthropic's recommended fallback model
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        category = response.stop_details.category if response.stop_details else None
        raise JudgeError(f"judge refused (category: {category})")
    if response.stop_reason == "max_tokens":
        raise JudgeError("judge output hit max_tokens")
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise JudgeError(f"judge returned no text (stop_reason: {response.stop_reason})")
    try:
        grade = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JudgeError(f"judge returned invalid JSON: {exc}") from exc

    grade["model"] = response.model  # differs from `model` if a fallback served the request
    grade["usage"] = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
    return grade
