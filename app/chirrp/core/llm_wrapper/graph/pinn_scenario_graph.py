"""
LLM semantic layer for the PINN surrogate (chat-to-model translation).

Translates a stakeholder's natural-language instruction into a validated
pumping-change scenario for the surrogate, and summarizes simulation results
in plain planning language. This implements the "social layer" of the
Hydro-AI framework: parse -> validate against the surrogate's trained
envelope -> explain.

Pipeline mirrors modflow_autofill_graph.py: LLM parse with regex fallback,
then bounds validation. All steps report wall-clock latency.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class PinnScenarioState(TypedDict, total=False):
    instruction: str
    percent_change: Optional[float]
    llm_error: str
    is_valid: bool
    error: str
    suggestion: str
    source: str


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _to_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _regex_percent(instruction: str) -> Optional[float]:
    """Fallback: signed percent near pumping keywords, sign from verbs."""
    if not instruction:
        return None
    lowered = instruction.lower()
    number = re.search(r"[-+]?\d+(?:\.\d+)?", lowered)
    if not number:
        return None
    value = float(number.group(0))

    decrease_words = ("reduce", "decrease", "lower", "cut", "less", "restrict")
    increase_words = ("increase", "raise", "boost", "more", "expand", "grow")
    if value > 0 and any(w in lowered for w in decrease_words) and \
            not any(w in lowered for w in increase_words):
        value = -value
    return value


def _resolve_api_key(
    api_key: Optional[str],
    gemini_api_file: Optional[str],
    env_var: str = "GOOGLE_API_KEY",
) -> Optional[str]:
    if api_key and api_key.strip():
        return api_key.strip()
    env_key = os.environ.get(env_var)
    if env_key and env_key.strip():
        return env_key.strip()
    if gemini_api_file and os.path.isfile(gemini_api_file):
        with open(gemini_api_file, "r", encoding="utf-8") as f:
            value = f.read().strip()
            if value:
                return value
    return None


def _make_llm(llm_name: str, api_key: Optional[str],
              gemini_api_file: Optional[str],
              max_output_tokens: int = 512) -> Optional[ChatGoogleGenerativeAI]:
    key = _resolve_api_key(api_key=api_key, gemini_api_file=gemini_api_file)
    if not key:
        return None
    os.environ.setdefault("GOOGLE_API_KEY", key)
    return ChatGoogleGenerativeAI(
        model=llm_name,
        temperature=0,
        max_retries=2,
        max_output_tokens=max_output_tokens,
    )


def parse_pumping_instruction(
    instruction: str,
    min_percent: float,
    max_percent: float,
    llm_name: str = "gemini-2.5-flash-lite",
    gemini_api_file: str = "./gemini_api.txt",
    api_key: Optional[str] = None,
) -> dict[str, Any]:
    """NL instruction -> validated pumping percent change.

    Sign convention: positive = more pumping (e.g. "increase pumping by 10%"
    => +10), negative = less. Bounds come from the surrogate's trained Q
    envelope (PinnConfig.percent_bounds()).

    Returns {is_valid, percent_change, error, suggestion, source, llm_error,
    latency_ms}.
    """
    llm = _make_llm(llm_name, api_key, gemini_api_file, max_output_tokens=256)

    parse_prompt = ChatPromptTemplate.from_template(
        """
Extract the requested change in groundwater PUMPING from the instruction.

Return JSON only with this exact schema:
{{
  "percent_change": number | null
}}

Rules:
- Positive = MORE pumping (e.g. "increase pumping by 10%" -> 10).
- Negative = LESS pumping (e.g. "reduce pumping by 15%" -> -15,
  "cut pumping by a quarter" -> -25).
- Convert fractions/words to percent ("half" -> 50, "a third" -> 33.3).
- Use null if the instruction does not mention a pumping change.

Instruction: {instruction}
"""
    )

    def node_parse_llm(state: PinnScenarioState) -> PinnScenarioState:
        if llm is None:
            state["llm_error"] = "No API key available for LLM parse"
            return state
        try:
            msg = parse_prompt.format(instruction=state.get("instruction", ""))
            resp = llm.invoke(msg)
            content = getattr(resp, "content", str(resp))
            parsed = _extract_json(content)
            state["percent_change"] = _to_float_or_none(
                parsed.get("percent_change"))
            state["source"] = "llm"
        except Exception as exc:
            state["llm_error"] = f"LLM parse failed: {exc}"
        return state

    def node_regex_fallback(state: PinnScenarioState) -> PinnScenarioState:
        if state.get("percent_change") is not None:
            return state
        value = _regex_percent(state.get("instruction", ""))
        if value is not None:
            state["percent_change"] = value
            state["source"] = "regex" if state.get("source") != "llm" \
                else "llm+regex"
        return state

    def node_validate(state: PinnScenarioState) -> PinnScenarioState:
        pct = state.get("percent_change")
        errors: list[str] = []
        suggestion = ""

        if pct is None:
            errors.append(
                "Could not detect a pumping change in the instruction.")
            suggestion = ('Try e.g. "reduce pumping by 15%" or '
                          '"increase pumping by 10%".')
        elif not (min_percent <= pct <= max_percent):
            errors.append(
                f"A pumping change of {pct:+.1f}% falls outside the "
                f"surrogate's trained envelope "
                f"[{min_percent:+.1f}%, {max_percent:+.1f}%]. Predictions "
                f"outside the training range are not physically reliable.")
            clamped = max(min(pct, max_percent), min_percent)
            suggestion = (f'Try a change within range, e.g. '
                          f'"{"increase" if clamped > 0 else "reduce"} '
                          f'pumping by {abs(clamped):.0f}%".')

        state["is_valid"] = len(errors) == 0
        state["error"] = " | ".join(errors)
        state["suggestion"] = suggestion
        return state

    graph = StateGraph(PinnScenarioState)
    graph.add_node("parse_llm", node_parse_llm)
    graph.add_node("regex_fallback", node_regex_fallback)
    graph.add_node("validate", node_validate)
    graph.add_edge(START, "parse_llm")
    graph.add_edge("parse_llm", "regex_fallback")
    graph.add_edge("regex_fallback", "validate")
    graph.add_edge("validate", END)
    app = graph.compile()

    start = time.perf_counter()
    output = app.invoke({"instruction": instruction})
    latency_ms = (time.perf_counter() - start) * 1000.0

    return {
        "is_valid": bool(output.get("is_valid", False)),
        "percent_change": output.get("percent_change"),
        "error": output.get("error", ""),
        "suggestion": output.get("suggestion", ""),
        "source": output.get("source", "none"),
        "llm_error": output.get("llm_error", ""),
        "latency_ms": latency_ms,
    }


def summarize_scenario_result(
    instruction: str,
    scenario: dict[str, Any],
    llm_name: str = "gemini-2.5-flash-lite",
    gemini_api_file: str = "./gemini_api.txt",
    api_key: Optional[str] = None,
) -> dict[str, Any]:
    """Plain-language summary of a surrogate run for stakeholders.

    Per the framework paper, a good response explains the assumptions used and
    the limits of validity — not just numbers. Falls back to a template if no
    LLM is available.

    ``scenario`` should carry: percent_change, q_rate, max_drawdown,
    head_min, well_drawdown_final, latency_ms.
    """
    llm = _make_llm(llm_name, api_key, gemini_api_file, max_output_tokens=512)

    facts = (
        f"- Requested change: {scenario['percent_change']:+.1f}% pumping "
        f"(rate {scenario['q_rate']:,.0f} m³/day)\n"
        f"- Maximum drawdown anywhere: {scenario['max_drawdown']:.2f} m "
        f"(initial head 90 m)\n"
        f"- Drawdown at the well after 30 days: "
        f"{scenario['well_drawdown_final']:.2f} m\n"
        f"- Minimum head: {scenario['head_min']:.2f} m\n"
        f"- Computed by a physics-informed neural network surrogate in "
        f"{scenario['latency_ms']:.0f} ms"
    )

    fallback = (
        f"Scenario applied: {scenario['percent_change']:+.1f}% pumping "
        f"(Q = {scenario['q_rate']:,.0f} m³/day). The surrogate predicts a "
        f"maximum drawdown of {scenario['max_drawdown']:.2f} m, with "
        f"{scenario['well_drawdown_final']:.2f} m at the well after 30 days. "
        f"Assumptions: single-well unconfined aquifer (1 km², K = 33.33 "
        f"m/day, Sy = 0.10), constant-head boundaries at 90 m north and "
        f"south; results are valid only within the trained pumping range."
    )

    if llm is None:
        return {"summary": fallback, "source": "template", "latency_ms": 0.0}

    prompt = ChatPromptTemplate.from_template(
        """
You are a groundwater planning assistant in a participatory water-resources
workshop. Summarize this simulation result for non-technical stakeholders in
3–5 sentences. State what was simulated, the key outcome (drawdown), one
sentence on assumptions (single-well unconfined aquifer surrogate, fixed
boundary heads at 90 m), and one on validity limits (trained pumping range
only). No headings, no bullet lists, no JSON.

User request: {instruction}

Simulation facts:
{facts}
"""
    )

    start = time.perf_counter()
    try:
        resp = llm.invoke(prompt.format(instruction=instruction, facts=facts))
        text = (getattr(resp, "content", "") or "").strip()
        latency_ms = (time.perf_counter() - start) * 1000.0
        if not text:
            return {"summary": fallback, "source": "template",
                    "latency_ms": latency_ms}
        return {"summary": text, "source": "llm", "latency_ms": latency_ms}
    except Exception:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {"summary": fallback, "source": "template",
                "latency_ms": latency_ms}


def summarize_session(
    messages: list[dict[str, Any]],
    llm_name: str = "gemini-2.5-flash-lite",
    gemini_api_file: str = "./gemini_api.txt",
    api_key: Optional[str] = None,
) -> dict[str, Any]:
    """Plain-language recap of what a planning session achieved.

    Lets a stakeholder see prior work at a glance without re-sending the whole
    thread to the model. ``messages`` is the ordered list of session messages,
    each a dict with at least ``role`` and ``content`` (assistant rows may also
    carry ``scenario_name``). Falls back to a deterministic template when no LLM
    key is available.

    Returns {summary, source, latency_ms}.
    """
    transcript_lines: list[str] = []
    scenarios: list[str] = []
    for m in messages:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        transcript_lines.append(f"{role.upper()}: {content}")
        name = m.get("scenario_name")
        if name:
            scenarios.append(str(name))
    transcript = "\n".join(transcript_lines)

    if not transcript:
        return {"summary": "This session has no messages yet.",
                "source": "template", "latency_ms": 0.0}

    fallback = (
        f"This session covered {len(scenarios)} saved scenario(s)"
        + (": " + ", ".join(scenarios) if scenarios else "")
        + f". It contains {len(transcript_lines)} messages exchanged between "
        f"the stakeholder and the planning assistant."
    )

    llm = _make_llm(llm_name, api_key, gemini_api_file, max_output_tokens=400)
    if llm is None:
        return {"summary": fallback, "source": "template", "latency_ms": 0.0}

    prompt = ChatPromptTemplate.from_template(
        """
You are a groundwater planning assistant. Recap what THIS participatory
planning session has achieved so far, for a non-technical stakeholder
returning to their work. In 3-5 sentences: which pumping scenarios were
explored, the key outcomes (e.g. drawdown), and any decision or open question.
No headings, no bullet lists, no JSON.

Session transcript:
{transcript}
"""
    )

    start = time.perf_counter()
    try:
        resp = llm.invoke(prompt.format(transcript=transcript[:8000]))
        text = (getattr(resp, "content", "") or "").strip()
        latency_ms = (time.perf_counter() - start) * 1000.0
        if not text:
            return {"summary": fallback, "source": "template",
                    "latency_ms": latency_ms}
        return {"summary": text, "source": "llm", "latency_ms": latency_ms}
    except Exception:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {"summary": fallback, "source": "template",
                "latency_ms": latency_ms}
