"""Fast LLM-based VC decision-maker for The Paradox of Inclusion.

NetLogo supplies every modeled attribute for every eligible VC candidate.
The investor keeps an episodic memory for the current simulation run.
No fixed score, ranking, numerical weight, or gender quota is used.

Requirements: pip install ollama; Ollama running; ollama pull llama3.2:3b
Optional faster model: ollama pull llama3.2:1b, then set VC_LLM_MODEL=llama3.2:1b
"""
from __future__ import annotations

import json
import os
import re
import time
from statistics import mean
from typing import Any

import ollama

MODEL_NAME = os.getenv("VC_LLM_MODEL", "llama3.2:3b")
TEMPERATURE = float(os.getenv("VC_LLM_TEMPERATURE", "0.1"))
NUM_CTX = int(os.getenv("VC_LLM_NUM_CTX", "8192"))
NUM_PREDICT = int(os.getenv("VC_LLM_NUM_PREDICT", "24"))
RETRY_NUM_PREDICT = int(os.getenv("VC_LLM_RETRY_NUM_PREDICT", "8"))

# Exactly the 25 fields sent by make-vc-candidate-packet in NetLogo.
FIELD_ORDER = [
    "id", "gender", "d", "funds", "social_net",
    "successful_male_neighbors", "successful_female_neighbors",
    "successful_mixed_neighbors", "accelerated", "initial_state",
    "mating_acc", "accelerated_period", "tie_vc", "selection1",
    "selection2", "meeting", "mating", "meet_prob", "mat_prob",
    "want_acc", "want_vc", "active_vc_neighbors", "active_vc_male",
    "active_vc_female", "active_vc_mixed",
]
GENDER_LABELS = {0: "male-founded", 1: "female-founded", 2: "mixed-founded"}
VC_MEMORY: list[dict[str, Any]] = []
LAST_ERROR = ""
LAST_DURATION = 0.0
LAST_CANDIDATE_COUNT = 0


def reset_memory() -> None:
    """Clear the VC's episodic memory at the start of each simulation replicate."""
    global LAST_ERROR, LAST_DURATION, LAST_CANDIDATE_COUNT
    VC_MEMORY.clear()
    LAST_ERROR = ""
    LAST_DURATION = 0.0
    LAST_CANDIDATE_COUNT = 0
    print("[VC] Memory reset.", flush=True)


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    low = value.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_candidate_packet(packet: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for record in packet.split(";"):
        if not record.strip():
            continue
        values = record.split("|")
        if len(values) != len(FIELD_ORDER):
            raise ValueError(
                f"NetLogo sent {len(values)} values per candidate; Python expects "
                f"{len(FIELD_ORDER)}. Candidate data: {record[:240]!r}"
            )
        item = dict(zip(FIELD_ORDER, (_parse_scalar(v) for v in values)))
        item["gender_label"] = GENDER_LABELS.get(item["gender"], "unknown")
        candidates.append(item)
    if not candidates:
        raise ValueError("NetLogo sent an empty candidate set.")
    ids = [int(c["id"]) for c in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate IDs are duplicated.")
    return candidates


def _memory_for_prompt() -> dict[str, Any]:
    """Retrieve all-run descriptive experience plus five detailed recent episodes."""
    if not VC_MEMORY:
        return {"number_of_previous_investments": 0, "experience": "No previous investments in this run."}

    def avg(field: str) -> float | None:
        vals = [x[field] for x in VC_MEMORY if isinstance(x.get(field), (int, float)) and not isinstance(x.get(field), bool)]
        return round(mean(vals), 2) if vals else None

    summary = {
        "investments_recorded": len(VC_MEMORY),
        "average_human_capital_d": avg("d"),
        "average_funds_at_investment": avg("funds"),
        "average_successful_social_network": avg("social_net"),
        "average_active_vc_neighbors": avg("active_vc_neighbors"),
        "accelerated_startups_funded": sum(bool(x.get("accelerated")) for x in VC_MEMORY),
        "gender_composition_of_past_investments": {
            label: sum(1 for x in VC_MEMORY if x["gender_label"] == label)
            for label in GENDER_LABELS.values()
        },
    }
    recent = [
        {
            "tick": x["tick"], "id": x["id"], "gender": x["gender_label"],
            "d": x["d"], "funds": x["funds"], "social_net": x["social_net"],
            "successful_neighbors_by_gender": [
                x["successful_male_neighbors"], x["successful_female_neighbors"],
                x["successful_mixed_neighbors"],
            ],
            "accelerated": x["accelerated"],
            "active_vc_neighbors_by_gender": [x["active_vc_male"], x["active_vc_female"], x["active_vc_mixed"]],
        }
        for x in VC_MEMORY[-5:]
    ]
    return {"all_run_descriptive_experience": summary, "five_most_recent_investments": recent}


def _call_ollama(system_prompt: str, user_prompt: str) -> str:
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT,
            "num_ctx": NUM_CTX,
        },
    )
    return response["message"]["content"].strip()


def _extract_id(raw: str) -> int:
    """Accept JSON selected_id or a bare integer, but never invent a choice."""
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return int(obj["selected_id"])
        if isinstance(obj, int):
            return obj
    except (json.JSONDecodeError, TypeError, ValueError, KeyError):
        pass
    match = re.search(r'"selected_id"\s*:\s*"?(\d+)"?', raw)
    if match:
        return int(match.group(1))
    match = re.fullmatch(r"\s*(\d+)\s*", raw)
    if match:
        return int(match.group(1))
    raise ValueError(f"The LLM response did not contain a valid selected_id: {raw[:300]!r}")


def select_investment(packet: str, tick: int) -> int:
    """Select exactly one available startup and store its full selection-time profile."""
    global LAST_ERROR, LAST_DURATION, LAST_CANDIDATE_COUNT
    started = time.perf_counter()
    LAST_ERROR = ""
    try:
        candidates = parse_candidate_packet(packet)
        LAST_CANDIDATE_COUNT = len(candidates)
        allowed_ids = {int(x["id"]) for x in candidates}

        # One candidate needs no model inference; there is no choice to compare.
        if len(candidates) == 1:
            selected_id = int(candidates[0]["id"])
            raw = "single eligible candidate"
            decision_source = "single_candidate"
        else:
            system = (
                "You are the VC investor in an agent-based simulation. Choose exactly ONE startup "
                "from the CURRENT candidate list. The only IDs you are permitted to return are "
                f"{sorted(allowed_ids)}. IDs in investor memory are historical and are NOT eligible. "
                "Never return an ID from memory unless it also appears in the current candidate list. "
                "Use your own contextual judgment; do not calculate a score, rank, fixed weights, "
                "thresholds, or gender quotas. Do not apply a predetermined priority to any attribute. "
                "The list contains every candidate attribute generated by the simulation. Gender codes: "
                "0 male-founded, 1 female-founded, 2 mixed-founded. Variable order is supplied separately. "
                "d=human capital; funds=current financial resources; social_net and the three successful-"
                "neighbor counts describe historical VC-successful social ties; the active_vc fields "
                "describe currently active VC-linked neighbors; the remaining fields are the startup's "
                "modeled accelerator, VC, meeting, and interaction states/probabilities. Use investor "
                "memory as experience about previously funded startups, not as a rigid rule. Memory "
                "describes their characteristics at the time of investment, not later success. Gender is "
                "visible but never infer competence from gender or use stereotypes. Do not invent business "
                "facts absent from the data. Return only JSON: {\"selected_id\": integer}."
            )
            # Array records avoid repeating 25 field names for each candidate,
            # cutting prompt size while transmitting every original value.
            rows = [[item[name] for name in FIELD_ORDER] for item in candidates]
            payload = {
                "tick": int(tick),
                "allowed_candidate_ids": sorted(allowed_ids),
                "candidate_field_order": FIELD_ORDER,
                "candidate_records": rows,
                "vc_memory": _memory_for_prompt(),
            }
            user_prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            raw = _call_ollama(system, user_prompt)
            decision_source = "llm"
            try:
                selected_id = _extract_id(raw)
                if selected_id not in allowed_ids:
                    raise ValueError(f"LLM chose ID {selected_id}, not in current candidate IDs {sorted(allowed_ids)}")
            except ValueError as first_error:
                # One short corrective attempt: explicitly show the ONLY valid IDs.
                retry_system = (
                    "Your previous answer was invalid because it selected an unavailable startup. "
                    "Choose exactly one ID from the allowed IDs below. Do not use IDs in memory. "
                    "Do not explain. Do not invent IDs. Return JSON only."
                )
                retry_user = (
                    f"CURRENT ALLOWED IDs: {sorted(allowed_ids)}\n"
                    f"Your previous answer: {raw[:160]}\n"
                    f"Candidate data (field order then records): {user_prompt}\n"
                    f"Return {{\"selected_id\": <one of {sorted(allowed_ids)}>}}."
                )
                try:
                    retry_raw = _call_ollama(retry_system, retry_user)
                    selected_id = _extract_id(retry_raw)
                    if selected_id not in allowed_ids:
                        raise ValueError(
                            f"Retry chose ID {selected_id}, not in current candidate IDs {sorted(allowed_ids)}"
                        )
                    raw = retry_raw
                    decision_source = "llm_retry"
                except Exception as retry_error:
                    # Ask NetLogo to draw the fallback candidate. This preserves the
                    # model's own seeded RNG and avoids consuming random draws on valid LLM turns.
                    raw = f"initial_error={first_error}; retry_error={retry_error}"
                    LAST_DURATION = time.perf_counter() - started
                    print(
                        f"[VC WARNING] tick={tick}; LLM returned invalid IDs twice; "
                        f"requesting NetLogo-seeded random fallback from {sorted(allowed_ids)}; {raw[:300]}",
                        flush=True,
                    )
                    return -2

        if selected_id not in allowed_ids:
            # Defensive guard: this should be unreachable after the validation above.
            raise ValueError(f"Internal error: selected ID {selected_id} is not in {sorted(allowed_ids)}")

        selected = next(x for x in candidates if int(x["id"]) == selected_id)
        record = dict(selected)
        record["tick"] = int(tick)
        # Store all raw characteristics for later retrieval; no performance outcome is inferred.
        VC_MEMORY.append(record)
        LAST_DURATION = time.perf_counter() - started
        print(
            f"[VC] tick={tick}; candidates={len(candidates)}; selected={selected_id}; "
            f"model={MODEL_NAME}; source={decision_source if len(candidates) > 1 else 'single_candidate'}; "
            f"seconds={LAST_DURATION:.2f}; memory={len(VC_MEMORY)}; response={raw[:160]!r}", flush=True
        )
        return selected_id
    except Exception as exc:
        LAST_DURATION = time.perf_counter() - started
        LAST_ERROR = f"{type(exc).__name__}: {exc}"
        print(f"[VC ERROR] tick={tick}; candidates={LAST_CANDIDATE_COUNT}; seconds={LAST_DURATION:.2f}; {LAST_ERROR}", flush=True)
        return -1


def record_fallback(packet: str, tick: int, selected_id: int) -> bool:
    """Record a fallback startup selected by NetLogo's seeded random generator."""
    global LAST_ERROR
    candidates = parse_candidate_packet(packet)
    allowed_ids = {int(x["id"]) for x in candidates}
    selected_id = int(selected_id)
    if selected_id not in allowed_ids:
        raise ValueError(
            f"NetLogo fallback ID {selected_id} is not in current candidate IDs {sorted(allowed_ids)}"
        )
    selected = next(x for x in candidates if int(x["id"]) == selected_id)
    record = dict(selected)
    record["tick"] = int(tick)
    record["decision_source"] = "netlogo_seeded_random_fallback"
    VC_MEMORY.append(record)
    LAST_ERROR = ""
    print(
        f"[VC] tick={tick}; candidates={len(candidates)}; selected={selected_id}; "
        f"source=netlogo_seeded_random_fallback; memory={len(VC_MEMORY)}",
        flush=True,
    )
    return True


def last_error() -> str:
    return LAST_ERROR or "Python returned -1 without recording a detailed exception."


def last_duration() -> float:
    return float(LAST_DURATION)


def memory_size() -> int:
    return len(VC_MEMORY)
