"""
myagv_lab/phase3_delivery/pddl_planner/llm_translator.py
=========================================================
Module 1 of 4 in the Phase 3 pipeline.

Translates a natural language task description into a PDDL problem
file by calling the DeepSeek API (deepseek-chat).

The PDDL domain is fixed (imported from domain.py).
The LLM only generates the problem file — objects, init, and goal.

Pipeline position
-----------------
  NL text  ──►  [THIS MODULE]  ──►  (domain_str, problem_str)
"""

from __future__ import annotations

import os
import re
import logging
from pathlib import Path

log = logging.getLogger("llm_translator")

# ── Domain import ─────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from myagv_lab.phase3_delivery.pddl_planner.domain import (
    DOMAIN_PDDL, ACTION_DESCRIPTIONS, KNOWN_LOCATIONS, FIXED_OBJECTS,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

def _build_system_prompt() -> str:
    action_block = "\n".join(
        f"  - {name}: {desc}"
        for name, desc in ACTION_DESCRIPTIONS.items()
    )
    fixed_block = "\n".join(
        f"  - {name} ({typ})" for name, typ in FIXED_OBJECTS.items()
    )
    loc_block = "\n".join(f"  - {l}" for l in KNOWN_LOCATIONS)

    return f"""You are a PDDL problem generator for a mobile robot delivery system.

== Fixed objects (always present) ==
{fixed_block}

== Known locations ==
{loc_block}
(You may introduce new location names if the user mentions them.)

== Available actions ==
{action_block}

== PDDL Domain ==
{DOMAIN_PDDL}

== Your task ==
Given a natural language task description, output ONLY a valid PDDL
problem file.  Rules:
  1. Start with exactly: (define (problem ...
  2. All object types must match the domain types.
  3. Initial state: reflect the physical starting conditions.
     - agv1 starts at home unless stated otherwise.
     - cobot1 is always at loading_area.
     - charger_station has a charger (charger-at charger_station).
  4. Goal: capture exactly what the user wants to be TRUE at the end.
  5. Packages are named package_A, package_B, etc.
  6. Do NOT include markdown fences, explanations, or any text other
     than the raw PDDL.
  7. Do NOT invent actions or predicates not in the domain.
"""

_SYSTEM_PROMPT: str = _build_system_prompt()


# ═══════════════════════════════════════════════════════════════════════════════
#  TRANSLATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def natural_language_to_pddl(task_description: str) -> tuple[str, str]:
    """
    Translate a natural language task description into a PDDL problem.

    Args
    ----
    task_description : str
        Plain English description of the delivery task.

    Returns
    -------
    (domain_pddl, problem_pddl) : tuple[str, str]
        Both as raw PDDL strings ready for pyperplan.

    Raises
    ------
    RuntimeError
        If the API call fails or the response is clearly not PDDL.
    EnvironmentError
        If DEEPSEEK_API_KEY is not set.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "DEEPSEEK_API_KEY is not set.\n"
            "  export DEEPSEEK_API_KEY='sk-...'\n"
            "  Or use --use-fallback to skip the LLM step."
        )

    from openai import OpenAI

    log.info(f"[LLM] Task: {task_description!r}")
    log.info("[LLM] Calling DeepSeek API (deepseek-chat) …")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    message = client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=1024,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": task_description},
        ],
    )

    raw = message.choices[0].message.content.strip()

    # Strip accidental markdown fences
    raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
    raw = raw.strip("`").strip()

    # Basic sanity check
    if not raw.startswith("(define"):
        raise RuntimeError(
            f"LLM response does not look like PDDL:\n{raw[:300]}"
        )

    log.info("[LLM] Generated problem.pddl:")
    for line in raw.splitlines():
        log.info(f"  {line}")

    return DOMAIN_PDDL, raw


# ═══════════════════════════════════════════════════════════════════════════════
#  FALLBACK  (for testing without an API key)
# ═══════════════════════════════════════════════════════════════════════════════

_FALLBACK_PROBLEMS: dict[str, str] = {
    "deliver_A": """\
(define (problem delivery-1)
  (:domain agv-delivery)
  (:objects
    agv1          - robot
    cobot1        - arm
    package_A     - package
    home          - location
    loading_area  - location
    delivery_area - location
  )
  (:init
    (at         agv1   home)
    (arm-at     cobot1 loading_area)
    (package-at package_A loading_area)
  )
  (:goal (and
    (delivered  package_A)
    (at         agv1 home)
  ))
)""",
    "deliver_AB": """\
(define (problem delivery-2)
  (:domain agv-delivery)
  (:objects
    agv1          - robot
    cobot1        - arm
    package_A     - package
    package_B     - package
    home          - location
    loading_area  - location
    delivery_area - location
  )
  (:init
    (at         agv1   home)
    (arm-at     cobot1 loading_area)
    (package-at package_A loading_area)
    (package-at package_B loading_area)
  )
  (:goal (and
    (delivered  package_A)
    (delivered  package_B)
    (at         agv1 home)
  ))
)""",
    "recharge_then_deliver": """\
(define (problem delivery-3)
  (:domain agv-delivery)
  (:objects
    agv1             - robot
    cobot1           - arm
    package_A        - package
    home             - location
    loading_area     - location
    delivery_area    - location
    charger_station  - location
  )
  (:init
    (at           agv1   home)
    (arm-at       cobot1 loading_area)
    (package-at   package_A loading_area)
    (charger-at   charger_station)
  )
  (:goal (and
    (charged      agv1)
    (delivered    package_A)
    (at           agv1 home)
  ))
)""",
}


def fallback_pddl(scenario: str = "deliver_A") -> tuple[str, str]:
    """Return a hard-coded PDDL problem without calling the LLM."""
    if scenario not in _FALLBACK_PROBLEMS:
        raise ValueError(
            f"Unknown scenario {scenario!r}. "
            f"Available: {list(_FALLBACK_PROBLEMS.keys())}"
        )
    problem = _FALLBACK_PROBLEMS[scenario]
    log.info(f"[LLM] Using fallback problem: {scenario!r}")
    return DOMAIN_PDDL, problem


# ═══════════════════════════════════════════════════════════════════════════════
#  QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    p = argparse.ArgumentParser()
    p.add_argument("--task", default=(
        "Go to the loading area, pick up package_A, deliver it to the "
        "delivery area, and return home."
    ))
    p.add_argument("--fallback", action="store_true",
                   help="Use hard-coded PDDL (no API key needed)")
    p.add_argument("--scenario", default="deliver_A")
    args = p.parse_args()

    if args.fallback:
        domain, problem = fallback_pddl(args.scenario)
    else:
        domain, problem = natural_language_to_pddl(args.task)

    print("\n=== DOMAIN ===")
    print(domain)
    print("\n=== PROBLEM ===")
    print(problem)
