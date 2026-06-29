"""
myagv_lab/phase3_delivery/pddl_planner/domain.py
=================================================
Single source of truth for the PDDL domain.

Both the LLM system prompt and the pyperplan solver import from here,
so any change to the domain is automatically reflected everywhere.
"""

DOMAIN_PDDL = """\
(define (domain agv-delivery)
  (:requirements :strips :typing)

  ;; ── Types ──────────────────────────────────────────────────────────────────
  (:types
    location   ;; e.g. home, loading_area, delivery_area, storage_area
    package    ;; a physical object to be transported
    robot      ;; the AGV
    arm        ;; the cobot manipulator
  )

  ;; ── Predicates ─────────────────────────────────────────────────────────────
  (:predicates
    (at           ?r - robot    ?l - location)  ;; robot is at location
    (package-at   ?p - package  ?l - location)  ;; package is at location
    (holding      ?r - robot    ?p - package)   ;; robot carries package
    (arm-at       ?a - arm      ?l - location)  ;; arm stationed at location
    (delivered    ?p - package)                 ;; package has been delivered
    (charged      ?r - robot)                   ;; robot battery is full
    (charger-at   ?l - location)               ;; charger exists at location
  )

  ;; ── Actions ────────────────────────────────────────────────────────────────

  ;; Navigate AGV from one location to another
  (:action navigate
    :parameters (?r - robot ?from - location ?to - location)
    :precondition (and
      (at ?r ?from)
    )
    :effect (and
      (not (at ?r ?from))
      (at ?r ?to)
    )
  )

  ;; Cobot arm loads a package from a fixed location onto the waiting AGV
  (:action load-package
    :parameters (?r - robot ?a - arm ?p - package ?l - location)
    :precondition (and
      (at ?r ?l)
      (arm-at ?a ?l)
      (package-at ?p ?l)
    )
    :effect (and
      (not (package-at ?p ?l))
      (holding ?r ?p)
    )
  )

  ;; AGV delivers the package at the destination
  (:action deliver-package
    :parameters (?r - robot ?p - package ?l - location)
    :precondition (and
      (at ?r ?l)
      (holding ?r ?p)
    )
    :effect (and
      (not (holding ?r ?p))
      (package-at ?p ?l)
      (delivered ?p)
    )
  )

  ;; Recharge AGV at a charger station
  (:action recharge
    :parameters (?r - robot ?l - location)
    :precondition (and
      (at ?r ?l)
      (charger-at ?l)
    )
    :effect (and
      (charged ?r)
    )
  )
)
"""

# Human-readable descriptions of each action (for the LLM system prompt)
ACTION_DESCRIPTIONS = {
    "navigate":        "Move the AGV (agv1) from one location to another.",
    "load-package":    "Have the cobot arm (cobot1) load a package onto the AGV at the loading_area.",
    "deliver-package": "Have the AGV deposit the package it is carrying at a destination location.",
    "recharge":        "Charge the AGV at a charger_station location.",
}

# All valid location names the LLM may use
KNOWN_LOCATIONS = [
    "home",
    "loading_area",
    "delivery_area",
    "storage_area",
    "charger_station",
]

# Fixed objects in every problem
FIXED_OBJECTS = {
    "agv1":   "robot",
    "cobot1": "arm",
}
