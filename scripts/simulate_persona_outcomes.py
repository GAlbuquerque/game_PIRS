#!/usr/bin/env python3
"""Reproducibly score two 16-quarter terms for every automated persona."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "game_PIRS"))

from economy import Economy  # noqa: E402
from endgame_logic import (  # noqa: E402
    EndGameContext,
    classify_public_view,
    evaluate_end_of_term,
    taylor_policy_deviations,
)

PERSONAS = ("good", "hawk", "dove", "careless")
TERM_LENGTH = 16


def score(economy: Economy, start: int) -> tuple[str, str]:
    end = start + TERM_LENGTH
    outcomes = economy.history.entries[start:end]
    decision_states = economy.history.entries[start - 1 : end - 1]
    deviations, _ = taylor_policy_deviations(
        [row.inflation_rate for row in decision_states],
        [row.unemployment_rate for row in decision_states],
        [row.natural_unemployment_rate for row in decision_states],
        [row.equilibrium_real_rate for row in decision_states],
        [row.interest_rate for row in outcomes],
        economy.parameters.inflation_target,
        economy.minimum_interest_rate,
    )
    context = EndGameContext(
        mandate="dual_mandate",
        initial_inflation=decision_states[0].inflation_rate,
        initial_unemployment=decision_states[0].unemployment_rate,
        dual_unemployment_target=economy.parameters.unemployment_target,
        inflation_history=[row.inflation_rate for row in outcomes],
        unemployment_history=[row.unemployment_rate for row in outcomes],
        real_interest_rate_history=[row.real_interest_rate for row in outcomes],
        inflation_target=economy.parameters.inflation_target,
        policy_deviation_history=deviations,
    )
    performance = evaluate_end_of_term(context)["performance"]
    return performance, classify_public_view(deviations)[0]


def simulate(runs: int, seed: int) -> dict:
    seeds = np.random.default_rng(seed).integers(0, 2**32, runs, dtype=np.uint32)
    results = {}
    for persona in PERSONAS:
        terms = [{"performance": Counter(), "stance": Counter()} for _ in range(2)]
        for run_seed in seeds:
            np.random.seed(int(run_seed))
            economy = Economy(difficulty="central_banker")
            economy.cb_persona = persona
            start = len(economy.history.entries)
            for _ in range(2 * TERM_LENGTH):
                economy.adjust_interest_rate_with_taylor()
                economy.simulate_quarter()
            for term in range(2):
                performance, stance = score(economy, start + term * TERM_LENGTH)
                terms[term]["performance"][performance] += 1
                terms[term]["stance"][stance] += 1
        results[persona] = terms
    return {"runs": runs, "turns": 32, "seed": seed, "mandate": "dual_mandate", "results": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(simulate(args.runs, args.seed), indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
