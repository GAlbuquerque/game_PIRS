"""Tests for news coverage of player-triggered policy actions."""

import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "game_PIRS"))

import app
from economy import Economy


def test_unavailable_policy_action_has_no_impact_and_draws_skepticism():
    economy = Economy(difficulty="central_banker")
    economy.reputation = 0.7
    state = SimpleNamespace(economy=economy, news_log=[], in_term_quarter=1)

    with patch.object(app.st, "session_state", state):
        app._trigger_player_event("high_rate_guidance")

    assert economy.player_event_queue == []
    assert "Skepticism" in state.news_log[0]["name"]
    assert "skeptical" in state.news_log[0]["detail"]


def test_available_policy_action_keeps_its_normal_news_and_impact():
    economy = Economy(difficulty="central_banker")
    state = SimpleNamespace(economy=economy, news_log=[], in_term_quarter=1)

    with patch.object(app.st, "session_state", state):
        app._trigger_player_event("quantitative_easing")

    assert economy.player_event_queue
    assert state.news_log[0]["name"] == "Central Bank Launches Asset Purchases"
    assert "skept" not in state.news_log[0]["detail"].lower()
