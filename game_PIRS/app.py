#!/usr/bin/env python3
"""Streamlit web UI for the Policy Interest Rate Simulator."""

import io

import altair as alt
import pandas as pd
import streamlit as st
from collections import defaultdict

from economy import Economy
from endgame_logic import EndGameContext, build_end_of_term_message, mandate_targets
from parameters import EconomyParameters
from settings_code import (
    MODEL_PARAMETER_ORDER,
    decode_settings_code as _decode_settings_code,
    encode_settings_code as _encode_settings_code,
)

APP_TITLE = "Policy Interest Rate Simulator"
PLAYER_START_TURN = 40
OFFSET = 0 # making this positive is messing with turn counter. Not worth it
TERM_LENGTH = 16
SCENARIOS = ["Random", "Stable Economy", "Stagflation", "High Inflation", "Depression"]
MANDATES = {
    "Inflation Target": "inflation_target",
    "Dual Mandate": "dual_mandate",
}
DIFFICULTIES = {
    "Principles": "principles",
    "Senior": "senior",
    "Central Bank Governor": "central_banker",
}
SHOW_START_EXPLAINERS = 1

DIFFICULTY_EXPLAINERS = {
    "principles": (
        "Principles mode simplifies the economy so cause-and-effect is easier to understand. "
        "Natural rate of unemployment is known, the effects of interest rates on employment are immediate, inflation reacts quickly to any change in unemployment, and economic shocks are very rare. "
        "This mode is designed for students taking Principles of Macroeconomics or players learning the basics of monetary policy."
    ),
    "senior": (
        "Senior mode adds more realism and complexity to the economy while keeping the simulation fairly interpretable. "
        "Shocks are common and the impact of your choices is neither immediate, nor short-lived."
        "The natural rate of unemployment is unknown, but you can estimate it observing the past." 
        "This mode is designed for advanced university students and players with a solid understanding of Macroeconomics and Monetary Economics."
    ),
    "central_banker": (
        "Central Banker mode delivers the most realistic and demanding version of the simulator. "
        "Policy lags, economic shocks, and interacting forces can push inflation and unemployment in conflicting directions at the same time. "
        "The natural rate of unemployment is unknown, but you can estimate it observing the past. "
        "This mode is designed for experienced players who want uncertainty, difficult judgment calls, and full-pressure policymaking. "
        "It also unlocks player-triggered policy events: quantitative easing and high- or low-rate forward guidance."
    ),
}

PLAYER_EVENTS = {
    "quantitative_easing": (
        "Quantitative Easing",
        "Central Bank Launches Asset Purchases",
        "The Central Bank has announced a new asset-purchase programme to support demand and ease financial conditions.",
    ),
    "high_rate_guidance": (
        "Announce Future High Rates",
        "Central Bank Signals Higher Rates Ahead",
        "The Central Bank has signalled that interest rates are likely to remain higher in the coming quarters.",
    ),
    "low_rate_guidance": (
        "Announce Future Low Rates",
        "Central Bank Signals Lower Rates Ahead",
        "The Central Bank has signalled that interest rates are likely to remain lower in the coming quarters.",
    ),
}

SCENARIO_EXPLAINERS = {
    "Random": (
        "Random starts from a neutral setup and lets the simulation draw a broad mix of possible developments. "
        "You must diagnose conditions as they evolve. "
        "This mode is ideal if you want replayability and surprise from run to run."
    ),
    "Stable Economy": (
        "Stable Economy begins with generally calm conditions and fewer immediate disruptions. "
        "It is a good scenario for practicing steady, disciplined policy adjustments. "
    ),
    "Stagflation": (
        "Stagflation places you in an environment where inflation pressure and labor weakness can appear together. "
        "Rate increases may help prices while worsening employment, so tradeoffs become sharper than usual. "
        "Use this scenario to practice policy choices under conflicting objectives."
    ),
    "High Inflation": (
        "High Inflation starts with elevated inflation and a policy setting that requires firm stabilization. "
        "You will likely need a credible path to cool inflation without triggering unnecessary economic damage. "
        "This scenario rewards consistency, patience, and clear anti-inflation strategy."
    ),
    "Depression": (
        "Depression begins in severe weakness, where demand is already under strain. "
        "Your main task is to support recovery while preventing secondary instability from compounding the downturn. "
        "Choose this if you want to focus on stabilization in a deeply stressed economy."
    ),
}

MANDATE_EXPLAINERS = {
    "Inflation Target": (
        "Inflation Target focuses your score on price stability as the primary mission. "
        "You can tolerate more labor-market variation if that helps return inflation toward target over time. "
        "This framework is used in countries such as Canada and Brazil, as well as in the Euro area."
    ),
    "Dual Mandate": (
        "Dual Mandate asks you to balance inflation control with employment outcomes at the same time. "
        "Policy moves that improve one side of the economy may weaken the other, making pacing and sequencing central to success. "
        "This framework is most strongly associated with countries such as the United States."
    ),
}

def _sample_scenario(_: str):
    return None


def _activate_player_difficulty(econ: Economy, difficulty: str) -> None:
    econ.set_difficulty(difficulty)


def _apply_scenario_initial_conditions(econ: Economy, scenario_name: str) -> None:
    if scenario_name == "High Inflation":
        econ.indicators.inflation_rate = 20.0
        econ.interest_rate = 6.0
        econ.indicators.unemployment_rate = 3.0
        econ._update_variables()


def _apply_bootstrap_persona(econ: Economy, scenario_name: str) -> None:
    if scenario_name == "Stable Economy":
        econ.cb_persona = "good"
    elif scenario_name == "Stagflation":
        econ.cb_persona = "dove"
    elif scenario_name == "High Inflation":
        econ.cb_persona = "careless"
    elif scenario_name == "Depression":
        econ.cb_persona = "hawk"


def _has_past_event(econ: Economy, event_name: str) -> bool:
    return any(event_name in quarter_events for quarter_events in econ.past_events)


def _force_event_by_name(econ: Economy, scenario_name: str, event_name: str, news_log: list[dict]) -> None:
    event = next((e for e in econ.events if e.name == event_name), None)
    if event is None:
        return
    econ.enqueue_event(event)
    econ.apply_event_effects(dict(econ.effect_queue[0]))
    econ.effect_queue[0] = defaultdict(float)
    econ.past_events.append([event.name])
    econ.past_events = econ.past_events[-8:]
    if econ.current_quarter > OFFSET:
        news_log.append({
            "quarter": econ.current_quarter - OFFSET,
            "in_term_quarter": 0,
            "name": event.name,
            "detail": event.description or "",
            "fired_this_turn": False,
        })


def _force_stagflation_supply_shock(econ: Economy, scenario_name: str, news_log: list[dict]) -> None:
    history = econ.event_history()
    weighted_candidates = []
    for event_name in ["Global Supply Shock", "Pandemic Outbreak", "Natural Disaster"]:
        event = next((e for e in econ.events if e.name == event_name), None)
        if event is None:
            continue
        weight = max(0.0, float(event.get_probability(history)))
        weighted_candidates.append((event.name, weight))
    if not weighted_candidates:
        return
    total_weight = sum(weight for _, weight in weighted_candidates)
    if total_weight <= 0:
        selected_name = weighted_candidates[0][0]
    else:
        import random
        names = [name for name, _ in weighted_candidates]
        weights = [weight for _, weight in weighted_candidates]
        selected_name = random.choices(names, weights=weights, k=1)[0]
    _force_event_by_name(econ, scenario_name, selected_name, news_log)


def _new_game(difficulty: str, scenario_name: str, mandate: str) -> None:
    model_settings = dict(st.session_state.get("model_settings", {}))
    # There is only one expectations anchor: the selected inflation target.
    model_settings["expected_inflation"] = model_settings.get(
        "inflation_target", EconomyParameters().inflation_target
    )
    parameters = EconomyParameters(**model_settings)
    econ = Economy(
        difficulty="central_banker",
        scenario=_sample_scenario(scenario_name),
        parameters=parameters,
        minimum_interest_rate=st.session_state.get("minimum_interest_rate", 0.0),
    )
    econ.offset = OFFSET
    econ.player_start_turn = PLAYER_START_TURN
    _apply_scenario_initial_conditions(econ, scenario_name)

    news_log = []
    total_turns = PLAYER_START_TURN
    _apply_bootstrap_persona(econ, scenario_name)
    hyperinflation_prob_boosted = False
    for idx in range(total_turns):
        if scenario_name == "Stable Economy" and idx >= total_turns - 10:
            econ.last_event_quarter = econ.current_quarter
        if scenario_name == "High Inflation" and not hyperinflation_prob_boosted:
            for event in econ.events:
                if event.name == "Spending Wave":
                    for term in event.prob_terms:
                        if term.label == "a_base":
                            original_fn = term.fn
                            term.fn = lambda h, _f=original_fn: min(1.0, 10 * float(_f(h)))
                            hyperinflation_prob_boosted = True

        econ.adjust_interest_rate_with_taylor()
        result = econ.simulate_quarter()
        three_before_player = total_turns - 3
        if idx == three_before_player:
            if scenario_name == "Depression":
                _force_event_by_name(econ, scenario_name, "Major Financial Crisis", news_log)
            if scenario_name == "Stagflation":
                _force_stagflation_supply_shock(econ, scenario_name, news_log)
        if scenario_name == "High Inflation":
            if not _has_past_event(econ, "Spending Wave"):
                _force_event_by_name(econ, scenario_name, "Spending Wave", news_log)

        if result.get("event") and econ.current_quarter > OFFSET:
            news_log.append({
                "quarter": max(1, econ.current_quarter - OFFSET),
                "in_term_quarter": 0,
                "name": result["event_name"],
                "detail": result.get("event") or "",
                "fired_this_turn": False,
            })

    _activate_player_difficulty(econ, difficulty)

    dual_target = parameters.unemployment_target

    st.session_state.economy = econ
    st.session_state.news_log = news_log[-100:]
    st.session_state.game_over = False
    st.session_state.player_turn = 1
    st.session_state.in_term_quarter = 1
    st.session_state.term_start_idx = max(0, econ.current_quarter - 1)
    st.session_state.initial_inflation = econ.indicators.inflation_rate
    st.session_state.initial_unemployment = econ.indicators.unemployment_rate
    st.session_state.difficulty = difficulty
    st.session_state.scenario_name = scenario_name
    st.session_state.mandate = mandate
    st.session_state.dual_unemployment_target = dual_target
    st.session_state.inflation_target = parameters.inflation_target
    st.session_state.end_message = ""
    st.session_state.graph_window_mode = "full"
    st.session_state.graph_split_mode = False
    st.session_state.show_targets_on_graph = False
    st.session_state.end_summary = None
    st.session_state.game_started = True
    st.session_state.show_end_dialog = False
    st.session_state.latest_fired = False


def _plot_histories(econ: Economy, window_mode: str, split_mode: bool, show_targets: bool, mandate: str, dual_unemployment_target: int, show_news_banner: bool):
    inflation_history = econ.variables.get_history("inflation_rate")
    unemployment_history = econ.variables.get_history("unemployment_rate")
    interest_rate_history = econ.variables.get_history("interest_rate")

    start_idx = max(0, len(inflation_history) - 20) if window_mode == "past20" else 0
    quarters = list(range(start_idx, len(inflation_history)))

    rows = []
    for i, q in enumerate(quarters):
        rows.append({"Quarter": q, "Metric": "Inflation", "Value": inflation_history[start_idx + i], "Panel": "left"})
        rows.append({"Quarter": q, "Metric": "Interest Rate", "Value": interest_rate_history[start_idx + i], "Panel": "left"})
        rows.append({"Quarter": q, "Metric": "Unemployment", "Value": unemployment_history[start_idx + i], "Panel": "right"})

    if econ.difficulty == "principles":
        natural = econ.variables.get_history("natural_unemployment_rate")[start_idx:]
        for i, q in enumerate(quarters):
            rows.append({"Quarter": q, "Metric": "Natural unemployment", "Value": natural[i], "Panel": "right"})

    df = pd.DataFrame(rows)
    palette = {"Inflation": "red", "Unemployment": "blue", "Interest Rate": "green", "Natural unemployment": "black"}

    base = alt.Chart(df).mark_line().encode(
        x=alt.X("Quarter:Q", title="Quarter"),
        y=alt.Y("Value:Q", title="Percent"),
        color=alt.Color("Metric:N", scale=alt.Scale(domain=list(palette.keys()), range=list(palette.values()))),
        strokeDash=alt.condition(alt.datum.Metric == "Interest Rate", alt.value([6, 4]), alt.value([1, 0])),
    )

    player_line = alt.Chart(pd.DataFrame([{"Quarter": PLAYER_START_TURN}])).mark_rule(color="black", strokeDash=[4, 4]).encode(x="Quarter:Q")

    target_layers_left, target_layers_right = [], []
    if show_targets:
        t = mandate_targets(mandate, dual_unemployment_target, econ.parameters.inflation_target)
        target_layers_left.append(alt.Chart(pd.DataFrame([{"Value": t["inflation"]}])).mark_rule(color="red", strokeDash=[2, 2], opacity=0.6).encode(y="Value:Q"))
        if t["unemployment"] is not None:
            target_layers_right.append(alt.Chart(pd.DataFrame([{"Value": t["unemployment"]}])).mark_rule(color="blue", strokeDash=[2, 2], opacity=0.6).encode(y="Value:Q"))

    news_layer = None
    if show_news_banner and quarters:
        max_y = max(inflation_history[start_idx:] + unemployment_history[start_idx:] + interest_rate_history[start_idx:])
        mid_q = quarters[len(quarters) // 2]
        news_layer = alt.Chart(pd.DataFrame([{"Quarter": mid_q, "Value": max_y + 0.8, "Label": "NEWS!"}])).mark_text(
            color="red", fontSize=20, fontWeight="bold", align="center", baseline="top"
        ).encode(x="Quarter:Q", y="Value:Q", text="Label:N")

    if split_mode:
        left_chart = alt.layer(base.transform_filter("datum.Panel == 'left'"), player_line, *target_layers_left).properties(height=220)
        right_layers = [base.transform_filter("datum.Panel == 'right'"), player_line, *target_layers_right]
        if news_layer is not None:
            right_layers.append(news_layer)
        right_chart = alt.layer(*right_layers).properties(height=220)
        return alt.hconcat(left_chart, right_chart).resolve_scale(color='shared')

    layers = [base, player_line, *target_layers_left, *target_layers_right]
    if news_layer is not None:
        layers.append(news_layer)
    return alt.layer(*layers).properties(height=320)


def _event_has_economic_impact(econ: Economy, event_name: str) -> bool:
    if not event_name:
        return False
    event = next((e for e in econ.events if e.name == event_name), None)
    if event is None:
        return False
    for values in event.effects_schedule.values():
        if isinstance(values, list):
            if any(abs(v) > 1e-12 for v in values):
                return True
        elif abs(values) > 1e-12:
            return True
    return False

def _finish_game_if_needed() -> None:
    if st.session_state.in_term_quarter <= TERM_LENGTH:
        return

    st.session_state.game_over = True
    econ = st.session_state.economy
    term_end_idx = econ.current_quarter
    term_start_idx = max(0, term_end_idx - TERM_LENGTH)

    infl_term = econ.variables.get_history("inflation_rate")[term_start_idx:term_end_idx]
    unemp_term = econ.variables.get_history("unemployment_rate")[term_start_idx:term_end_idx]
    real_term = econ.variables.get_history("real_interest_rate")[term_start_idx:term_end_idx]

    term_events_raw = [
        e["name"]
        for e in st.session_state.news_log
        if e.get("in_term_quarter", 0) > 0
        and e["in_term_quarter"] <= TERM_LENGTH
        and _event_has_economic_impact(econ, e.get("name", ""))
    ]
    term_events = list(dict.fromkeys(term_events_raw))

    end_ctx = EndGameContext(
        mandate=st.session_state.mandate,
        initial_inflation=st.session_state.initial_inflation,
        initial_unemployment=st.session_state.initial_unemployment,
        dual_unemployment_target=st.session_state.dual_unemployment_target,
        inflation_history=infl_term,
        unemployment_history=unemp_term,
        real_interest_rate_history=real_term,
        term_event_names=term_events,
        inflation_target=econ.parameters.inflation_target,
    )

    message = build_end_of_term_message(end_ctx)
    st.session_state.end_message = message
    st.session_state.show_end_dialog = True


def _next_quarter(user_rate: float) -> None:
    econ = st.session_state.economy
    econ.adjust_interest_rate(float(user_rate))
    result = econ.simulate_quarter()

    st.session_state.latest_fired = bool(result.get("event_name"))
    if st.session_state.latest_fired:
        st.session_state.news_log.append({
            "quarter": max(1, econ.current_quarter - OFFSET),
            "in_term_quarter": st.session_state.in_term_quarter,
            "name": result["event_name"],
            "detail": result.get("event") or "",
            "fired_this_turn": True,
        })
        st.session_state.news_log = st.session_state.news_log[-100:]

    st.session_state.player_turn += 1
    st.session_state.in_term_quarter += 1
    _finish_game_if_needed()


def _trigger_player_event(event_name: str) -> None:
    """Trigger an available action without advancing the quarter."""
    econ = st.session_state.economy
    succeeded, _ = econ.trigger_player_event(event_name)
    if not succeeded:
        return
    _, headline, detail = PLAYER_EVENTS[event_name]
    st.session_state.news_log.append({
        "quarter": max(1, econ.current_quarter - OFFSET),
        "in_term_quarter": st.session_state.in_term_quarter,
        "name": headline,
        "detail": detail,
        "fired_this_turn": True,
    })
    st.session_state.news_log = st.session_state.news_log[-100:]


def _render_end_dialog() -> None:
    if not st.session_state.get("show_end_dialog", False):
        return

    @st.dialog("End of Term")
    def _dlg():
        st.write(st.session_state.end_message)
        c1, c2 = st.columns(2)
        if c1.button("Continue Playing", width="stretch"):
            st.session_state.game_over = False
            st.session_state.show_end_dialog = False
            st.session_state.in_term_quarter = 1
            st.rerun()
        if c2.button("Retire", width="stretch"):
            st.session_state.show_end_dialog = False
            st.rerun()

    _dlg()


def _render_start_page() -> None:
    left_col, right_col = st.columns([1.0, 1.0]) if SHOW_START_EXPLAINERS == 1 else (st.container(), None)

    with left_col:
        st.markdown("### Start Menu")
        difficulty_label = st.radio("Difficulty", list(DIFFICULTIES.keys()), index=2, key="start_difficulty")
        difficulty = DIFFICULTIES[difficulty_label]
        scenario_name = st.radio("Scenario", SCENARIOS, index=0, key="start_scenario")
        mandate_label = st.radio("Mandate", list(MANDATES.keys()), index=0, key="start_mandate")

        button_col, _ = st.columns([0.42, 0.58])
        with button_col:
            if st.button("Start Game", type="primary", width="stretch"):
                _new_game(difficulty, scenario_name, MANDATES[mandate_label])
                st.rerun()
            if st.button("Advanced Settings", width="stretch"):
                st.session_state.start_page = "settings"
                st.rerun()

    if SHOW_START_EXPLAINERS == 1 and right_col is not None:
        with right_col:
            st.markdown("### Setup Explainer")
            st.markdown(f"**Difficulty:** {DIFFICULTY_EXPLAINERS[difficulty]}")
            st.markdown(f"**Scenario:** {SCENARIO_EXPLAINERS[scenario_name]}")
            st.markdown(f"**Mandate:** {MANDATE_EXPLAINERS[mandate_label]}")

PARAMETER_GROUPS = {
    "Output gap and monetary transmission": [
        ("interest_rate_pressure_persistence", "Interest-pressure persistence (rho)"),
        ("output_gap_expectation_persistence", "Expected output-gap persistence (phi)"),
        ("intertemporal_elasticity_inverse", "Modified IS coefficient (sigma tilde)"),
    ],
    "Inflation and unemployment": [
        (
            "inflation_expectation_discount",
            "Temporal preference / discount factor (beta)",
        ),
        ("phillips_output_gap", "Phillips-curve slope (k)"),
        ("okun_coefficient", "Okun coefficient"),
        ("minimum_inflation", "Minimum inflation"),
        ("minimum_unemployment", "Minimum unemployment"),
    ],
    "Expectations & targets": [
        ("inflation_target", "Inflation target"),
        ("reputation_expectation_coefficient", "Reputation impact coefficient (k_a)"),
        ("unemployment_target", "Unemployment target"),
    ],
    "Events": [
        ("event_probability_scale", "Event probability multiplier"),
    ],
    "Background economy & shocks": [
        ("natural_unemployment_anchor", "Natural-unemployment anchor"),
        ("natural_unemployment_reversion", "Natural-rate reversion speed"),
        ("minimum_natural_unemployment", "Minimum natural unemployment"),
        ("equilibrium_real_rate_anchor", "Equilibrium real-rate anchor"),
        ("equilibrium_real_rate_reversion", "Equilibrium real-rate reversion speed"),
    ],
}
def _apply_settings_code_from_state() -> None:
    """Apply an entered calibration before keyed widgets are rendered again."""
    try:
        loaded = _decode_settings_code(st.session_state.settings_code_input)
    except ValueError as exc:
        st.session_state.settings_code_error = str(exc)
        st.session_state.settings_code_success = None
        return

    st.session_state.model_settings = loaded
    st.session_state.settings_simulation = None
    for name, value in loaded.items():
        if name == "shock_std_devs":
            for index, shock_value in enumerate(value):
                st.session_state[f"setting_shock_{index}"] = shock_value
        elif name == "unemployment_target":
            st.session_state[f"setting_{name}"] = str(value)
            st.session_state[f"setting_{name}_mode"] = "Other"
        elif name == "inflation_target":
            st.session_state[f"setting_{name}"] = str(value)
            st.session_state[f"setting_{name}_mode"] = "Other"
        else:
            st.session_state[f"setting_{name}"] = value
    st.session_state.settings_code_error = None
    st.session_state.settings_code_success = (
        "Calibration code applied. The editor now shows the decoded values."
    )

PARAMETER_EQUATIONS = {
    "Output gap and monetary transmission": (
        r"R_t=\rho R_{t-1}+(1-\rho)(r_{t-1}-r^n_{t-1}),\qquad "
        r"\widetilde y_t=\phi\widetilde y_{t-1}-R_t/\widetilde\sigma+\varepsilon_t^d",
        "The real-rate gap selected last quarter enters output with a lag; existing "
        "output gaps are expected to shrink at a rate controlled by phi.",
    ),
    "Inflation and unemployment": (
        r"\pi_t^{raw}=\beta\pi_t^e+\kappa_t\widetilde y_t+\varepsilon_t^\pi,\qquad "
        r"u_t=u_t^n-\lambda_u\widetilde y_t",
        "Beta is the temporal-preference (discount) factor: it determines how strongly "
        "expected future inflation affects inflation today. The same Phillips slope "
        "applies during inflation and disinflation, and Okun's law "
        "translates the output gap into unemployment.",
    ),
    "Expectations & targets": (
        r"\pi_t^e=\alpha\pi^*+(1-\alpha)\pi_{t-1},\qquad "
        r"\alpha=A_t k_a",
        "Better central-bank reputation gives the inflation target more weight; otherwise "
        "expectations remain closer to last quarter's inflation.",
    ),
    "Events": (
        r"P(\text{event})=\operatorname{clip}(s_{event}P_0,0,1)",
        "The multiplier scales each eligible event's probability. Zero disables random "
        "events; 2 doubles their underlying probabilities up to 100%.",
    ),
    "Background economy & shocks": (
        r"u_t^n=u_{t-1}^n-\rho_u(u_{t-1}^n-\bar u^n)+\varepsilon_t^u",
        "Natural unemployment drifts toward its anchor while random shocks move the four "
        "economic processes each quarter.",
    ),
}


def _simulate_settings(
    parameters: EconomyParameters,
    runs=100,
    turns=100,
    initialization_turns=40,
    scenario_name="Random",
    persona="good",
) -> dict:
    """Batch-test a calibration using the legacy automated-policy specification."""
    rows = []
    events_fired = 0
    for run in range(runs):
        econ = Economy(
            difficulty="central_banker",
            scenario=_sample_scenario(scenario_name),
            parameters=parameters,
        )
        _apply_scenario_initial_conditions(econ, scenario_name)
        _apply_bootstrap_persona(econ, scenario_name)
        preview_news = []
        hyperinflation_prob_boosted = False
        for initialization_index in range(initialization_turns):
            if (
                scenario_name == "Stable Economy"
                and initialization_index >= initialization_turns - 10
            ):
                econ.last_event_quarter = econ.current_quarter
            if scenario_name == "High Inflation" and not hyperinflation_prob_boosted:
                for event in econ.events:
                    if event.name != "Spending Wave":
                        continue
                    for term in event.prob_terms:
                        if term.label == "a_base":
                            original_fn = term.fn
                            term.fn = lambda history, fn=original_fn: min(
                                1.0, 10 * float(fn(history))
                            )
                            hyperinflation_prob_boosted = True
            econ.adjust_interest_rate_with_taylor()
            econ.simulate_quarter()
            rows.append({
                "Run": run + 1,
                "Quarter": initialization_index + 1,
                "Phase": "Pre-player",
                "Inflation": econ.indicators.inflation_rate,
                "Unemployment": econ.indicators.unemployment_rate,
                "Natural unemployment": econ.indicators.natural_unemployment_rate,
                "Interest rate": econ.interest_rate,
                "Reputation": econ.reputation,
            })
            if initialization_index == initialization_turns - 3:
                if scenario_name == "Depression":
                    _force_event_by_name(
                        econ, scenario_name, "Major Financial Crisis", preview_news
                    )
                elif scenario_name == "Stagflation":
                    _force_stagflation_supply_shock(
                        econ, scenario_name, preview_news
                    )
            if scenario_name == "High Inflation" and not _has_past_event(
                econ, "Spending Wave"
            ):
                _force_event_by_name(
                    econ, scenario_name, "Spending Wave", preview_news
                )

        # The selected persona substitutes for the player only after the
        # initialization period, matching the legacy batch simulator.
        econ.cb_persona = persona
        for turn in range(turns):
            econ.adjust_interest_rate_with_taylor()
            result = econ.simulate_quarter()
            events_fired += bool(result.get("event_name"))
            rows.append({
                "Run": run + 1,
                "Turn": turn + 1,
                "Quarter": initialization_turns + turn + 1,
                "Phase": "Player substitute",
                "Inflation": econ.indicators.inflation_rate,
                "Unemployment": econ.indicators.unemployment_rate,
                "Natural unemployment": econ.indicators.natural_unemployment_rate,
                "Interest rate": econ.interest_rate,
                "Reputation": econ.reputation,
            })
    frame = pd.DataFrame(rows)
    return {
        "frame": frame,
        "runs": runs,
        "turns": turns,
        "initialization_turns": initialization_turns,
        "scenario_name": scenario_name,
        "persona": persona,
        "event_rate": events_fired / (runs * turns),
    }


def _render_simulation_result(result: dict) -> None:
    """Show a compact outcome summary for a settings-page batch test."""
    frame = result["frame"]
    st.markdown("### Simulation preview")
    st.caption(
        f"{result['runs']} runs × {result['turns']} evaluated quarters, after "
        f"{result['initialization_turns']} initialization quarters. "
        f"Scenario: {result['scenario_name']}; player substitute: "
        f"{result['persona'].replace('_', ' ').title()}."
    )
    player_frame = frame[frame["Phase"] == "Player substitute"]
    metric_cols = st.columns(4)
    metric_cols[0].metric("Mean inflation", f"{player_frame['Inflation'].mean():.2f}%")
    metric_cols[1].metric("Mean unemployment", f"{player_frame['Unemployment'].mean():.2f}%")
    metric_cols[2].metric("Mean interest rate", f"{player_frame['Interest rate'].mean():.2f}%")
    metric_cols[3].metric("Events per quarter", f"{result['event_rate']:.1%}")

    show_natural_unemployment = st.toggle(
        "Show natural unemployment",
        key="settings_preview_natural_unemployment",
    )
    show_reputation = st.toggle(
        "Show reputation evolution",
        key="settings_preview_reputation",
        help="Display the simulated central bank reputation score over time.",
    )
    chart_indicators = ["Inflation", "Unemployment", "Interest rate"]
    if show_natural_unemployment:
        chart_indicators.append("Natural unemployment")
    long_frame = frame.melt(
        ["Run", "Quarter", "Phase"],
        value_vars=chart_indicators,
        var_name="Indicator",
        value_name="Percent",
    )
    chart_data = long_frame.groupby(["Quarter", "Indicator"])["Percent"].agg(
        Mean="mean",
        Bottom_5=lambda values: values.quantile(0.05),
        Top_5=lambda values: values.quantile(0.95),
    ).reset_index()
    split_chart = st.toggle("Split chart mode", key="settings_preview_split")
    base = alt.Chart(chart_data).encode(
        x=alt.X("Quarter:Q", title="Quarter"),
        color="Indicator:N",
    )
    mean_line = base.mark_line(strokeWidth=2.5).encode(
        y=alt.Y("Mean:Q", title="Percent")
    )
    lower_line = base.mark_line(strokeWidth=1, opacity=0.3, strokeDash=[4, 3]).encode(
        y=alt.Y("Bottom_5:Q", title="Percent")
    )
    upper_line = base.mark_line(strokeWidth=1, opacity=0.3, strokeDash=[4, 3]).encode(
        y=alt.Y("Top_5:Q", title="Percent")
    )
    player_line = alt.Chart(chart_data).mark_rule(
        color="black", strokeDash=[4, 4]
    ).encode(x=alt.datum(result["initialization_turns"]))
    chart = alt.layer(lower_line, upper_line, mean_line, player_line)
    if split_chart:
        chart = chart.facet(
            column=alt.Column("Indicator:N", title=None),
        ).resolve_scale(y="independent")
    st.altair_chart(chart, width="stretch")
    if show_reputation:
        reputation_data = frame.groupby("Quarter")["Reputation"].agg(
            Mean="mean",
            Bottom_5=lambda values: values.quantile(0.05),
            Top_5=lambda values: values.quantile(0.95),
        ).reset_index()
        reputation_base = alt.Chart(reputation_data).encode(
            x=alt.X("Quarter:Q", title="Quarter"),
        )
        reputation_chart = alt.layer(
            reputation_base.mark_line(
                strokeWidth=1, opacity=0.3, strokeDash=[4, 3]
            ).encode(y=alt.Y("Bottom_5:Q", title="Reputation", scale=alt.Scale(domain=[0, 1]))),
            reputation_base.mark_line(
                strokeWidth=1, opacity=0.3, strokeDash=[4, 3]
            ).encode(y=alt.Y("Top_5:Q", title="Reputation", scale=alt.Scale(domain=[0, 1]))),
            reputation_base.mark_line(strokeWidth=2.5, color="#7b2cbf").encode(
                y=alt.Y("Mean:Q", title="Reputation", scale=alt.Scale(domain=[0, 1]))
            ),
            alt.Chart(reputation_data).mark_rule(
                color="black", strokeDash=[4, 4]
            ).encode(x=alt.datum(result["initialization_turns"])),
        ).properties(title="Central bank reputation evolution", height=180)
        st.altair_chart(reputation_chart, width="stretch")
    st.caption(
        "Solid lines are averages. The lighter dashed lines mark the bottom and top "
        "5% of simulated outcomes. The black dashed line marks when the selected "
        "player substitute assumes control."
    )


def _render_settings_page() -> None:
    """Render advanced setup and launch a game with the displayed values."""
    defaults = EconomyParameters()
    saved = st.session_state.get("model_settings", {})
    st.markdown("### Advanced Settings")
    minimum_interest_rate = st.number_input(
        "Minimum interest rate allowed (%)",
        value=float(st.session_state.get("minimum_interest_rate", 0.0)),
        step=0.25,
        format="%.2f",
        key="setting_minimum_interest_rate",
    )
    st.warning(
        "The model may behave weirdly with very negative interest rates. Very few "
        "countries have tried rates between 0% and -1%."
    )

    setup_cols = st.columns(3)
    initial_difficulty = st.session_state.get(
        "advanced_difficulty", st.session_state.get("start_difficulty", "Central Bank Governor")
    )
    difficulty_label = setup_cols[0].selectbox(
        "Difficulty", list(DIFFICULTIES), index=list(DIFFICULTIES).index(initial_difficulty),
        key="advanced_difficulty",
    )
    setup_cols[0].caption(
        "Player-triggered policy events are enabled only at Central Banker difficulty."
    )
    initial_scenario = st.session_state.get(
        "advanced_scenario", st.session_state.get("start_scenario", SCENARIOS[0])
    )
    scenario_name = setup_cols[1].selectbox(
        "Scenario", SCENARIOS, index=SCENARIOS.index(initial_scenario),
        key="advanced_scenario",
    )
    mandate_labels = list(MANDATES)
    initial_mandate = st.session_state.get(
        "advanced_mandate", st.session_state.get("start_mandate", mandate_labels[0])
    )
    mandate_label = setup_cols[2].selectbox(
        "Mandate", mandate_labels, index=mandate_labels.index(initial_mandate),
        key="advanced_mandate",
    )
    st.caption(
        "Edit the calibration used for this game. Default values are shown in "
        "parentheses; select Other when you want to enter a different target."
    )

    # Do not put the calibration editor in a Streamlit form. Forms deliberately
    # defer widget updates until a submit button is pressed, which left the
    # password below showing the previous calibration while users were editing.
    with st.container():
        edited = {}
        validation_errors = []
        columns = st.columns(2)
        for group_index, (group_name, fields) in enumerate(PARAMETER_GROUPS.items()):
            with columns[group_index % 2]:
                with st.container(border=True):
                    st.markdown(f"#### {group_name}")
                    equation, explanation = PARAMETER_EQUATIONS[group_name]
                    st.latex(equation)
                    st.caption(explanation)
                    for field_name, label in fields:
                        default = saved.get(field_name, getattr(defaults, field_name))
                        if field_name == "inflation_target":
                            inflation_mode = st.radio(
                                label,
                                ["Default (2%)", "Other"],
                                horizontal=True,
                                key="setting_inflation_target_mode",
                            )
                            target_text = st.text_input(
                                "Other inflation target (%)",
                                value=str(default),
                                key="setting_inflation_target",
                                disabled=inflation_mode != "Other",
                            )
                            if inflation_mode == "Other":
                                try:
                                    edited[field_name] = float(target_text)
                                    if edited[field_name] < 0:
                                        validation_errors.append("Inflation target cannot be negative.")
                                except ValueError:
                                    edited[field_name] = defaults.inflation_target
                                    validation_errors.append("Inflation target must be a number.")
                            else:
                                edited[field_name] = defaults.inflation_target
                            continue
                        if field_name == "unemployment_target":
                            unemployment_mode = st.radio(
                                label,
                                ["Default (4%)", "Other"],
                                horizontal=True,
                                key="setting_unemployment_target_mode",
                            )
                            target_text = st.text_input(
                                "Other unemployment target (%)",
                                value=str(default if default is not None else defaults.unemployment_target),
                                key="setting_unemployment_target",
                                disabled=(unemployment_mode != "Other" or mandate_label != "Dual Mandate"),
                            )
                            if unemployment_mode == "Other" and mandate_label == "Dual Mandate":
                                try:
                                    edited[field_name] = float(target_text)
                                    if edited[field_name] < 0:
                                        validation_errors.append("Unemployment target cannot be negative.")
                                except ValueError:
                                    edited[field_name] = defaults.unemployment_target
                                    validation_errors.append("Unemployment target must be a number.")
                            else:
                                edited[field_name] = defaults.unemployment_target
                            continue
                        is_integer = False
                        bounded_ratio_fields = {
                            "interest_rate_pressure_persistence",
                            "output_gap_expectation_persistence",
                            "equilibrium_real_rate_reversion",
                            "inflation_expectation_discount",
                            "reputation_expectation_coefficient",
                        }
                        strictly_positive_fields = {
                            "intertemporal_elasticity_inverse",
                            "inflation_expectation_discount",
                            "okun_coefficient",
                        }
                        minimum = (
                            0.000001
                            if field_name in strictly_positive_fields
                            else 0.0
                            if field_name == "event_probability_scale"
                            or field_name in bounded_ratio_fields
                            else None
                        )
                        maximum = 1.0 if field_name in bounded_ratio_fields else None
                        edited[field_name] = st.number_input(
                            f"{label} (default: {getattr(defaults, field_name):g})",
                            value=default,
                            min_value=minimum,
                            max_value=maximum,
                            step=1 if is_integer else None,
                            format="%d" if is_integer else "%.6g",
                            key=f"setting_{field_name}",
                        )

        with st.container(border=True):
            st.markdown("#### Shock standard deviations")
            st.caption("Quarterly volatility for inflation, demand, the natural unemployment rate, and the equilibrium real rate.")
            shock_defaults = saved.get("shock_std_devs", defaults.shock_std_devs)
            shock_cols = st.columns(4)
            shock_labels = tuple(
                f"{label} (default: {defaults.shock_std_devs[index]:g})"
                for index, label in enumerate(("Inflation", "Demand", "Natural rate", "Equilibrium rate"))
            )
            shock_values = [
                col.number_input(label, min_value=0.0, value=float(shock_defaults[index]), format="%.6g", key=f"setting_shock_{index}")
                for index, (col, label) in enumerate(zip(shock_cols, shock_labels))
            ]

        with st.container(border=True):
            st.markdown("#### Simulation test")
            st.caption(
                "Choose the batch size for the preview. Simulations use the values "
                "currently in this form without saving them."
            )
            simulation_cols = st.columns(3)
            preview_runs = simulation_cols[0].number_input(
                "Number of simulations", min_value=1, value=100,
                step=1, key="settings_preview_runs"
            )
            preview_turns = simulation_cols[1].number_input(
                "Evaluated quarters", min_value=1, value=100,
                step=1, key="settings_preview_turns"
            )
            initialization_turns = simulation_cols[2].number_input(
                "Initialization quarters", min_value=0, value=40, step=1,
                key="settings_preview_initialization_turns",
            )
            choice_cols = st.columns(2)
            preview_scenario = choice_cols[0].selectbox("Scenario", SCENARIOS)
            persona_labels = {
                "Balanced": "good",
                "Dove": "dove",
                "Hawk": "hawk",
                "Careless": "careless",
            }
            preview_persona_label = choice_cols[1].selectbox(
                "Player substitute persona", list(persona_labels)
            )
            st.caption(
                "The scenario's automated central bank runs initialization. The chosen "
                "persona replaces the player only for the evaluated quarters."
            )

        edited["shock_std_devs"] = tuple(shock_values)
        edited["expected_inflation"] = edited["inflation_target"]
        play_col, simulate_col, reset_col, cancel_col = st.columns(4)
        play = play_col.button("Play", type="primary", width="stretch")
        simulate = simulate_col.button("Simulate", width="stretch")
        reset = reset_col.button("Restore defaults", width="stretch")
        cancel = cancel_col.button("Cancel", width="stretch")

    if play:
        if validation_errors:
            for error in validation_errors:
                st.error(error)
            return
        st.session_state.model_settings = edited
        st.session_state.minimum_interest_rate = float(minimum_interest_rate)
        _new_game(DIFFICULTIES[difficulty_label], scenario_name, MANDATES[mandate_label])
        st.rerun()
    if reset:
        st.session_state.model_settings = {}
        st.session_state.minimum_interest_rate = 0.0
        st.session_state.settings_simulation = None
        for key in list(st.session_state):
            if key.startswith("setting_"):
                del st.session_state[key]
        st.rerun()
    if cancel:
        st.session_state.start_page = "menu"
        st.rerun()
    if simulate:
        try:
            with st.spinner("Testing this calibration..."):
                st.session_state.settings_simulation = _simulate_settings(
                    EconomyParameters(**edited),
                    runs=int(preview_runs),
                    turns=int(preview_turns),
                    initialization_turns=int(initialization_turns),
                    scenario_name=preview_scenario,
                    persona=persona_labels[preview_persona_label],
                )
        except (ValueError, RuntimeError, ArithmeticError) as exc:
            st.error(f"This calibration could not be simulated: {exc}")

    if st.session_state.get("settings_simulation") is not None:
        _render_simulation_result(st.session_state.settings_simulation)

    st.markdown("#### Calibration password")
    st.caption(
        "This code is a direct field-by-field map of the calibration. Each setting name "
        "and value is visible in the JSON after `PIRS2:`. The same calibration always "
        "produces the same code; the game does not upload or store it."
    )
    st.code(_encode_settings_code(edited), language=None, wrap_lines=True)
    code_col, apply_col = st.columns([3, 1])
    entered_code = code_col.text_input(
        "Return to saved settings",
        placeholder="Paste a PIRS2:{…} calibration code",
        key="settings_code_input",
    )
    apply_col.button(
        "Apply code",
        width="stretch",
        disabled=not entered_code,
        on_click=_apply_settings_code_from_state,
    )
    if st.session_state.get("settings_code_error"):
        st.error(f"Could not apply this code: {st.session_state.settings_code_error}")
    if st.session_state.get("settings_code_success"):
        st.success(st.session_state.settings_code_success)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.markdown("""<style>.block-container {padding-top: 3rem;}</style>""", unsafe_allow_html=True)
    st.title(APP_TITLE)
    if "game_started" not in st.session_state:
        st.session_state.game_started = False
    if "start_page" not in st.session_state:
        st.session_state.start_page = "menu"

    if not st.session_state.game_started:
        if st.session_state.start_page == "settings":
            _render_settings_page()
        else:
            _render_start_page()
        return

    if "economy" not in st.session_state:
        _new_game("central_banker", "Random", "inflation_target")

    _render_end_dialog()
    econ = st.session_state.economy
    state = econ.get_state()

    outer_left, outer_right = st.columns([1.1, 2.2])

    with outer_left:
        st.markdown("### News Feed")
        #top_panel_height = 220
        news_container = st.container(height=687, border=True)
        with news_container:
            if st.session_state.news_log:
                for idx, item in enumerate(list(reversed(st.session_state.news_log))):
                    color = "red" if idx == 0 and st.session_state.latest_fired else "inherit"
                    label = f"Q{item['quarter']}: {item['name']}"
                    st.markdown(f"<div style='color:{color};font-weight:600'>{label}</div>", unsafe_allow_html=True)
                    if item.get("detail"):
                        with st.expander(f"▶ Details", expanded=False):
                            st.write(item["detail"])
            else:
                st.write("No events yet.")

    with outer_right:
        st.markdown("### Economic Indicators")
       # indicators_container = st.container(height=top_panel_height, border=False)        
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Inflation Rate:** {state['inflation_rate']:.2f}%")
        c2.markdown(f"**Unemployment Rate:** {state['unemployment_rate']:.2f}%")
        c3.markdown(f"**Interest Rate:** {state['interest_rate']:.2f}%")

        st.markdown("### Time Series")
        graph_container = st.container(height=375, border=False)
        with graph_container:
            g1, g2, g3, g4 = st.columns(4)
            st.session_state.graph_window_mode = "past20" if g1.toggle("Past 20 turns", value=(st.session_state.graph_window_mode == "past20")) else "full"
            st.session_state.graph_split_mode = g2.toggle("Split charts", value=st.session_state.graph_split_mode)
            st.session_state.show_targets_on_graph = g3.toggle("Show targets", value=st.session_state.show_targets_on_graph)

            chart = _plot_histories(econ, st.session_state.graph_window_mode, st.session_state.graph_split_mode, st.session_state.show_targets_on_graph, st.session_state.mandate, st.session_state.dual_unemployment_target, st.session_state.latest_fired)
            #with g4:
             #   try:
              #      chart_png = _chart_png_bytes(chart)
               #     st.download_button("Download graph (PNG)", data=chart_png, file_name="economic_graph.png", mime="image/png", width="stretch")
                #except Exception:
                 #   chart_html = chart.to_html().encode("utf-8")
                  #  st.download_button("Download graph (HTML)", data=chart_html, file_name="economic_graph.html", mime="text/html", width="stretch")
            st.altair_chart(chart, width="stretch")

        st.markdown("##### New Interest Rate")
        if "rate_text" not in st.session_state:
            st.session_state.rate_text = f"{state['interest_rate']:.2f}"

        user_rate_text = st.text_input(
            "New Interest Rate_invisible",
            key="rate_text",
            label_visibility="collapsed",
        )
        other_policies_column, next_column = st.columns([1, 3])
        with other_policies_column:
            if econ.difficulty == "central_banker":
                with st.popover("Other Policies", use_container_width=True):
                    for event_name, (label, _, _) in PLAYER_EVENTS.items():
                        available, _ = econ.player_event_status(event_name)
                        if st.button(
                            label,
                            key=f"player_event_{event_name}",
                            disabled=not available,
                            width="stretch",
                        ):
                            _trigger_player_event(event_name)
                            st.rerun()
            else:
                st.button("Other Policies", disabled=True, width="stretch")
        with next_column:
            submitted = st.button(
                "Next",
                type="primary",
                width="stretch",
                disabled=st.session_state.game_over,
            )

        if submitted:
            st.session_state.rate_text = user_rate_text
            try:
                user_rate = float(user_rate_text)
            except ValueError:
                st.error("Please enter a valid number for the interest rate.")
                return
            minimum_rate = st.session_state.get("minimum_interest_rate", 0.0)
            if user_rate < minimum_rate:
                st.error(f"Interest rate cannot be below {minimum_rate:.2f}%.")
                return
            _next_quarter(user_rate)
            st.session_state.rate_text = f"{st.session_state.economy.interest_rate:.2f}"
            st.rerun()


if __name__ == "__main__":
    main()
